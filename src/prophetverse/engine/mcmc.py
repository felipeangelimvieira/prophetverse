"""Numpyro inference engines for prophet models.

The classes in this module take a model, the data and perform inference using Numpyro.
"""

from operator import attrgetter
from typing import Dict, Tuple

import jax.numpy as jnp
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_mean

from prophetverse.engine.base import BaseInferenceEngine
from prophetverse.engine.utils import assert_mcmc_converged


class MCMCInferenceEngine(BaseInferenceEngine):
    """
    Perform MCMC (Markov Chain Monte Carlo) inference for a given model.

    Parameters
    ----------
    model : Callable
        The model function to perform inference on.
    num_samples : int
        The number of MCMC samples to draw.
    num_warmup : int
        The number of warmup samples to discard.
    num_chains : int
        The number of MCMC chains to run in parallel.
    dense_mass : bool
        Whether to use dense mass matrix for NUTS sampler.
    rng_key : Optional
        The random number generator key.
    r_hat: Optional
        The required r_hat for considering the chains to have converged.

    Attributes
    ----------
    num_samples : int
        The number of MCMC samples to draw.
    num_warmup : int
        The number of warmup samples to discard.
    num_chains : int
        The number of MCMC chains to run in parallel.
    dense_mass : bool
        Whether to use dense mass matrix for NUTS sampler.
    mcmc_ : MCMC
        The MCMC object used for inference.
    posterior_samples_ : Dict[str, np.ndarray]
        The posterior samples obtained from MCMC.
    samples_predictive_ : Dict[str, np.ndarray]
        The predictive samples obtained from MCMC.
    samples_ : Dict[str, np.ndarray]
        The MCMC samples obtained from MCMC.
    """

    _tags = {
        "inference_method": "mcmc",
    }

    def __init__(
        self,
        num_samples=1000,
        num_warmup=200,
        num_chains=1,
        dense_mass=False,
        rng_key=None,
        r_hat: float = None,
    ):
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.dense_mass = dense_mass
        self.r_hat = r_hat

        self.summary_ = None

        super().__init__(rng_key)

    def _infer(self, **kwargs):
        """
        Run MCMC inference.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed to the MCMC run method.

        Returns
        -------
        self
            The MCMCInferenceEngine object.
        """

        def get_posterior_samples(
            rng_key,
            model,
            dense_mass,
            init_strategy,
            num_samples,
            num_warmup,
            num_chains,
            **kwargs
        ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
            mcmc_ = MCMC(
                NUTS(model, dense_mass=dense_mass, init_strategy=init_strategy),
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
            )
            mcmc_.run(rng_key, **kwargs)

            group_by_chain = True
            samples = mcmc_.get_samples(group_by_chain=group_by_chain)

            # NB: we fetch sample sites to avoid calculating convergence
            # check on deterministic sites, basically same approach
            # as in `mcmc.print_summary`.
            sites = attrgetter(mcmc_._sample_field)(mcmc_._last_state)

            # NB: we keep it simple and only calculate a summary check whenever it
            # satisfies the requirements of split_gelman_rubin. As it's not likely
            # users will use less than four samples.
            if num_samples >= 4:
                filtered_samples = {k: v for k, v in samples.items() if k in sites}
                summary_ = summary(filtered_samples, group_by_chain=group_by_chain)
            else:
                summary_ = {}

            flattened_samples = {
                k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()
            }

            return flattened_samples, summary_

        self.posterior_samples_, self.summary_ = get_posterior_samples(
            self._rng_key,
            self.model_,
            self.dense_mass,
            init_strategy=init_to_mean,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            num_chains=self.num_chains,
            **kwargs
        )

        if self.r_hat and self.summary_:
            assert_mcmc_converged(self.summary_, self.r_hat)

        return self

    def _predict(self, **kwargs):
        """
        Generate predictive samples.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be passed to the Predictive method.

        Returns
        -------
        Dict[str, np.ndarray]
            The predictive samples.
        """
        num_samples = int(self.num_samples * self.num_chains)

        predictive = Predictive(
            self.model_, self.posterior_samples_, num_samples=num_samples
        )

        self.samples_predictive_ = predictive(self._rng_key, **kwargs)
        return self.samples_predictive_
