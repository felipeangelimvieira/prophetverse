"""Numpyro inference engines for prophet models.

The classes in this module take a model, the data and perform inference using Numpyro.
"""

from functools import partial
from operator import attrgetter
from typing import Callable, Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.random import PRNGKey, split
from numpyro.diagnostics import summary
from numpyro.handlers import condition
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_mean
from numpyro.infer.mcmc import MCMCKernel

from prophetverse.engine.base import BaseInferenceEngine
from prophetverse.engine.utils import assert_mcmc_converged


def _get_posterior_samples(
    rng_key, kernel, num_samples, num_warmup, num_chains, progress_bar, **kw
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Dict[str, jnp.ndarray]]]:
    mcmc_ = MCMC(
        kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc_.run(rng_key, **kw)

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

    flattened_samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}

    return flattened_samples, summary_


# NB: we use separate function s.t. we may vmap over it
def _update_posterior(
    rng_key,
    model,
    kernel_builder,
    to_condition_on,
    num_samples,
    num_warmup,
    num_chains,
    progress_bar,
    **kwargs,
):
    conditioned_model = condition(model, data=to_condition_on)
    kernel = kernel_builder(conditioned_model)

    update_key, _ = split(rng_key)

    # TODO: support full mode by sampling over full posterior, i.e. generate one
    #  posterior sample per posterior sample
    update_samples, _ = _get_posterior_samples(
        update_key,
        kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        progress_bar=progress_bar,
        **kwargs,
    )

    return update_samples


class MCMCInferenceEngine(BaseInferenceEngine):
    """
    Perform MCMC (Markov Chain Monte Carlo) inference for a given model.

    Parameters
    ----------
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
    progress_bar : bool
        Whether to show a progress bar during MCMC sampling.

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
    posterior_samples_ : Dict[str, np.ndarray]
        The posterior samples obtained from MCMC.
    samples_predictive_ : Dict[str, np.ndarray]
        The predictive samples obtained from MCMC.
    summary_ : Dict[str, Dict[str, np.ndarray]]
        Summary of statistics for the posterior samples.
    """

    _tags = {
        "inference_method": "mcmc",
    }

    def __init__(
        self,
        num_samples=1000,
        num_warmup=200,
        num_chains=1,
        dense_mass: Union[bool, List[Tuple[str, ...]]] = False,
        rng_key: PRNGKey = None,
        r_hat: Union[float, None] = None,
        progress_bar: bool = True,
    ):
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.dense_mass = dense_mass
        self.r_hat = r_hat
        self.progress_bar = progress_bar

        self.summary_ = None

        super().__init__(rng_key)

    def build_kernel(self, model: Callable) -> MCMCKernel:
        """
        Build the MCMC kernel.

        Parameters
        ----------
        model : Callable
            The model function to perform inference on.

        Returns
        -------
        MCMCKernel
            The MCMC kernel for the model.
        """
        return NUTS(
            model,
            init_strategy=init_to_mean,
            dense_mass=self.dense_mass,
        )

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
        kernel = self.build_kernel(self.model_)
        self.posterior_samples_, self.summary_ = _get_posterior_samples(
            self._rng_key,
            kernel,
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
            **kwargs,
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

    def _update(self, site_names, mode="mean", **kwargs):
        assert (
            self.posterior_samples_ is not None
        ), "Can only update from a fitted instance!"

        if not site_names:
            return self.infer(self.model_, **kwargs)

        posterior_samples = {
            k: v for k, v in self.posterior_samples_.items() if k not in site_names
        }

        if mode == "mean":
            to_condition_on = {
                k: np.mean(v, axis=0, keepdims=True)
                for k, v in posterior_samples.items()
            }
            num_samples = self.num_samples
            num_chains = self.num_chains
        else:
            to_condition_on = posterior_samples
            num_samples = 1 * self.num_chains
            num_chains = 1

        fun = partial(
            _update_posterior,
            rng_key=self._rng_key,
            model=self.model_,
            kernel_builder=self.build_kernel,
            num_samples=num_samples,
            num_warmup=self.num_warmup,
            num_chains=num_chains,
            progress_bar=self.progress_bar,
            **kwargs,
        )

        if do_vmap := (mode == "full"):
            fun = vmap(fun)

        updated_posterior = fun(to_condition_on=to_condition_on)

        if do_vmap:
            updated_posterior = {
                k: np.reshape(v, (-1,) + v.shape[2:])
                for k, v in updated_posterior.items()
            }

        posterior_samples.update(updated_posterior)
        self.posterior_samples_ = posterior_samples

        return self
