from typing import Callable
import numpyro
from numpyro.infer.initialization import init_to_mean
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value, Trace_ELBO, MCMC, NUTS, Predictive
from numpyro.infer.autoguide import AutoDelta
import jax


class InferenceEngine:

    def __init__(self, model: Callable, rng_key=None):
        self.model = model
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        self.rng_key = rng_key

    def infer(self, **kwargs): ...

    def predict(self, **kwargs): ...


class MAPInferenceEngine(InferenceEngine):

    def __init__(
        self,
        model: Callable,
        optimizer: numpyro.optim._NumPyroOptim = None,
        num_steps=10000,
        rng_key=None,
    ):
        if optimizer is None:
            optimizer = numpyro.optim.Adam(step_size=0.001)
        self.optimizer = optimizer
        self.num_steps = num_steps
        super().__init__(model, rng_key)

    def infer(self, **kwargs):
        self.guide_ = AutoDelta(self.model, init_loc_fn=init_to_mean())
        self.svi_ = SVI(self.model, self.guide_, self.optimizer, loss=Trace_ELBO())
        self.run_results_ = self.svi_.run(
            rng_key=self.rng_key, num_steps=self.num_steps, **kwargs
        )
        self.posterior_samples_ = self.guide_.sample_posterior(self.rng_key, params=self.run_results_.params, **kwargs)
        return self

    def predict(self, **kwargs):
        predictive = numpyro.infer.Predictive(
            self.model,
            params=self.run_results_.params,
            guide=self.guide_,
            #posterior_samples=self.posterior_samples_,
            num_samples=1000,
        )
        self.samples_ = predictive(
            rng_key=self.rng_key,
            **kwargs
        )
        return self.samples_


class MCMCInferenceEngine(InferenceEngine):

    def __init__(
        self,
        model: Callable,
        num_samples=1000,
        num_warmup=200,
        num_chains=1,
        dense_mass=False,
        rng_key=None,
    ):
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.num_chains = num_chains
        self.dense_mass = dense_mass
        super().__init__(model, rng_key)

    def infer(self, **kwargs):
        self.mcmc_ = MCMC(
            NUTS(self.model, dense_mass=self.dense_mass, init_strategy=init_to_mean()),
            num_samples=self.num_samples,
            num_warmup=self.num_warmup,
        )
        self.mcmc_.run(self.rng_key, **kwargs)
        self.posterior_samples_ = self.mcmc_.get_samples()
        return self

    def predict(self, **kwargs):
        sites = set(self.posterior_samples_.keys()).union(["obs"])
        predictive = Predictive(self.model, self.posterior_samples_, return_sites=sites)

        self.samples_predictive_ = predictive(self.rng_key, **kwargs)
        self.samples_ = self.mcmc_.get_samples()
        return self.samples_predictive_
