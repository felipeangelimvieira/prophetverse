"""Custom ELBO objectives for use in prophetverse inference engines.

This module provides a deterministic variant of ``Trace_ELBO`` that produces
identical results regardless of the Python process's ``PYTHONHASHSEED`` value.
"""

import warnings

import jax
import jax.numpy as jnp
from jax import random, vmap
from numpyro.handlers import replay, seed, substitute, trace
from numpyro.infer import Trace_ELBO
from numpyro.infer.util import compute_log_probs, is_identically_one
from numpyro.util import _validate_model, check_model_guide_match


class DeterministicTraceELBO(Trace_ELBO):
    """A ``Trace_ELBO`` variant that is insensitive to ``PYTHONHASHSEED``.

    Numpyro's ``Trace_ELBO`` builds the per-site ELBO contribution dict by
    iterating over ``set(model_log_probs).union(guide_log_probs)``.  Python's
    built-in ``set`` has a hash-randomised iteration order that changes with
    ``PYTHONHASHSEED``.  Summing floating-point values in different orders
    produces slightly different results (due to IEEE 754 non-associativity),
    which can cause the LBFGS stopping criterion to trigger at a different
    iteration — leading to visibly different MAP estimates across Python
    processes even with identical random seeds.

    This subclass fixes the issue by sorting the union of site names
    alphabetically before accumulating the ELBO, making the summation order
    deterministic regardless of ``PYTHONHASHSEED``.
    """

    def loss_with_mutable_state(
        self,
        rng_key,
        param_map,
        model,
        guide,
        *args,
        **kwargs,
    ):
        """Compute the ELBO loss with sorted site ordering for determinism.

        This is identical to ``Trace_ELBO.loss_with_mutable_state`` except that
        the union of site names is sorted alphabetically before the per-site
        ELBO contributions are accumulated.  This removes sensitivity to
        ``PYTHONHASHSEED``.
        """

        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_guide = seed(guide, guide_seed)
            guide_log_probs, guide_trace = compute_log_probs(
                seeded_guide, args, kwargs, param_map
            )
            mutable_params = {
                name: site["value"]
                for name, site in guide_trace.items()
                if site["type"] == "mutable"
            }
            params.update(mutable_params)
            if self.multi_sample_guide:
                plates = {
                    name: site["value"]
                    for name, site in guide_trace.items()
                    if site["type"] == "plate"
                }

                def compute_model_log_probs(key, latent):
                    with seed(rng_seed=key), substitute(data={**latent, **plates}):
                        model_log_probs, model_trace = compute_log_probs(
                            model, args, kwargs, params
                        )
                    _validate_model(model_trace, plate_warning="loose")
                    return model_log_probs

                num_guide_samples = None
                for site in guide_trace.values():
                    if site["type"] == "sample":
                        num_guide_samples = site["value"].shape[0]
                        break
                if num_guide_samples is None:
                    raise ValueError("guide is missing `sample` sites.")
                seeds = random.split(model_seed, num_guide_samples)
                latents = {
                    name: site["value"]
                    for name, site in guide_trace.items()
                    if (site["type"] == "sample" and site["value"].size > 0)
                    or (site["type"] == "deterministic")
                }
                model_log_probs = vmap(compute_model_log_probs)(seeds, latents)
                model_log_probs = jax.tree.map(
                    lambda x: jnp.sum(x, axis=0), model_log_probs
                )
            else:
                seeded_model = seed(model, model_seed)
                replay_model = replay(seeded_model, guide_trace)
                model_log_probs, model_trace = compute_log_probs(
                    replay_model, args, kwargs, params
                )
                check_model_guide_match(model_trace, guide_trace)
                _validate_model(model_trace, plate_warning="loose")
                mutable_params.update(
                    {
                        name: site["value"]
                        for name, site in model_trace.items()
                        if site["type"] == "mutable"
                    }
                )

            # Compute log p(z) - log q(z).
            # KEY FIX: sort site names so the summation order is deterministic
            # regardless of PYTHONHASHSEED.  numpyro's Trace_ELBO uses an
            # unordered set here, whose iteration order is hash-randomised.
            union = sorted(set(model_log_probs).union(guide_log_probs))
            _elbo_particle = {
                name: model_log_probs.get(name, jnp.array(0.0))
                - guide_log_probs.get(name, jnp.array(0.0))
                for name in union
            }
            if self.sum_sites:
                elbo_particle = sum(_elbo_particle.values(), start=jnp.array(0.0))
            else:
                elbo_particle = _elbo_particle

            if mutable_params:
                if self.num_particles == 1:
                    return elbo_particle, mutable_params
                warnings.warn(
                    "mutable state is currently ignored when num_particles > 1."
                )
            return elbo_particle, None

        if self.num_particles == 1:
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {
                "loss": jax.tree.map(jnp.negative, elbo),
                "mutable_state": mutable_state,
            }
        else:
            rng_keys = random.split(rng_key, self.num_particles)
            elbos, mutable_state = self.vectorize_particles_fn(
                single_particle_elbo, rng_keys
            )
            return {
                "loss": jax.tree.map(lambda x: -jnp.mean(x), elbos),
                "mutable_state": mutable_state,
            }
