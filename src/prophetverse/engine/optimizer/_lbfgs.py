"""LBFGS solver for NumPyro."""

from typing import Any, Callable, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from numpyro.optim import _NumPyroOptim

from prophetverse.logger import logger


class LBFGS(_NumPyroOptim):
    """
    An L-BFGS optimizer integrated with NumPyro, using optax.

    Parameters
    ----------
    gtol : float, default=1e-6
        Gradient tolerance for stopping criterion.
    tol : float, default=1e-6
        Function value tolerance for stopping criterion.
    learning_rate : float, default=1e-3
        Initial learning rate.
    memory_size : int, default=10
        Memory size for L-BFGS updates.
    scale_init_precond : bool, default=True
        Whether to scale the initial preconditioner.
    max_linesearch_steps : int, default=20
        Maximum number of line search steps.
    initial_guess_strategy : str, default="one"
        Strategy for the initial line search step size guess.
    max_learning_rate : float, optional
        Maximum allowed learning rate during line search.
    linesearch_tol : float, default=0
        Tolerance parameter for line search.
    increase_factor : float, default=2
        Factor by which to increase step size during line search when conditions are
        met.
    slope_rtol : float, default=0.0001
        Relative tolerance for slope in the line search.
    curv_rtol : float, default=0.9
        Curvature condition tolerance for line search.
    approx_dec_rtol : float, default=0.000001
        Approximate decrease tolerance for line search.
    stepsize_precision : float, default=1e5
        Stepsize precision tolerance.
    """

    def __init__(
        self,
        max_iter: int = 100,
        gtol=1e-6,
        tol: float = 1e-6,
        learning_rate=1e-3,
        memory_size=10,
        scale_init_precond=True,
        # linesearch
        max_linesearch_steps=50,
        initial_guess_strategy="one",
        max_learning_rate=None,
        linesearch_tol=0,
        increase_factor=2,
        slope_rtol=0.0001,
        curv_rtol=0.9,
        approx_dec_rtol=0.000001,
        stepsize_precision=1e5,
    ):
        self.max_iter = max_iter
        self.gtol = gtol
        self.tol = tol
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.scale_init_precond = scale_init_precond

        self.max_linesearch_steps = max_linesearch_steps
        self.initial_guess_strategy = initial_guess_strategy
        self.max_learning_rate = max_learning_rate
        self.linesearch_tol = linesearch_tol
        self.increase_factor = increase_factor
        self.slope_rtol = slope_rtol
        self.curv_rtol = curv_rtol
        self.approx_dec_rtol = approx_dec_rtol
        self.stepsize_precision = stepsize_precision

        self._transformation = self.get_transformation()

    def get_transformation(self) -> optax.GradientTransformationExtraArgs:
        """
        Construct the gradient transformation for L-BFGS optimization with line search.

        Returns
        -------
        optax.GradientTransformation
            An optax transformation that first prints iteration information,
            then applies L-BFGS updates with optional line search.
        """
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=self.max_linesearch_steps,
            initial_guess_strategy="one",
            max_learning_rate=self.max_learning_rate,
            tol=self.linesearch_tol,
            increase_factor=self.increase_factor,
            slope_rtol=self.slope_rtol,
            curv_rtol=self.curv_rtol,
            approx_dec_rtol=self.approx_dec_rtol,
            stepsize_precision=self.stepsize_precision,
            verbose=False,
        )
        return optax.chain(
            print_info(),
            optax.lbfgs(
                learning_rate=self.learning_rate,
                memory_size=self.memory_size,
                scale_init_precond=self.scale_init_precond,
                linesearch=linesearch,
            ),
        )

    def eval_and_update(
        self,
        fn: Callable[[Any], tuple],
        state,
        forward_mode_differentiation=False,
    ):
        """
        Evaluate the function at the current parameters and perform the L-BFGS update.

        Parameters
        ----------
        fn : Callable[[Any], tuple]
            A function that given parameters returns a tuple (value, auxiliary_data).
        state : Any
            Current state of the optimization, including parameters and opt state.
        forward_mode_differentiation : bool, default=False
            Whether to use forward-mode differentiation (not currently used).

        Returns
        -------
        tuple
            A tuple ((value, grad), svi_state) where:
            - (value, grad) is the objective function value and gradients at the updated
                parameters.
            - svi_state is the updated SVI state, containing the updated parameters and
                optimizer state.
        """
        params = self.get_params(state)

        final_params, final_state, value, grad = run_opt(
            params,
            fun=fn,
            opt=self._transformation,
            max_iter=self.max_iter,
            gtol=self.gtol,
            tol=self.tol,
        )

        step_num = otu.tree_get(final_state, "count")
        svi_state = step_num, (final_params, final_state)

        return (value, None), svi_state

    def get_params(self, state):
        """Get the parameters from the state.

        This method is called from numpyro's SVI.
        """
        step, (params, opt_state) = state
        return params

    def init(self, params):
        """Initialize the optimizer state.

        This method is called from numpyro's SVI.
        """
        opt_state = self._transformation.init(params)
        return jnp.array(0), (params, opt_state)


def run_opt(init_params, fun, opt, max_iter, gtol, tol):
    """
    Run the optimization loop until convergence criteria are met or max iter reached.

    Parameters
    ----------
    init_params : Any
        Initial parameters for the optimization.
    fun : Callable[[Any], tuple]
        A function that given parameters returns a tuple (value, auxiliary_data).
    opt : optax.GradientTransformation
        The optax optimizer transformation.
    max_iter : int
        Maximum number of iterations to run the optimization.
    gtol : float
        Gradient norm tolerance for stopping criterion.
    tol : float
        Function value tolerance for stopping criterion.

    Returns
    -------
    final_params : Any
        The optimized parameters.
    final_state : Any
        The final state of the optimizer.
    final_value : chex.Numeric
        The final value of the objective function.
    grad : Any
        The gradient at the final parameters.
    """

    def _fun(params):
        value, _ = fun(params)
        return value

    value_and_grad_fun = optax.value_and_grad_from_state(_fun)

    def step(carry):
        params, state, _ = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=_fun
        )
        params = optax.apply_updates(params, updates)
        return params, state, value

    def continuing_criterion(carry):
        _, state, value = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | (
            (iter_num < max_iter) & (err >= gtol) & (value >= tol)
        )

    init_carry = (init_params, opt.init(init_params), jnp.inf)
    final_params, final_state, final_value = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )

    _, grad = value_and_grad_fun(final_params, state=final_state)
    return final_params, final_state, final_value, grad


class InfoState(NamedTuple):
    """
    A named tuple holding iteration information.

    Attributes
    ----------
    iter_num : chex.Numeric
        The current iteration number.
    """

    iter_num: chex.Numeric


def print_info():
    """
    Create a gradient transformation.

    Prints iteration number, function value,
    and gradient norm at each step. This is useful for debugging and monitoring
    the optimization process.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        A gradient transformation that, when applied, prints optimization info
        at each iteration.
    """

    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):

        del params, extra_args

        fmt = "Iteration: {i}, Value: {v}, Gradient norm: {e}"

        def _log(*args, **kwargs):
            logger.debug(fmt.format(*args, **kwargs))

        jax.debug.callback(_log, i=state.iter_num, v=value, e=otu.tree_l2_norm(grad))

        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
