import chex
import jax
import haiku as hk
import optax
from typing import Callable, Tuple
from src.utils.containers import Graph


def training_step(
    params: hk.Params,
    x: Graph,
    opt_state: optax.OptState,
    key: chex.PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[
        [chex.PRNGKey, chex.ArrayTree, Graph], Tuple[chex.Array, dict]
    ],
    use_pmap: bool = False,
    pmap_axis_name: str = "data",
) -> Tuple[hk.Params, optax.OptState, dict]:
    
    (loss, info), grad = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(key, params, x)
    if use_pmap:
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
    updates, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, updates)

    info.update(
        grad_norm=optax.global_norm(grad),
        update_norm=optax.global_norm(updates),
        param_norm=optax.global_norm(params),
        loss=loss
    )

    return new_params, new_opt_state, info


def validation_step(
    params: hk.Params,
    x: Graph,
    key: chex.PRNGKey,
    loss_fn: Callable[
        [chex.PRNGKey, chex.ArrayTree, Graph], Tuple[chex.Array, dict]
    ],
) -> dict:
    
    loss, info = loss_fn(key, params, x)
    
    info.update(
        loss=loss,
        param_norm=optax.global_norm(params)
    )

    return info