from src.mlip.message_passing import MessagePassing
from src.utils.containers import Graph
import chex
import haiku as hk
import jax.numpy as jnp
import optax


def loss_fn_apply(
    key: chex.PRNGKey,
    params: hk.Params,
    x: Graph,
    model_apply: MessagePassing,
    forces_weight: float = 1.0,
) -> chex.Array:
    energy, forces = model_apply(params, x)
    energy_loss = jnp.mean(optax.l2_loss(energy, x.energy))
    forces_loss = jnp.mean(optax.l2_loss(forces, x.forces))
    loss = energy_loss + forces_weight * forces_loss
    info = {"energy_loss": energy_loss,
            "forces_loss": forces_loss,
            "loss": loss,
            "energy_mae": jnp.mean(jnp.abs(energy - x.energy)),
            "forces_mae": jnp.mean(jnp.abs(forces - x.forces))
            }
    return loss, info