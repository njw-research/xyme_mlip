import jax.numpy as jnp
from jax import random, grad, vmap

# Define a more complex energy function: Double-Well Potential
def double_well_1D(x):
    return (x**2 - 1)**2

def double_well_2D(x, a=1, b=6, c=1, d=1):
    """
    Compute the double well potential energy for a single 2D coordinate.
    """

    x1, x2 = x
    return jnp.float32(0.25 * a * x1**4 - 0.5 * b * x1**2 + c * x1 + 0.5 * d * x2**2)

def mueller_potential_2D(x, alpha=0.1):
    """
    Compute the Mueller potential energy for a single 2D coordinate.
    """
    x1, x2 = x

    # Constants for the Mueller potential
    A = jnp.array([-200, -100, -170, 15], dtype=jnp.float32)
    a = jnp.array([-1, -1, -6.5, 0.7], dtype=jnp.float32)
    b = jnp.array([0, 0, 11, 0.6], dtype=jnp.float32)
    c = jnp.array([-10, -10, -6.5, 0.7], dtype=jnp.float32)
    x0 = jnp.array([1, 0, -0.5, -1], dtype=jnp.float32)
    y0 = jnp.array([0, 0.5, 1.5, 1], dtype=jnp.float32)

    # Compute the Mueller potential energy
    energy = 0
    for i in range(4):
        energy += A[i] * jnp.exp(
            a[i] * (x1 - x0[i])**2 + b[i] * (x1 - x0[i]) * (x2 - y0[i]) + c[i] * (x2 - y0[i])**2
        )
    
    return alpha * energy

# Define the Langevin sampling step with damping coefficient
def langevin_step(rng_key, func, x, kBT, step_size, gamma):
    rng_key, rng_subkey = random.split(rng_key)
    grad_func = grad(func)
    grad_func = vmap(grad_func)
    grad_E = grad_func(x)
    noise = random.normal(rng_subkey)
    x_new = x - gamma * grad_E * step_size + jnp.sqrt(2 * gamma * kBT * step_size) * noise
    return x_new, rng_key

# Perform Langevin sampling
def langevin_sampling(rng_key, func, x0, kBT, step_size, gamma, n_samples):
    samples = []
    x = x0
    for _ in range(n_samples):
        x, rng_key = langevin_step(rng_key, func, x, kBT, step_size, gamma)
        samples.append(x)
    return jnp.array(samples)

# Define the Langevin sampling step with damping coefficient
def md_step(rng_key, func, x, kBT, step_size, gamma, damping):
    rng_key, rng_subkey = random.split(rng_key)
    grad_func = grad(func)
    grad_func = vmap(grad_func)
    grad_E = grad_func(x)
    noise = random.normal(rng_subkey) * damping
    x_new = x - gamma * grad_E * step_size + jnp.sqrt(2 * gamma * kBT * step_size) * noise
    return x_new, rng_key

# Perform Langevin sampling
def md_sampling(rng_key, func, x0, kBT, step_size, gamma, n_samples, damping):
    samples = []
    x = x0
    for _ in range(n_samples):
        x, rng_key = md_step(rng_key, func, x, kBT, step_size, gamma, damping)
        samples.append(x)
    return jnp.array(samples)

def biasing_potential(x, x_i, y_i, k=1.0):
    """
    Harmonic biasing potential centered at (x_i, y_i).
    """
    x1, x2 = x
    return 0.5 * k * ((x1 - x_i)**2 + (x2 - y_i)**2)