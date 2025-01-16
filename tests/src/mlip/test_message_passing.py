# Import standard libraries
import jax
import haiku as hk
import haiku.experimental.flax as hkflax
import jax.numpy as jnp
import pytest
from src.utils.containers import Graph
from src.mlip.message_passing import MessagePassing

# Model hyperparameters
num_features = 4
max_degree = 1
num_iterations = 1
num_basis_functions = 8
cutoff = 3.0
max_atomic_number = 9
num_atoms = 3

# Initialize random key
key = jax.random.PRNGKey(0)

# Test function to check the shapes of energy and forces
@pytest.fixture(scope="module")
def setup_graph():
    key = jax.random.PRNGKey(0)
    graph = Graph(features=jnp.ones((1, 1, 3), dtype=jnp.uint8), positions=jax.random.uniform(key, (3, 3)))
    return graph

def test_energy_and_forces(setup_graph):
    # Initialize message passing
    message_passing = MessagePassing(
        features=num_features,
        max_degree=max_degree,
        num_iterations=num_iterations,
        num_basis_functions=num_basis_functions,
        cutoff=cutoff,
        max_atomic_number=max_atomic_number
    )

    # Define energy and forces function 
    @hk.without_apply_rng
    @hk.transform
    def energy_and_forces(graph: Graph):
        mod = hkflax.lift(message_passing, name='e3x_mlip')
        return mod(graph.features, graph.positions)
    
    key = jax.random.PRNGKey(0)
    graph = setup_graph
    params = energy_and_forces.init(key, graph)
    energy, forces = energy_and_forces.apply(params, graph)
    
    # Assertions to check the shapes of energy and forces
    assert energy.shape == ()  # Scalar shape
    assert forces.shape == (3, 3)  # (3, 3) shape
