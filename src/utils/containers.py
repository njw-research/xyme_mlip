from typing import NamedTuple, Callable, Tuple, Any, Optional, Dict
import chex
import haiku as hk
import optax

class FullGraphSample(NamedTuple):
    """
    A named tuple that holds a single sample from a full graph.
    Includes positions and features for the graph nodes.
    """
    positions: chex.Array  # Node positions in the graph (e.g., coordinates)
    features: chex.Array   # Node features (e.g., atom types, other descriptors)

    def __getitem__(self, i):
        """
        Allows indexing to access a specific element of the FullGraphSample.
        Returns a new FullGraphSample with the specified index.
        """
        return FullGraphSample(self.positions[i], self.features[i])


class Graph(NamedTuple):
    """
    A named tuple representing a graph with additional attributes.
    Includes node positions, features, energy, forces, and graph structure.
    """
    positions: chex.Array      # Node positions in the graph (e.g., coordinates)
    features: chex.Array       # Node features (e.g., atom types)
    energy: chex.Array = None  # Node energies (optional)
    forces: chex.Array = None  # Node forces (optional)
    dst_idx: chex.Array = None # Destination indices for edges (optional)
    src_idx: chex.Array = None # Source indices for edges (optional)
    segments: chex.Array = None # Segment indices for graph segmentation (optional)
    extra: dict = {}           # Any additional data related to the graph

    def __getitem__(self, i):
        """
        Allows indexing to access a specific element of the Graph.
        Returns a new Graph instance with the specified index, including optional fields.
        """
        return Graph(
            self.positions[i], 
            self.features[i], 
            self.energy[i] if self.energy is not None else None,
            self.forces[i] if self.forces is not None else None,
            self.dst_idx[i] if self.dst_idx is not None else None,
            self.src_idx[i] if self.src_idx is not None else None,
            self.segments[i] if self.segments is not None else None,
            self.extra
        )


class TrainingState(NamedTuple):
    """
    Holds the state of the training process.
    Includes model parameters, optimizer state, and random number generator key.
    """
    params: Any                # Model parameters
    opt_state: optax.OptState  # Optimizer state
    key: chex.PRNGKey          # Random number generator key

