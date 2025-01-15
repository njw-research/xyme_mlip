from ase.neighborlist import NeighborList

# Function to calculate bond lengths for a single molecule
def calculate_bond_lengths(atoms, cutoff=2.0):
    """
    Calculate bond lengths for an Atoms object using NeighborList.
    """
    # Initialize NeighborList with cutoff distances
    cutoffs = [cutoff] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    bond_lengths = []

    # Loop through all atoms and their neighbors
    for atom_index in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(atom_index)
        for neighbor, offset in zip(neighbors, offsets):
            if atom_index < neighbor:  # Avoid duplicates
                distance = atoms.get_distance(atom_index, neighbor)
                bond_lengths.append(distance)

    return bond_lengths

# Function to calculate dihedral angles for a single molecule
def calculate_dihedral_angles(atoms, cutoff=2.0):
    """
    Calculate dihedral angles for an Atoms object using NeighborList.

    Parameters:
        atoms: ASE Atoms object.
        cutoff: Cutoff distance to define bonds.

    Returns:
        dihedral_angles: List of dihedral angles in degrees.
    """
    # Initialize NeighborList with cutoff distances
    cutoffs = [cutoff] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    dihedral_angles = []

    # Loop through all atoms and identify sets of four connected atoms
    for i in range(len(atoms)):
        neighbors_i, _ = nl.get_neighbors(i)
        for j in neighbors_i:
            if j > i:  # Avoid duplicates
                neighbors_j, _ = nl.get_neighbors(j)
                for k in neighbors_j:
                    if k != i and k > j:  # Avoid loops
                        neighbors_k, _ = nl.get_neighbors(k)
                        for l in neighbors_k:
                            if l != j and l > k:  # Ensure uniqueness
                                # Calculate dihedral angle
                                angle = atoms.get_dihedral(i, j, k, l, mic=False)
                                dihedral_angles.append(angle)

    return dihedral_angles


