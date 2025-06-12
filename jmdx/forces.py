from jax import jit, Array
import jax.numpy as jnp
from jmdx.space import calc_distance_matrix, calc_distance, apply_pbc


@jit
def lennard_jones_potential(r: Array, epsilon: float = 1.0, sigma: float = 1.0) -> Array:
    """Calculate Lennard-Jones potential energy."""
    sigma_over_r = sigma / r
    sigma_over_r6 = sigma_over_r**6
    sigma_over_r12 = sigma_over_r6**2
    return 4.0 * epsilon * (sigma_over_r12 - sigma_over_r6)


@jit
def lennard_jones_force(r: Array, epsilon: float = 1.0, sigma: float = 1.0) -> Array:
    """Calculate Lennard-Jones force magnitude."""
    sigma_over_r = sigma / r
    sigma_over_r6 = sigma_over_r**6
    sigma_over_r12 = sigma_over_r6**2
    return 24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / r


@jit
def lennard_jones(positions: Array, box: Array, epsilon: float = 1.0, sigma: float = 1.0, 
                  cutoff: float = 2.5) -> tuple[Array, Array]:
    """Calculate Lennard-Jones potential energy and forces."""
    n_atoms = positions.shape[0]
    
    # Calculate distance matrix
    distances = calc_distance_matrix(positions, box)
    
    # Apply cutoff
    mask = (distances > 0) & (distances < cutoff * sigma)
    
    # Calculate potential energy
    r_masked = jnp.where(mask, distances, jnp.inf)
    potential = jnp.sum(lennard_jones_potential(r_masked, epsilon, sigma)) / 2.0
    
    # Calculate forces
    forces = jnp.zeros_like(positions)
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j and distances[i, j] < cutoff * sigma:
                dr = positions[j] - positions[i]
                dr = apply_pbc(dr, box)
                r = distances[i, j]
                
                force_mag = lennard_jones_force(r, epsilon, sigma)
                force_vec = force_mag * dr / r
                forces = forces.at[i].add(force_vec)
    
    return potential, forces


@jit
def coulomb_potential(r: Array, q1: float, q2: float, k_e: float = 1.0) -> Array:
    """Calculate Coulomb potential energy."""
    return k_e * q1 * q2 / r


@jit  
def coulomb_force(r: Array, q1: float, q2: float, k_e: float = 1.0) -> Array:
    """Calculate Coulomb force magnitude."""
    return k_e * q1 * q2 / (r**2)


@jit
def coulomb(positions: Array, charges: Array, box: Array, k_e: float = 1.0,
            cutoff: float = 10.0) -> tuple[Array, Array]:
    """Calculate Coulomb potential energy and forces."""
    n_atoms = positions.shape[0]
    
    # Calculate distance matrix
    distances = calc_distance_matrix(positions, box)
    
    # Apply cutoff
    mask = (distances > 0) & (distances < cutoff)
    
    # Calculate potential energy
    potential = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] < cutoff:
                potential += coulomb_potential(distances[i, j], charges[i], charges[j], k_e)
    
    # Calculate forces
    forces = jnp.zeros_like(positions)
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j and distances[i, j] < cutoff:
                dr = positions[j] - positions[i]
                dr = apply_pbc(dr, box)
                r = distances[i, j]
                
                force_mag = coulomb_force(r, charges[i], charges[j], k_e)
                force_vec = force_mag * dr / r
                forces = forces.at[i].add(force_vec)
    
    return potential, forces


@jit
def neighbor_list(positions: Array, box: Array, cutoff: float) -> Array:
    """Create neighbor list for efficient force calculations."""
    n_atoms = positions.shape[0]
    distances = calc_distance_matrix(positions, box)
    
    # Create neighbor mask
    mask = (distances > 0) & (distances < cutoff)
    
    # Get indices of neighbors
    i_indices, j_indices = jnp.where(mask)
    
    return jnp.column_stack([i_indices, j_indices])


def verlet_list(positions: Array, box: Array, cutoff: float, skin: float = 0.5):
    """Create Verlet neighbor list with skin distance."""
    verlet_cutoff = cutoff + skin
    return neighbor_list(positions, box, verlet_cutoff)