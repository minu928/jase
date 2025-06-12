from jax import jit, Array
import jax.numpy as jnp


@jit
def wrap(vec: Array, box: Array) -> Array:
    return jnp.mod(vec, box)


@jit
def apply_pbc(vec: Array, box: Array) -> Array:
    half_box = box * 0.5
    return jnp.mod(vec + half_box, box) - half_box


@jit
def calc_distance(vec: Array) -> Array:
    return jnp.sqrt(jnp.sum(jnp.square(vec), axis=-1))


def vectorize_box(box, *, dtype=None):
    box = jnp.array(box, dtype=dtype)
    ndim = box.ndim
    if ndim == 1 and box.shape == (3,):
        return box
    elif ndim == 2:
        if not box.shape == (3, 3):
            raise ValueError(f"Box shape should be (3,) or (3,3)")
        if jnp.any(box[~jnp.eye(3, 3, dtype=bool)] != 0):
            raise ValueError("Currently only support orthogonal/cubic boxes")
        return jnp.diag(box)
    else:
        raise ValueError("Box should be 1D or 2D array")


@jit
def calc_distance_matrix(positions: Array, box: Array) -> Array:
    """Calculate pairwise distance matrix with PBC."""
    n_atoms = positions.shape[0]
    i_indices, j_indices = jnp.meshgrid(jnp.arange(n_atoms), jnp.arange(n_atoms), indexing='ij')
    
    dr = positions[i_indices] - positions[j_indices]
    dr = apply_pbc(dr, box)
    distances = calc_distance(dr)
    
    return distances


@jit
def calc_displacement(pos1: Array, pos2: Array, box: Array) -> Array:
    """Calculate displacement vector with minimum image convention."""
    dr = pos2 - pos1
    return apply_pbc(dr, box)


@jit
def calc_minimum_image(dr: Array, box: Array) -> Array:
    """Apply minimum image convention to displacement vectors."""
    return apply_pbc(dr, box)


@jit
def calc_center_of_mass(positions: Array, masses: Array) -> Array:
    """Calculate center of mass."""
    total_mass = jnp.sum(masses)
    return jnp.sum(positions * masses[:, None], axis=0) / total_mass


@jit
def calc_gyration_radius(positions: Array, masses: Array) -> Array:
    """Calculate radius of gyration."""
    com = calc_center_of_mass(positions, masses)
    dr = positions - com
    total_mass = jnp.sum(masses)
    rg_squared = jnp.sum(masses[:, None] * jnp.sum(dr**2, axis=1, keepdims=True)) / total_mass
    return jnp.sqrt(rg_squared)
