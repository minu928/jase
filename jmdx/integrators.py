from jax import jit, Array, random
import jax.numpy as jnp


@jit
def velocity_verlet_step(positions: Array, velocities: Array, forces: Array, 
                        masses: Array, dt: float) -> tuple[Array, Array]:
    """Single step of velocity Verlet integration."""
    # Update positions
    accelerations = forces / masses[:, None]
    new_positions = positions + velocities * dt + 0.5 * accelerations * dt**2
    
    # Store old accelerations for velocity update
    return new_positions, accelerations


@jit
def velocity_verlet_velocity_update(velocities: Array, old_accelerations: Array,
                                   new_accelerations: Array, dt: float) -> Array:
    """Update velocities in velocity Verlet scheme."""
    return velocities + 0.5 * (old_accelerations + new_accelerations) * dt


def velocity_verlet(positions: Array, velocities: Array, forces_func, 
                   masses: Array, dt: float, n_steps: int) -> tuple[Array, Array]:
    """Complete velocity Verlet integration."""
    trajectory_pos = []
    trajectory_vel = []
    
    current_pos = positions
    current_vel = velocities
    
    # Initial force calculation
    _, current_forces = forces_func(current_pos)
    
    for step in range(n_steps):
        # Position update and get old accelerations
        new_pos, old_acc = velocity_verlet_step(current_pos, current_vel, 
                                               current_forces, masses, dt)
        
        # Calculate new forces
        _, new_forces = forces_func(new_pos)
        new_acc = new_forces / masses[:, None]
        
        # Velocity update
        new_vel = velocity_verlet_velocity_update(current_vel, old_acc, new_acc, dt)
        
        # Store trajectory
        trajectory_pos.append(current_pos)
        trajectory_vel.append(current_vel)
        
        # Update for next step
        current_pos = new_pos
        current_vel = new_vel
        current_forces = new_forces
    
    return jnp.array(trajectory_pos), jnp.array(trajectory_vel)


@jit
def leapfrog_step(positions: Array, velocities: Array, forces: Array,
                  masses: Array, dt: float) -> tuple[Array, Array]:
    """Single step of leapfrog integration."""
    # Update velocities by half step
    half_vel = velocities + 0.5 * forces / masses[:, None] * dt
    
    # Update positions by full step
    new_positions = positions + half_vel * dt
    
    return new_positions, half_vel


@jit
def leapfrog_velocity_finalize(half_velocities: Array, forces: Array,
                              masses: Array, dt: float) -> Array:
    """Finalize velocities in leapfrog scheme."""
    return half_velocities + 0.5 * forces / masses[:, None] * dt


def leapfrog(positions: Array, velocities: Array, forces_func,
            masses: Array, dt: float, n_steps: int) -> tuple[Array, Array]:
    """Complete leapfrog integration."""
    trajectory_pos = []
    trajectory_vel = []
    
    current_pos = positions
    current_vel = velocities
    
    for step in range(n_steps):
        # Calculate current forces
        _, current_forces = forces_func(current_pos)
        
        # Leapfrog step
        new_pos, half_vel = leapfrog_step(current_pos, current_vel, 
                                         current_forces, masses, dt)
        
        # Calculate new forces for velocity finalization
        _, new_forces = forces_func(new_pos)
        
        # Finalize velocities
        new_vel = leapfrog_velocity_finalize(half_vel, new_forces, masses, dt)
        
        # Store trajectory
        trajectory_pos.append(current_pos)
        trajectory_vel.append(current_vel)
        
        # Update for next step
        current_pos = new_pos
        current_vel = new_vel
    
    return jnp.array(trajectory_pos), jnp.array(trajectory_vel)


@jit
def langevin_step(positions: Array, velocities: Array, forces: Array,
                  masses: Array, temperature: float, gamma: float, dt: float,
                  key: Array) -> tuple[Array, Array, Array]:
    """Single step of Langevin dynamics."""
    # Langevin coefficients
    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt((1 - c1**2) * temperature / masses[:, None])
    
    # Update velocities (Langevin thermostat)
    random_forces = random.normal(key, velocities.shape)
    new_velocities = c1 * velocities + forces / masses[:, None] * dt + c2 * random_forces
    
    # Update positions
    new_positions = positions + new_velocities * dt
    
    return new_positions, new_velocities, key


def langevin(positions: Array, velocities: Array, forces_func,
            masses: Array, temperature: float, gamma: float, dt: float,
            n_steps: int, key: Array) -> tuple[Array, Array]:
    """Complete Langevin dynamics integration."""
    trajectory_pos = []
    trajectory_vel = []
    
    current_pos = positions
    current_vel = velocities
    current_key = key
    
    for step in range(n_steps):
        # Calculate current forces
        _, current_forces = forces_func(current_pos)
        
        # Split random key
        current_key, subkey = random.split(current_key)
        
        # Langevin step
        new_pos, new_vel, _ = langevin_step(current_pos, current_vel, current_forces,
                                           masses, temperature, gamma, dt, subkey)
        
        # Store trajectory
        trajectory_pos.append(current_pos)
        trajectory_vel.append(current_vel)
        
        # Update for next step
        current_pos = new_pos
        current_vel = new_vel
    
    return jnp.array(trajectory_pos), jnp.array(trajectory_vel)