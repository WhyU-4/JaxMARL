"""
Evader Policies for 3v1 Pursuit-Evasion Environment

This module provides heuristic evader policies at different difficulty levels.
These policies allow testing pursuer strategies against opponents of varying skill.
"""

import jax
import jax.numpy as jnp
import chex
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from jaxmarl.environments.mpe.simple import State


class EvaderPolicy(ABC):
    """Base class for evader policies."""
    
    @abstractmethod
    def get_action(
        self, 
        obs: chex.Array, 
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Get action for the evader agent.
        
        Args:
            obs: Observation array for the evader
            state: Current environment state
            key: JAX random key
            
        Returns:
            Action index (0-4 for discrete actions: no-op, left, right, down, up)
        """
        pass


class RandomEvaderPolicy(EvaderPolicy):
    """
    Level 0: Random Evader
    
    The evader takes completely random actions.
    This is the easiest difficulty level.
    """
    
    def get_action(
        self, 
        obs: chex.Array,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """Select random action."""
        return jax.random.randint(key, (), 0, 5)


class BasicEvaderPolicy(EvaderPolicy):
    """
    Level 1: Basic Evader
    
    The evader runs away from the nearest pursuer.
    This is a simple reactive strategy that only considers
    the closest threat.
    """
    
    def get_action(
        self, 
        obs: chex.Array,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Run away from the nearest pursuer.
        
        Observation structure for evader:
        - obs[0:2]: velocity
        - obs[2:4]: position
        - obs[4:8]: landmark positions (2 landmarks * 2D)
        - obs[8:14]: other agent positions (3 pursuers * 2D)
        """
        # Extract pursuer positions (relative to evader)
        pursuer_positions = obs[8:14].reshape(3, 2)
        
        # Calculate distances to each pursuer
        distances = jnp.sqrt(jnp.sum(pursuer_positions**2, axis=1))
        
        # Find nearest pursuer
        nearest_idx = jnp.argmin(distances)
        nearest_pursuer_pos = pursuer_positions[nearest_idx]
        
        # Run in opposite direction from nearest pursuer
        dx, dy = -nearest_pursuer_pos[0], -nearest_pursuer_pos[1]
        
        # Choose action based on escape direction
        abs_dx = jnp.abs(dx)
        abs_dy = jnp.abs(dy)
        
        action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(dx > 0, 2, 1),  # right if dx > 0, else left
            jnp.where(dy > 0, 4, 3)   # up if dy > 0, else down
        )
        
        return action


class IntermediateEvaderPolicy(EvaderPolicy):
    """
    Level 2: Intermediate Evader
    
    The evader considers all pursuers and moves away from their
    center of mass. This provides better evasion against coordinated
    pursuit strategies.
    """
    
    def get_action(
        self, 
        obs: chex.Array,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Run away from the center of mass of all pursuers.
        
        This strategy is more effective against multiple pursuers
        as it considers all threats simultaneously.
        """
        # Extract pursuer positions
        pursuer_positions = obs[8:14].reshape(3, 2)
        
        # Calculate center of mass of pursuers
        pursuer_com = jnp.mean(pursuer_positions, axis=0)
        
        # Add small random noise to break symmetries
        noise = jax.random.normal(key, (2,)) * 0.1
        
        # Run in opposite direction from center of mass
        dx = -pursuer_com[0] + noise[0]
        dy = -pursuer_com[1] + noise[1]
        
        # Choose action based on escape direction
        abs_dx = jnp.abs(dx)
        abs_dy = jnp.abs(dy)
        
        action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(dx > 0, 2, 1),
            jnp.where(dy > 0, 4, 3)
        )
        
        return action


class AdvancedEvaderPolicy(EvaderPolicy):
    """
    Level 3: Advanced Evader
    
    The evader uses a sophisticated strategy that:
    1. Moves away from the weighted center of pursuers (closer pursuers have more weight)
    2. Avoids getting too close to boundaries
    3. Exploits gaps between pursuers
    
    This is the most challenging difficulty level.
    """
    
    def get_action(
        self, 
        obs: chex.Array,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Advanced evasion strategy with multiple considerations.
        """
        # Extract current position and pursuer positions
        evader_pos = obs[2:4]
        pursuer_positions = obs[8:14].reshape(3, 2)
        
        # Calculate distances to each pursuer
        distances = jnp.sqrt(jnp.sum(pursuer_positions**2, axis=1))
        
        # Prevent division by zero
        safe_distances = jnp.maximum(distances, 0.01)
        
        # Calculate weighted center of mass (closer pursuers have more weight)
        weights = 1.0 / safe_distances
        weights = weights / jnp.sum(weights)
        weighted_com = jnp.sum(pursuer_positions * weights[:, None], axis=0)
        
        # Primary escape direction: away from weighted center
        escape_dx = -weighted_com[0]
        escape_dy = -weighted_com[1]
        
        # Boundary avoidance: add repulsive force from boundaries
        # Assuming map bounds are around [-1, 1] in each dimension
        boundary_margin = 0.7
        boundary_force_x = 0.0
        boundary_force_y = 0.0
        
        # Repel from left/right boundaries
        boundary_force_x = jnp.where(
            evader_pos[0] > boundary_margin,
            -0.5,  # push left if near right boundary
            boundary_force_x
        )
        boundary_force_x = jnp.where(
            evader_pos[0] < -boundary_margin,
            0.5,   # push right if near left boundary
            boundary_force_x
        )
        
        # Repel from top/bottom boundaries
        boundary_force_y = jnp.where(
            evader_pos[1] > boundary_margin,
            -0.5,  # push down if near top boundary
            boundary_force_y
        )
        boundary_force_y = jnp.where(
            evader_pos[1] < -boundary_margin,
            0.5,   # push up if near bottom boundary
            boundary_force_y
        )
        
        # Gap exploitation: find the largest gap between pursuers
        # Calculate angles of pursuers from evader's perspective
        angles = jnp.arctan2(pursuer_positions[:, 1], pursuer_positions[:, 0])
        sorted_angles = jnp.sort(angles)
        
        # Calculate gaps between adjacent pursuers
        angle_gaps = jnp.concatenate([
            sorted_angles[1:] - sorted_angles[:-1],
            jnp.array([2 * jnp.pi + sorted_angles[0] - sorted_angles[-1]])
        ])
        
        # Find the largest gap
        largest_gap_idx = jnp.argmax(angle_gaps)
        gap_angle = sorted_angles[largest_gap_idx] + angle_gaps[largest_gap_idx] / 2
        
        # Direction towards gap
        gap_dx = jnp.cos(gap_angle)
        gap_dy = jnp.sin(gap_angle)
        
        # Combine all factors
        # Weight: 60% escape from pursuers, 30% boundary avoidance, 10% gap exploitation
        combined_dx = 0.6 * escape_dx + 0.3 * boundary_force_x + 0.1 * gap_dx
        combined_dy = 0.6 * escape_dy + 0.3 * boundary_force_y + 0.1 * gap_dy
        
        # Add small random noise to avoid predictability
        noise = jax.random.normal(key, (2,)) * 0.05
        combined_dx += noise[0]
        combined_dy += noise[1]
        
        # Choose action
        abs_dx = jnp.abs(combined_dx)
        abs_dy = jnp.abs(combined_dy)
        
        action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(combined_dx > 0, 2, 1),
            jnp.where(combined_dy > 0, 4, 3)
        )
        
        return action


# Default policy mapping for easy access
EVADER_POLICIES = {
    "random": RandomEvaderPolicy,
    "basic": BasicEvaderPolicy,
    "intermediate": IntermediateEvaderPolicy,
    "advanced": AdvancedEvaderPolicy,
}


def get_evader_policy(policy_name: str) -> EvaderPolicy:
    """
    Factory function to get an evader policy by name.
    
    Args:
        policy_name: Name of the policy ("random", "basic", "intermediate", "advanced")
        
    Returns:
        Instance of the requested policy
        
    Raises:
        ValueError: If policy_name is not recognized
    """
    if policy_name not in EVADER_POLICIES:
        raise ValueError(
            f"Unknown evader policy: {policy_name}. "
            f"Available policies: {list(EVADER_POLICIES.keys())}"
        )
    return EVADER_POLICIES[policy_name]()
