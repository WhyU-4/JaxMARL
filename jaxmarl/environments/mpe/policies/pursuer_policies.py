"""
Pursuer (Adversary) Policies for 3v1 Pursuit-Evasion Environment

This module provides modular policy implementations for pursuers in the 3v1 
pursuit-evasion task. These policies can be easily swapped to test different
pursuit strategies.
"""

import jax
import jax.numpy as jnp
import chex
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from jaxmarl.environments.mpe.simple import State


class PursuerPolicy(ABC):
    """Base class for pursuer policies."""
    
    @abstractmethod
    def get_action(
        self, 
        obs: chex.Array, 
        agent_idx: int,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Get action for a pursuer agent.
        
        Args:
            obs: Observation array for the agent
            agent_idx: Index of the agent
            state: Current environment state
            key: JAX random key
            
        Returns:
            Action index (0-4 for discrete actions: no-op, left, right, down, up)
        """
        pass


class RandomPursuerPolicy(PursuerPolicy):
    """Random policy - selects actions uniformly at random."""
    
    def get_action(
        self, 
        obs: chex.Array, 
        agent_idx: int,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """Select random action."""
        return jax.random.randint(key, (), 0, 5)


class GreedyPursuerPolicy(PursuerPolicy):
    """
    Greedy policy - moves directly towards the evader.
    
    Each pursuer independently moves towards the evader's position
    without considering other pursuers.
    """
    
    def get_action(
        self, 
        obs: chex.Array, 
        agent_idx: int,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Move directly towards the evader.
        
        Observation structure for pursuer:
        - obs[0:2]: velocity
        - obs[2:4]: position
        - obs[4:8]: landmark positions (2 landmarks * 2D)
        - obs[8:14]: other agent positions (3 agents * 2D)
        - obs[14:16]: evader velocity
        
        The evader is always the last agent, so its relative position
        is in obs[12:14] (last agent in other_pos).
        """
        # Extract evader's relative position (last agent in observation)
        evader_rel_pos = obs[12:14]
        
        # Calculate the direction to the evader
        dx, dy = evader_rel_pos[0], evader_rel_pos[1]
        
        # Choose action based on direction
        # Actions: 0=no-op, 1=left, 2=right, 3=down, 4=up
        
        # Prioritize the dimension with larger difference
        abs_dx = jnp.abs(dx)
        abs_dy = jnp.abs(dy)
        
        # If horizontal distance is larger
        action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(dx > 0, 2, 1),  # right if dx > 0, else left
            jnp.where(dy > 0, 4, 3)   # up if dy > 0, else down
        )
        
        return action


class CoordinatedPursuerPolicy(PursuerPolicy):
    """
    Coordinated policy - pursuers coordinate to surround the evader.
    
    This policy attempts to encircle the evader by assigning different
    pursuers to approach from different angles.
    """
    
    def get_action(
        self, 
        obs: chex.Array, 
        agent_idx: int,
        state: State,
        key: chex.PRNGKey
    ) -> int:
        """
        Coordinate with other pursuers to surround the evader.
        
        Each pursuer is assigned a target angle around the evader:
        - Pursuer 0: 0° (right)
        - Pursuer 1: 120° (upper-left)
        - Pursuer 2: 240° (lower-left)
        """
        # Extract evader's relative position
        evader_rel_pos = obs[12:14]
        dx, dy = evader_rel_pos[0], evader_rel_pos[1]
        
        # Calculate current angle to evader
        current_angle = jnp.arctan2(dy, dx)
        
        # Target angles for each pursuer (in radians)
        # Distribute pursuers evenly around a circle
        target_angles = jnp.array([0.0, 2.0 * jnp.pi / 3.0, 4.0 * jnp.pi / 3.0])
        target_angle = target_angles[agent_idx]
        
        # Calculate desired offset position (circular formation)
        formation_radius = 0.3  # Distance to maintain from evader
        target_dx = dx - formation_radius * jnp.cos(target_angle)
        target_dy = dy - formation_radius * jnp.sin(target_angle)
        
        # Choose action to move towards target position
        abs_dx = jnp.abs(target_dx)
        abs_dy = jnp.abs(target_dy)
        
        # If we're close enough to formation position, move towards evader
        at_formation = (abs_dx < 0.1) & (abs_dy < 0.1)
        
        # Regular movement towards target position
        regular_action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(target_dx > 0, 2, 1),  # right if dx > 0, else left
            jnp.where(target_dy > 0, 4, 3)   # up if dy > 0, else down
        )
        
        # If at formation, move directly towards evader
        direct_action = jnp.where(
            abs_dx > abs_dy,
            jnp.where(dx > 0, 2, 1),
            jnp.where(dy > 0, 4, 3)
        )
        
        action = jnp.where(at_formation, direct_action, regular_action)
        
        return action


# Default policy mapping for easy access
PURSUER_POLICIES = {
    "random": RandomPursuerPolicy,
    "greedy": GreedyPursuerPolicy,
    "coordinated": CoordinatedPursuerPolicy,
}


def get_pursuer_policy(policy_name: str) -> PursuerPolicy:
    """
    Factory function to get a pursuer policy by name.
    
    Args:
        policy_name: Name of the policy ("random", "greedy", "coordinated")
        
    Returns:
        Instance of the requested policy
        
    Raises:
        ValueError: If policy_name is not recognized
    """
    if policy_name not in PURSUER_POLICIES:
        raise ValueError(
            f"Unknown pursuer policy: {policy_name}. "
            f"Available policies: {list(PURSUER_POLICIES.keys())}"
        )
    return PURSUER_POLICIES[policy_name]()
