"""
Example usage of the 3v1 pursuit-evasion environment with heuristic policies.

This script demonstrates how to:
1. Create the 3v1 environment
2. Use different pursuer policies
3. Use different evader policies at various difficulty levels
4. Run episodes and visualize results
"""

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.mpe.policies import (
    get_pursuer_policy,
    get_evader_policy,
)


def run_episode_with_policies(
    env,
    pursuer_policy_name="greedy",
    evader_policy_name="basic",
    max_steps=100,
    seed=0
):
    """
    Run a single episode with specified policies.
    
    Args:
        env: The environment instance
        pursuer_policy_name: Name of pursuer policy ("random", "greedy", "coordinated")
        evader_policy_name: Name of evader policy ("random", "basic", "intermediate", "advanced")
        max_steps: Maximum steps per episode
        seed: Random seed
        
    Returns:
        Episode statistics including total rewards and captures
    """
    # Initialize policies
    pursuer_policy = get_pursuer_policy(pursuer_policy_name)
    evader_policy = get_evader_policy(evader_policy_name)
    
    # Initialize environment
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    
    total_rewards = {agent: 0.0 for agent in env.agents}
    num_captures = 0
    
    for step in range(max_steps):
        key, *action_keys = jax.random.split(key, len(env.agents) + 1)
        actions = {}
        
        # Get actions for pursuers
        for i, agent_name in enumerate(env.adversaries):
            agent_key = action_keys[i]
            action = pursuer_policy.get_action(
                obs[agent_name],
                i,
                state,
                agent_key
            )
            actions[agent_name] = action
        
        # Get action for evader
        evader_name = env.good_agents[0]
        evader_key = action_keys[len(env.adversaries)]
        action = evader_policy.get_action(
            obs[evader_name],
            state,
            evader_key
        )
        actions[evader_name] = action
        
        # Step environment
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        
        # Track rewards
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
        
        # Check if evader was captured (negative reward for evader)
        if rewards[evader_name] < -5:
            num_captures += 1
        
        if dones["__all__"]:
            break
    
    return {
        "total_rewards": total_rewards,
        "num_captures": num_captures,
        "steps": step + 1,
        "pursuer_policy": pursuer_policy_name,
        "evader_policy": evader_policy_name,
    }


def compare_policies():
    """Compare different policy combinations."""
    env = make("MPE_simple_tag_3v1")
    
    pursuer_policies = ["random", "greedy", "coordinated"]
    evader_policies = ["random", "basic", "intermediate", "advanced"]
    
    print("=" * 80)
    print("3v1 Pursuit-Evasion Policy Comparison")
    print("=" * 80)
    print()
    
    for pursuer_policy in pursuer_policies:
        print(f"\nPursuer Policy: {pursuer_policy.upper()}")
        print("-" * 80)
        
        for evader_policy in evader_policies:
            # Run multiple episodes for statistical significance
            results = []
            for seed in range(5):
                result = run_episode_with_policies(
                    env,
                    pursuer_policy_name=pursuer_policy,
                    evader_policy_name=evader_policy,
                    max_steps=100,
                    seed=seed
                )
                results.append(result)
            
            # Calculate average metrics
            avg_pursuer_reward = sum(
                r["total_rewards"]["adversary_0"] for r in results
            ) / len(results)
            avg_evader_reward = sum(
                r["total_rewards"]["agent_0"] for r in results
            ) / len(results)
            avg_captures = sum(r["num_captures"] for r in results) / len(results)
            avg_steps = sum(r["steps"] for r in results) / len(results)
            
            print(f"  Evader: {evader_policy:12} | "
                  f"Pursuer Reward: {avg_pursuer_reward:6.1f} | "
                  f"Evader Reward: {avg_evader_reward:6.1f} | "
                  f"Captures: {avg_captures:4.1f} | "
                  f"Steps: {avg_steps:5.1f}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    print("\n🎮 JaxMARL 3v1 Pursuit-Evasion Environment\n")
    
    # Example 1: Single episode with specific policies
    print("Example 1: Running single episode with greedy pursuers vs intermediate evader")
    print("-" * 80)
    
    env = make("MPE_simple_tag_3v1")
    result = run_episode_with_policies(
        env,
        pursuer_policy_name="greedy",
        evader_policy_name="intermediate",
        max_steps=100,
        seed=42
    )
    
    print(f"Episode completed in {result['steps']} steps")
    print(f"Pursuer rewards: {result['total_rewards']['adversary_0']:.2f}")
    print(f"Evader reward: {result['total_rewards']['agent_0']:.2f}")
    print(f"Number of captures: {result['num_captures']}")
    print()
    
    # Example 2: Compare all policy combinations
    print("\nExample 2: Comparing all policy combinations (5 episodes each)")
    compare_policies()
