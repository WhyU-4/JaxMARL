"""
Test suite for the 3v1 pursuit-evasion environment and policies.
"""

import pytest
import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.mpe.policies import (
    get_pursuer_policy,
    get_evader_policy,
    PURSUER_POLICIES,
    EVADER_POLICIES,
)


class TestSimpleTag3v1Environment:
    """Test the SimpleTag3v1MPE environment."""
    
    def test_environment_creation(self):
        """Test that the environment can be created."""
        env = make("MPE_simple_tag_3v1")
        assert env is not None
        assert len(env.agents) == 4
        assert len(env.adversaries) == 3
        assert len(env.good_agents) == 1
    
    def test_environment_reset(self):
        """Test environment reset."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        assert len(obs) == 4
        assert state.p_pos.shape == (6, 2)  # 4 agents + 2 landmarks
        assert state.p_vel.shape == (6, 2)
    
    def test_observation_spaces(self):
        """Test observation space dimensions."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        # Adversary observations should be 16-dimensional
        for adversary in env.adversaries:
            assert obs[adversary].shape == (16,)
        
        # Good agent observations should be 14-dimensional
        for agent in env.good_agents:
            assert obs[agent].shape == (14,)
    
    def test_environment_step(self):
        """Test environment step."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        # Random actions
        key, *action_keys = jax.random.split(key, len(env.agents) + 1)
        actions = {
            agent: jax.random.randint(action_keys[i], (), 0, 5)
            for i, agent in enumerate(env.agents)
        }
        
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        
        assert len(rewards) == 4
        assert len(dones) == 5  # 4 agents + "__all__"
        assert state.step == 1
    
    def test_reward_structure(self):
        """Test that rewards are structured correctly."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        actions = {agent: 0 for agent in env.agents}  # No-op actions
        
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        
        # All adversaries should have the same reward (convert to float for comparison)
        adversary_rewards = [float(rewards[adv]) for adv in env.adversaries]
        assert len(set(adversary_rewards)) == 1  # All same


class TestPursuerPolicies:
    """Test pursuer policies."""
    
    def test_all_policies_exist(self):
        """Test that all advertised policies can be created."""
        for policy_name in PURSUER_POLICIES.keys():
            policy = get_pursuer_policy(policy_name)
            assert policy is not None
    
    def test_policy_actions(self):
        """Test that policies return valid actions."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        for policy_name in PURSUER_POLICIES.keys():
            policy = get_pursuer_policy(policy_name)
            
            for i, agent_name in enumerate(env.adversaries):
                key, subkey = jax.random.split(key)
                action = policy.get_action(obs[agent_name], i, state, subkey)
                
                # Action should be in valid range [0, 4]
                assert 0 <= action <= 4
    
    def test_random_policy_variance(self):
        """Test that random policy produces different actions."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        policy = get_pursuer_policy("random")
        
        actions = []
        for i in range(20):
            key, subkey = jax.random.split(key)
            action = policy.get_action(obs["adversary_0"], 0, state, subkey)
            actions.append(int(action))
        
        # Should have some variance (not all the same)
        assert len(set(actions)) > 1
    
    def test_greedy_policy_deterministic(self):
        """Test that greedy policy is deterministic for same observation."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        policy = get_pursuer_policy("greedy")
        
        actions = []
        for i in range(5):
            key, subkey = jax.random.split(key)
            action = policy.get_action(obs["adversary_0"], 0, state, subkey)
            actions.append(int(action))
        
        # All actions should be the same
        assert len(set(actions)) == 1


class TestEvaderPolicies:
    """Test evader policies."""
    
    def test_all_policies_exist(self):
        """Test that all advertised policies can be created."""
        for policy_name in EVADER_POLICIES.keys():
            policy = get_evader_policy(policy_name)
            assert policy is not None
    
    def test_policy_actions(self):
        """Test that policies return valid actions."""
        env = make("MPE_simple_tag_3v1")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        for policy_name in EVADER_POLICIES.keys():
            policy = get_evader_policy(policy_name)
            
            key, subkey = jax.random.split(key)
            action = policy.get_action(obs["agent_0"], state, subkey)
            
            # Action should be in valid range [0, 4]
            assert 0 <= action <= 4
    
    def test_policy_difficulty_levels(self):
        """Test that we have policies at different difficulty levels."""
        assert "random" in EVADER_POLICIES  # Level 0
        assert "basic" in EVADER_POLICIES  # Level 1
        assert "intermediate" in EVADER_POLICIES  # Level 2
        assert "advanced" in EVADER_POLICIES  # Level 3


class TestIntegration:
    """Integration tests combining environment and policies."""
    
    def test_full_episode_execution(self):
        """Test a full episode with policies."""
        env = make("MPE_simple_tag_3v1")
        pursuer_policy = get_pursuer_policy("greedy")
        evader_policy = get_evader_policy("basic")
        
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        
        max_steps = 50
        step_count = 0
        for step in range(max_steps):
            key, *action_keys = jax.random.split(key, len(env.agents) + 1)
            actions = {}
            
            for i, agent_name in enumerate(env.adversaries):
                actions[agent_name] = pursuer_policy.get_action(
                    obs[agent_name], i, state, action_keys[i]
                )
            
            actions[env.good_agents[0]] = evader_policy.get_action(
                obs[env.good_agents[0]], state, action_keys[-1]
            )
            
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, infos = env.step(step_key, state, actions)
            step_count += 1
            
            if dones["__all__"]:
                break
        
        # Episode should complete successfully (at least some steps were taken)
        assert step_count > 0
    
    def test_multiple_policy_combinations(self):
        """Test various combinations of pursuer and evader policies."""
        env = make("MPE_simple_tag_3v1")
        
        combinations = [
            ("random", "random"),
            ("greedy", "basic"),
            ("coordinated", "intermediate"),
            ("greedy", "advanced"),
        ]
        
        for pursuer_name, evader_name in combinations:
            pursuer_policy = get_pursuer_policy(pursuer_name)
            evader_policy = get_evader_policy(evader_name)
            
            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key)
            
            # Run a few steps
            for _ in range(10):
                key, *action_keys = jax.random.split(key, len(env.agents) + 1)
                actions = {}
                
                for i, agent_name in enumerate(env.adversaries):
                    actions[agent_name] = pursuer_policy.get_action(
                        obs[agent_name], i, state, action_keys[i]
                    )
                
                actions[env.good_agents[0]] = evader_policy.get_action(
                    obs[env.good_agents[0]], state, action_keys[-1]
                )
                
                key, step_key = jax.random.split(key)
                obs, state, rewards, dones, infos = env.step(step_key, state, actions)
                
                if dones["__all__"]:
                    break
            
            # Should complete without errors
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
