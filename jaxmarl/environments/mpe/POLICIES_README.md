# 3v1 Pursuit-Evasion Environment

This directory contains a specialized 3v1 pursuit-evasion environment based on MPE's `simple_tag`, along with modular heuristic policies for both pursuers and evaders.

## Environment: `MPE_simple_tag_3v1`

A 3-versus-1 pursuit-evasion scenario where:
- **3 Pursuers (Adversaries)**: Slower agents that aim to catch the evader
- **1 Evader**: Faster, more agile agent trying to avoid capture

### Key Features
- Pursuers receive positive rewards for catching the evader
- Evader receives negative rewards for being caught and for going out of bounds
- Environment includes 2 landmarks as obstacles/reference points
- Fully compatible with JaxMARL's API and registration system

### Usage

```python
from jaxmarl import make

# Create environment
env = make("MPE_simple_tag_3v1")

# Reset environment
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Take actions
actions = {agent: env.action_space(agent).sample(key) for agent in env.agents}
obs, state, rewards, dones, infos = env.step(key, state, actions)
```

## Modular Policy System

The policy system provides easy-to-use heuristic strategies for both pursuers and evaders, enabling:
- Quick prototyping and testing
- Baseline comparisons for learned policies
- Multi-level difficulty settings for evaluation

### Pursuer Policies

Located in `policies/pursuer_policies.py`:

1. **RandomPursuerPolicy**: Takes random actions (baseline)
2. **GreedyPursuerPolicy**: Each pursuer independently chases the evader
3. **CoordinatedPursuerPolicy**: Pursuers coordinate to surround the evader

#### Usage

```python
from jaxmarl.environments.mpe.policies import get_pursuer_policy

# Create a pursuer policy
policy = get_pursuer_policy("greedy")  # or "random", "coordinated"

# Get action for a pursuer
action = policy.get_action(
    obs=observation,
    agent_idx=0,
    state=env_state,
    key=rng_key
)
```

### Evader Policies

Located in `policies/evader_policies.py`:

1. **RandomEvaderPolicy** (Level 0): Random actions - easiest difficulty
2. **BasicEvaderPolicy** (Level 1): Runs from nearest pursuer - basic reactive strategy
3. **IntermediateEvaderPolicy** (Level 2): Runs from center of mass of pursuers - better against coordination
4. **AdvancedEvaderPolicy** (Level 3): Sophisticated strategy with:
   - Weighted evasion (closer pursuers weighted more)
   - Boundary avoidance
   - Gap exploitation between pursuers
   - Randomization to avoid predictability

#### Usage

```python
from jaxmarl.environments.mpe.policies import get_evader_policy

# Create an evader policy
policy = get_evader_policy("advanced")  # or "random", "basic", "intermediate"

# Get action for the evader
action = policy.get_action(
    obs=observation,
    state=env_state,
    key=rng_key
)
```

## Example Script

Run the example to see all policy combinations in action:

```bash
cd /home/runner/work/JaxMARL/JaxMARL
python -m jaxmarl.environments.mpe.example_3v1_policies
```

This will:
1. Run a single episode with greedy pursuers vs intermediate evader
2. Compare all pursuer-evader policy combinations with statistical results

## Integration with Training

The modular policy design allows easy integration with your training loops:

```python
from jaxmarl import make
from jaxmarl.environments.mpe.policies import get_pursuer_policy, get_evader_policy

env = make("MPE_simple_tag_3v1")

# Train pursuer agents against different evader difficulty levels
for difficulty in ["basic", "intermediate", "advanced"]:
    evader_policy = get_evader_policy(difficulty)
    
    # Your training loop here
    # Use evader_policy.get_action() for evader actions
    # Train learned policies for pursuers
```

## Policy Architecture

All policies inherit from base classes (`PursuerPolicy` or `EvaderPolicy`) and implement:

```python
def get_action(self, obs, agent_idx, state, key) -> int:
    """
    Returns action index (0-4):
    - 0: no-op
    - 1: move left
    - 2: move right  
    - 3: move down
    - 4: move up
    """
```

This consistent interface makes it easy to:
- Add new custom policies
- Swap policies without changing environment code
- Compare learned vs heuristic strategies

## File Structure

```
mpe/
├── simple_tag_3v1.py          # 3v1 environment implementation
├── policies/
│   ├── __init__.py            # Policy exports
│   ├── pursuer_policies.py    # Pursuer policy implementations
│   └── evader_policies.py     # Evader policy implementations
├── example_3v1_policies.py    # Example usage script
└── POLICIES_README.md         # This file
```

## Extending the Policies

To add a new policy:

1. Create a new class inheriting from `PursuerPolicy` or `EvaderPolicy`
2. Implement the `get_action()` method
3. Add to the policy dictionary (`PURSUER_POLICIES` or `EVADER_POLICIES`)
4. Export in `__init__.py`

Example:

```python
class MyCustomPursuerPolicy(PursuerPolicy):
    def get_action(self, obs, agent_idx, state, key):
        # Your custom logic here
        return action

# Add to dictionary
PURSUER_POLICIES["my_custom"] = MyCustomPursuerPolicy
```
