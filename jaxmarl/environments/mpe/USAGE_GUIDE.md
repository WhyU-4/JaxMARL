# 3v1 Pursuit-Evasion Environment - User Guide

## 简介 (Introduction)

基于MPE的simple_tag环境，我们创建了一个新的3v1追逃问题研究环境，包含：

Based on MPE's simple_tag environment, we have created a new 3v1 pursuit-evasion research environment that includes:

- **新环境**: `MPE_simple_tag_3v1` - 3个追捕者对1个逃逸者
- **模块化追捕者策略系统** - 可轻松替换不同的追捕策略
- **多级别启发式逃逸策略** - 4个不同难度级别的逃逸策略

- **New Environment**: `MPE_simple_tag_3v1` - 3 pursuers vs 1 evader
- **Modular Pursuer Policy System** - Easy strategy replacement for pursuers
- **Multi-level Heuristic Evader Strategies** - 4 difficulty levels

## 环境特性 (Environment Features)

### 智能体 (Agents)

- **3个追捕者 (Pursuers/Adversaries)**: `adversary_0`, `adversary_1`, `adversary_2`
  - 速度较慢但可以协作
  - 观测空间：16维（包括位置、速度、其他智能体信息）
  
- **1个逃逸者 (Evader)**: `agent_0`
  - 速度更快，更灵活
  - 观测空间：14维

### 奖励机制 (Reward Structure)

- **追捕者**: 抓到逃逸者时获得正奖励 (+10)
- **逃逸者**: 被抓时获得负奖励 (-10)，越界时受到惩罚

## 快速开始 (Quick Start)

### 1. 创建环境 (Create Environment)

```python
from jaxmarl import make
import jax

# 创建环境
env = make("MPE_simple_tag_3v1")

# 重置环境
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

print(f"智能体列表: {env.agents}")
print(f"追捕者: {env.adversaries}")
print(f"逃逸者: {env.good_agents}")
```

### 2. 使用启发式策略 (Using Heuristic Policies)

```python
from jaxmarl.environments.mpe.policies import (
    get_pursuer_policy,
    get_evader_policy,
)

# 创建策略
pursuer_policy = get_pursuer_policy("greedy")  # 贪婪追捕
evader_policy = get_evader_policy("advanced")  # 高级逃逸

# 运行一个episode
for step in range(100):
    key, *action_keys = jax.random.split(key, len(env.agents) + 1)
    actions = {}
    
    # 追捕者动作
    for i, agent_name in enumerate(env.adversaries):
        actions[agent_name] = pursuer_policy.get_action(
            obs[agent_name], i, state, action_keys[i]
        )
    
    # 逃逸者动作
    actions[env.good_agents[0]] = evader_policy.get_action(
        obs[env.good_agents[0]], state, action_keys[-1]
    )
    
    # 执行步骤
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, infos = env.step(step_key, state, actions)
    
    if dones["__all__"]:
        break
```

## 追捕者策略 (Pursuer Policies)

位置：`jaxmarl/environments/mpe/policies/pursuer_policies.py`

### 可用策略 (Available Strategies)

1. **随机策略 (Random)**: `"random"`
   - 完全随机选择动作
   - 用作基准比较

2. **贪婪策略 (Greedy)**: `"greedy"`
   - 每个追捕者独立地直接追向逃逸者
   - 简单有效的基础策略

3. **协同策略 (Coordinated)**: `"coordinated"`
   - 追捕者协调包围逃逸者
   - 分别从不同角度接近（0°, 120°, 240°）

### 使用示例 (Usage Example)

```python
from jaxmarl.environments.mpe.policies import get_pursuer_policy

# 创建策略
policy = get_pursuer_policy("coordinated")

# 获取动作
action = policy.get_action(
    obs=observation,      # 观测
    agent_idx=0,          # 智能体索引 (0, 1, 2)
    state=env_state,      # 环境状态
    key=rng_key           # 随机数生成器
)
```

## 逃逸者策略 (Evader Policies)

位置：`jaxmarl/environments/mpe/policies/evader_policies.py`

### 难度级别 (Difficulty Levels)

1. **级别0 - 随机策略 (Random)**: `"random"`
   - 完全随机移动
   - 最简单的对手

2. **级别1 - 基础策略 (Basic)**: `"basic"`
   - 远离最近的追捕者
   - 简单的反应式策略

3. **级别2 - 中级策略 (Intermediate)**: `"intermediate"`
   - 远离所有追捕者的质心
   - 对协同追捕有更好的应对

4. **级别3 - 高级策略 (Advanced)**: `"advanced"`
   - 加权逃离（近的追捕者权重更大）
   - 边界避让
   - 利用追捕者之间的空隙
   - 添加随机性避免可预测
   - 最具挑战性的对手

### 使用示例 (Usage Example)

```python
from jaxmarl.environments.mpe.policies import get_evader_policy

# 创建不同难度的策略
easy_evader = get_evader_policy("basic")
hard_evader = get_evader_policy("advanced")

# 获取动作
action = evader_policy.get_action(
    obs=observation,      # 观测
    state=env_state,      # 环境状态
    key=rng_key           # 随机数生成器
)
```

## 策略模块化替换 (Modular Policy Replacement)

### 为什么需要模块化？(Why Modular?)

- ✅ **易于测试**: 快速切换不同策略进行对比
- ✅ **灵活训练**: 对学习型策略提供不同难度的对手
- ✅ **代码复用**: 统一的接口，易于扩展

### 自定义策略 (Custom Policies)

创建自己的追捕者策略：

```python
from jaxmarl.environments.mpe.policies.pursuer_policies import PursuerPolicy

class MyCustomPursuerPolicy(PursuerPolicy):
    def get_action(self, obs, agent_idx, state, key):
        # 实现你的策略逻辑
        # 返回动作: 0=no-op, 1=left, 2=right, 3=down, 4=up
        return action

# 注册策略
from jaxmarl.environments.mpe.policies.pursuer_policies import PURSUER_POLICIES
PURSUER_POLICIES["my_custom"] = MyCustomPursuerPolicy

# 使用策略
policy = get_pursuer_policy("my_custom")
```

创建自己的逃逸者策略：

```python
from jaxmarl.environments.mpe.policies.evader_policies import EvaderPolicy

class MyCustomEvaderPolicy(EvaderPolicy):
    def get_action(self, obs, state, key):
        # 实现你的策略逻辑
        return action

# 注册和使用方法相同
```

## 运行示例 (Running Examples)

### 示例1: 单个episode (Single Episode)

```bash
cd /home/runner/work/JaxMARL/JaxMARL
python -c "
from jaxmarl import make
from jaxmarl.environments.mpe.policies import get_pursuer_policy, get_evader_policy
import jax

env = make('MPE_simple_tag_3v1')
pursuer_policy = get_pursuer_policy('greedy')
evader_policy = get_evader_policy('intermediate')

key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

# 运行100步
for step in range(100):
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
    
    if dones['__all__']:
        print(f'Episode ended at step {step+1}')
        break
"
```

### 示例2: 策略比较 (Policy Comparison)

```bash
python -m jaxmarl.environments.mpe.example_3v1_policies
```

这将运行所有策略组合的比较测试。

## 训练建议 (Training Recommendations)

### 渐进式训练 (Progressive Training)

1. **初期训练** - 对抗基础逃逸者
   ```python
   evader_policy = get_evader_policy("basic")
   ```

2. **中期训练** - 对抗中级逃逸者
   ```python
   evader_policy = get_evader_policy("intermediate")
   ```

3. **后期训练** - 对抗高级逃逸者
   ```python
   evader_policy = get_evader_policy("advanced")
   ```

### 课程学习 (Curriculum Learning)

```python
# 根据性能动态调整对手难度
if avg_success_rate > 0.8:
    evader_difficulty = "advanced"
elif avg_success_rate > 0.5:
    evader_difficulty = "intermediate"
else:
    evader_difficulty = "basic"

evader_policy = get_evader_policy(evader_difficulty)
```

## 测试 (Testing)

运行测试套件：

```bash
pytest tests/mpe/test_simple_tag_3v1.py -v
```

## 文件结构 (File Structure)

```
jaxmarl/environments/mpe/
├── simple_tag_3v1.py              # 3v1环境实现
├── policies/
│   ├── __init__.py                # 策略导出
│   ├── pursuer_policies.py        # 追捕者策略
│   └── evader_policies.py         # 逃逸者策略
├── example_3v1_policies.py        # 使用示例
└── POLICIES_README.md             # 详细文档

tests/mpe/
└── test_simple_tag_3v1.py         # 测试套件
```

## 性能指标 (Performance Metrics)

在比较不同策略时，可以关注以下指标：

- **捕获次数** (Captures): 追捕者成功捕获逃逸者的次数
- **平均回报** (Average Reward): 双方的累积奖励
- **episode长度** (Episode Length): 完成episode所需的步数
- **成功率** (Success Rate): 多个episode的捕获成功率

## 常见问题 (FAQ)

**Q: 如何修改环境参数？**

A: 创建环境时传入参数：
```python
env = make("MPE_simple_tag_3v1", num_good_agents=1, num_adversaries=3, num_obs=2)
```

**Q: 策略可以使用学习到的参数吗？**

A: 可以！继承基类并在策略内部维护神经网络参数即可。

**Q: 如何可视化环境？**

A: 使用MPE的可视化工具：
```python
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
viz = MPEVisualizer(env, state)
```

## 贡献 (Contributing)

欢迎贡献新的策略实现！请遵循现有的代码风格和接口规范。

## 许可 (License)

遵循JaxMARL项目的Apache 2.0许可证。
