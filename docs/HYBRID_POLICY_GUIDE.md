# 混合策略训练使用指南

## 概述

这个功能允许你在训练新模型时，将**预训练模型的输出**与**当前训练模型的输出**加权混合，作为最终的动作。

**应用场景：**
- 你已经有一个训练好的基础模型
- 现在想改变观测配置（添加/删除观测项）
- 新模型可以从预训练模型中获得指导，加速训练

---

## 架构设计

### 数据流

```
环境观测
    ├─> pretrained_policy 观测组 ─> 预训练模型 ─> action_pretrained
    └─> policy 观测组 ─────────────> 当前训练模型 ─> action_current
                                                            ↓
                                        action_final = α·action_pretrained + β·action_current
                                                            ↓
                                                        执行动作
```

### 关键组件

1. **观测配置（3 个独立组）：**
   - `pretrained_policy`: 预训练模型使用的观测（与原始训练时一致，不可修改）
   - `policy`: 当前训练模型使用的观测（可以自由修改）
   - `critic`: Critic 网络使用的观测（可选特权信息）

2. **混合动作（HybridPolicyAction）：**
   - 加载预训练模型权重
   - 在每步推理时计算两个模型的输出
   - 按权重混合后作为最终动作

---

## 快速开始

### 步骤 1: 准备预训练模型

#### 方法 A: 使用 WandB（推荐）

```bash
# 1. 训练原始模型
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --num_envs=4096

# 2. 模型会自动上传到 WandB
# 记录 artifact 路径，格式如: wandb://your-entity/beyond_mimic/model-abc123:v0
```

#### 方法 B: 使用本地文件

```bash
# 找到训练好的模型文件
ls logs/rsl_rl/g1_flat/

# 复制模型到指定位置
cp logs/rsl_rl/g1_flat/2025-01-13_12-34-56/model_10000.pt \
   /home/xlh/beyond_mimic/pretrained_models/g1_base_v1.pt
```

---

### 步骤 2: 配置混合策略环境

编辑 `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/hybrid_env_cfg.py`：

#### 2.1 从 WandB 加载

```python
@configclass
class MyHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ✅ 设置 WandB artifact 路径
        self.actions.joint_pos.pretrained_model_path = (
            "wandb://your-entity/beyond_mimic/model-abc123:v0"
        )
        
        # ✅ 设置混合权重
        self.actions.joint_pos.pretrained_policy_coef = 0.7  # 预训练模型 70%
        self.actions.joint_pos.current_policy_coef = 0.3     # 当前模型 30%
        
        # ✅ 修改当前训练模型的观测（可选）
        # 例如：移除某些观测项
        # self.observations.policy.motion_anchor_pos_b = None
        
        # ✅ 预训练模型的观测保持不变（self.observations.pretrained_policy）
```

#### 2.2 从本地文件加载

```python
@configclass
class MyHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ✅ 设置本地模型路径
        self.actions.joint_pos.pretrained_model_path = (
            "/home/xlh/beyond_mimic/pretrained_models/g1_base_v1.pt"
        )
        
        # 其他配置同上...
```

---

### 步骤 3: 注册新环境（如果需要）

编辑 `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`：

```python
gym.register(
    id="Tracking-Flat-G1-My-Hybrid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hybrid_env_cfg.MyHybridEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)
```

---

### 步骤 4: 开始训练

```bash
# 使用预定义的混合环境
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Hybrid-Wandb-v0 \
    --num_envs=4096 \
    --headless

# 或使用你自定义的环境
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-My-Hybrid-v0 \
    --num_envs=4096
```

---

## 高级功能

### 课程学习：动态调整混合权重

在训练过程中逐步降低预训练模型的权重，提高新模型的权重。

#### 方法 1: 使用预定义的课程学习环境

```bash
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Hybrid-Curriculum-v0 \
    --num_envs=4096
```

这个环境初始设置为：
- 预训练模型：90%
- 当前模型：10%

#### 方法 2: 在训练脚本中动态调整

修改 `scripts/rsl_rl/train.py`（或创建自定义训练脚本）：

```python
# 在训练循环中添加
def update_hybrid_weights(env, current_iteration, total_iterations):
    """根据训练进度调整混合权重"""
    progress = current_iteration / total_iterations
    
    # 线性衰减：预训练权重从 0.9 降到 0.1
    pretrained_coef = max(0.1, 0.9 - 0.8 * progress)
    current_coef = min(0.9, 0.1 + 0.8 * progress)
    
    # 应用新权重
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    action_term.cfg.pretrained_policy_coef = pretrained_coef
    action_term.cfg.current_policy_coef = current_coef
    
    print(f"[Curriculum] Iteration {current_iteration}/{total_iterations}: "
          f"pretrained={pretrained_coef:.2f}, current={current_coef:.2f}")

# 在 PPO 训练循环中调用
for iteration in range(total_iterations):
    # 更新权重
    update_hybrid_weights(env, iteration, total_iterations)
    
    # 正常训练
    agent.learn(...)
```

---

## 修改观测配置

### 场景：新模型使用不同的观测

```python
@configclass
class MyCustomObsHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 设置预训练模型路径
        self.actions.joint_pos.pretrained_model_path = "..."
        
        # ===== 修改当前训练模型的观测 =====
        
        # 方法 1: 移除某些观测项
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        
        # 方法 2: 添加新的观测项（需要先在 mdp/observations.py 中定义）
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from whole_body_tracking.tasks.tracking import mdp
        
        self.observations.policy.my_custom_obs = ObsTerm(
            func=mdp.my_custom_observation_function,
            params={"param1": "value1"}
        )
        
        # ===== 预训练模型的观测不受影响 =====
        # self.observations.pretrained_policy 保持不变
```

---

## 调试与监控

### 1. 验证模型加载

训练开始时应看到：

```
[HybridPolicyAction] Loaded pretrained model from WandB: wandb://...
[HybridPolicyAction] Model architecture: obs_dim=XXX, action_dim=23
```

### 2. 监控混合权重

在 WandB 中记录混合权重（需要修改训练脚本）：

```python
# 在训练循环中
wandb.log({
    "hybrid/pretrained_coef": action_term.cfg.pretrained_policy_coef,
    "hybrid/current_coef": action_term.cfg.current_policy_coef,
})
```

### 3. 可视化两个模型的输出

```python
# 在 HybridPolicyAction.process_actions() 中
wandb.log({
    "actions/pretrained_mean": self._pretrained_actions.mean().item(),
    "actions/current_mean": actions.mean().item(),
    "actions/combined_mean": combined_actions.mean().item(),
})
```

---

## 常见问题

### Q1: 预训练模型观测维度不匹配

**错误：**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x120 and 150x256)
```

**原因：**
预训练模型期望的观测维度（150）与当前 `pretrained_policy` 观测组的维度（120）不匹配。

**解决：**
确保 `observations.pretrained_policy` 的配置与原始训练时完全一致。

---

### Q2: WandB artifact 下载失败

**错误：**
```
wandb.errors.CommError: Artifact not found
```

**解决：**
1. 检查 artifact 路径是否正确
2. 确认 WandB 登录状态：`wandb login`
3. 验证 artifact 存在：`wandb artifact get <artifact_path>`

---

### Q3: 训练不稳定

**现象：**
混合策略训练时奖励振荡剧烈。

**解决：**
1. 降低学习率：`learning_rate = 1e-4`
2. 增加预训练模型权重：`pretrained_policy_coef = 0.8`
3. 使用课程学习，逐步过渡

---

## 预定义环境总览

| 环境 ID | 说明 | 预训练权重 | 当前权重 |
|:---|:---|:---:|:---:|
| `Tracking-Flat-G1-Hybrid-v0` | 基础混合策略 | 0.5 | 0.5 |
| `Tracking-Flat-G1-Hybrid-Curriculum-v0` | 课程学习（初始） | 0.9 | 0.1 |
| `Tracking-Flat-G1-Hybrid-Wandb-v0` | 从 WandB 加载 | 0.5 | 0.5 |
| `Tracking-Flat-G1-Hybrid-Local-v0` | 从本地加载 | 0.5 | 0.5 |

---

## 完整示例

### 完整训练流程

```bash
# 1. 训练基础模型
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --num_envs=4096 \
    --max_iterations=10000

# 2. 记录 WandB artifact 路径
# 例如: wandb://myteam/beyond_mimic/model-run123:v0

# 3. 修改 hybrid_env_cfg.py，设置预训练模型路径

# 4. 训练混合策略模型
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Hybrid-Curriculum-v0 \
    --num_envs=4096 \
    --max_iterations=10000

# 5. 逐步降低预训练权重（可选，通过自定义训练脚本实现）
```

---

## 扩展阅读

- **观测配置**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- **混合动作实现**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hybrid_actions.py`
- **环境注册**: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`
