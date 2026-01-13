# 混合策略训练 - 快速配置指南

## 核心配置

### 1. 修改混合策略环境配置

编辑文件：`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/hybrid_env_cfg.py`

找到或创建你的配置类：

```python
@configclass
class G1FlatHybridWandbEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # TODO: 修改为你的 WandB artifact 路径
        self.actions.joint_pos.pretrained_model_path = (
            "wandb://your-entity/beyond_mimic/model-abc123:v0"
        )
```

### 2. 运行训练

```bash
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Hybrid-Wandb-v0 \
    --num_envs=4096
```

---

## 关键参数说明

### 混合权重

```python
pretrained_policy_coef = 0.5  # 预训练模型权重（0-1）
current_policy_coef = 0.5     # 当前训练模型权重（0-1）
```

**推荐设置：**
- 初期：`pretrained=0.8, current=0.2`（依赖预训练模型）
- 中期：`pretrained=0.5, current=0.5`（平衡）
- 后期：`pretrained=0.2, current=0.8`（主要靠新模型）

### 观测配置

```python
# 预训练模型的观测（不要修改，与原始训练保持一致）
self.observations.pretrained_policy  

# 当前训练模型的观测（可以自由修改）
self.observations.policy
```

---

## 快速示例

### 示例 1：从本地文件加载

```python
@configclass
class MyHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 本地模型路径
        self.actions.joint_pos.pretrained_model_path = (
            "/home/xlh/beyond_mimic/logs/rsl_rl/g1_flat/model_10000.pt"
        )
        
        # 混合权重
        self.actions.joint_pos.pretrained_policy_coef = 0.7
        self.actions.joint_pos.current_policy_coef = 0.3
```

### 示例 2：从 WandB 加载

```python
@configclass
class MyHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # WandB artifact 路径
        self.actions.joint_pos.pretrained_model_path = (
            "wandb://myteam/beyond_mimic/model-run123:v0"
        )
        
        # 混合权重
        self.actions.joint_pos.pretrained_policy_coef = 0.6
        self.actions.joint_pos.current_policy_coef = 0.4
```

---

## 获取预训练模型路径

### 方法 1：从 WandB UI

1. 打开你的 WandB 项目页面
2. 找到训练好的 run
3. 进入 "Artifacts" 标签页
4. 复制 artifact 路径（格式：`entity/project/artifact-name:version`）
5. 添加前缀 `wandb://`

**完整路径示例：**
```
wandb://myteam/beyond_mimic/model-abc123:v0
```

### 方法 2：从本地日志

```bash
# 查看训练日志目录
ls logs/rsl_rl/g1_flat/

# 找到模型文件（通常是 .pt 文件）
ls logs/rsl_rl/g1_flat/2025-01-13_12-34-56/

# 使用完整路径
/home/xlh/beyond_mimic/logs/rsl_rl/g1_flat/2025-01-13_12-34-56/model_10000.pt
```

---

## 修改当前训练模型的观测

如果你想让新模型使用不同的观测配置：

```python
@configclass
class MyCustomObsHybridEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 设置预训练模型路径
        self.actions.joint_pos.pretrained_model_path = "..."
        
        # 移除某些观测项（示例）
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        
        # 注意：预训练模型的观测（pretrained_policy）保持不变
```

---

## 故障排查

### 问题 1：模型加载失败

```
FileNotFoundError: Pretrained model not found
```

**检查：**
- 文件路径是否正确
- WandB artifact 是否存在
- 是否已登录 WandB：`wandb login`

### 问题 2：观测维度不匹配

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**检查：**
- `observations.pretrained_policy` 的配置是否与原始训练时一致
- 不要修改 `pretrained_policy` 的观测项

---

## 完整工作流程

```bash
# 1. 训练基础模型（如果还没有）
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-v0 \
    --num_envs=4096

# 2. 记录模型路径（从 WandB 或本地日志）

# 3. 修改 hybrid_env_cfg.py，设置 pretrained_model_path

# 4. 训练混合策略模型
python scripts/rsl_rl/train.py \
    --task=Tracking-Flat-G1-Hybrid-Wandb-v0 \
    --num_envs=4096
```

---

## 可用的预定义环境

| 环境 ID | 说明 |
|:---|:---|
| `Tracking-Flat-G1-Hybrid-v0` | 基础混合策略（需手动设置路径） |
| `Tracking-Flat-G1-Hybrid-Curriculum-v0` | 课程学习版本 |
| `Tracking-Flat-G1-Hybrid-Wandb-v0` | WandB 加载模板 |
| `Tracking-Flat-G1-Hybrid-Local-v0` | 本地文件加载模板 |

---

## 需要帮助？

查看完整文档：`docs/HYBRID_POLICY_GUIDE.md`
