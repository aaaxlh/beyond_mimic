"""Configuration for G1 hybrid policy training (pretrained + current model)."""

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.mdp import hybrid_actions as mdp_hybrid


@configclass
class G1FlatHybridEnvCfg(G1FlatEnvCfg):
    """
    混合策略训练配置
    
    使用场景：
    1. 你已经训练了一个基础模型（使用原始观测配置）
    2. 现在想在新的观测配置下训练新模型
    3. 新模型的输出与旧模型的输出加权混合后作为最终动作
    
    配置说明：
    - pretrained_policy_coef: 预训练模型输出的权重（0-1）
    - current_policy_coef: 当前训练模型输出的权重（0-1）
    - pretrained_model_path: 预训练模型的路径
      * WandB: "wandb://<entity>/<project>/<artifact>:<version>"
      * 本地: "/path/to/model.pt"
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # ===== 混合动作配置 =====
        # 替换默认的 JointPositionAction 为 HybridPolicyAction
        self.actions.joint_pos = mdp_hybrid.HybridPolicyActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            
            # ✅ 混合系数（可根据训练阶段调整）
            pretrained_policy_coef=0.5,  # 预训练模型权重
            current_policy_coef=0.5,     # 当前训练模型权重
            
            # ✅ 预训练模型路径（需要修改为你的实际路径）
            pretrained_model_path="",  # TODO: 设置为你的模型路径
            
            # ✅ 观测组名称
            pretrained_obs_group="pretrained_policy",  # 预训练模型使用的观测
            current_obs_group="policy",                # 当前训练模型使用的观测
        )
        



@configclass
class G1FlatHybridCurriculumEnvCfg(G1FlatHybridEnvCfg):
    """
    课程学习版本：逐步降低预训练模型的权重
    
    训练流程：
    1. 初期：pretrained_coef=0.9, current_coef=0.1（依赖预训练模型）
    2. 中期：pretrained_coef=0.5, current_coef=0.5（平衡）
    3. 后期：pretrained_coef=0.1, current_coef=0.9（主要靠新模型）
    
    使用方法：
    在训练脚本中动态调整系数，例如：
    ```python
    # 根据训练进度调整
    progress = current_iteration / total_iterations
    pretrained_coef = max(0.1, 1.0 - progress)
    current_coef = min(0.9, progress)
    
    env.action_manager.get_term("joint_pos").cfg.pretrained_policy_coef = pretrained_coef
    env.action_manager.get_term("joint_pos").cfg.current_policy_coef = current_coef
    ```
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # 初始阶段：高度依赖预训练模型
        self.actions.joint_pos.pretrained_policy_coef = 0.9
        self.actions.joint_pos.current_policy_coef = 0.1


# ===== 使用示例 =====

# 1. 从 WandB 加载预训练模型
@configclass
class G1FlatHybridWandbEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.class_type = mdp_hybrid.HybridPolicyAction
        # TODO: 修改为你的 WandB artifact 路径
        self.actions.joint_pos.pretrained_model_path = (
            "wandb://2940562534-dalian-university-of-technology/train_mimic_23dofs/run-zcru65cr/files/model_15000.pt"
        )


# 2. 从本地文件加载预训练模型
@configclass
class G1FlatHybridLocalEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # TODO: 修改为你的本地模型路径
        self.actions.joint_pos.pretrained_model_path = (
            "/home/xlh/beyond_mimic/logs/rsl_rl/g1_flat/model_10000.pt"
        )


# 3. 自定义新模型的观测配置
@configclass
class G1FlatHybridCustomObsEnvCfg(G1FlatHybridEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ===== 修改当前训练模型的观测 =====
        # 例如：移除某些观测项
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        
        # 或者添加新的观测项（需要在 mdp/observations.py 中定义）
        # from isaaclab.managers import ObsTerm
        # self.observations.policy.my_new_obs = ObsTerm(
        #     func=mdp.my_new_observation_function,
        #     params={...}
        # )
        
        # ===== 预训练模型的观测保持不变 =====
        # self.observations.pretrained_policy 不需要修改
