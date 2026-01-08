from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

# 导入 Command 类以获取类型提示 (可选)
# from .commands import MotionCommand 

@configclass
class MotionTrackingActionCfg(ActionTermCfg):
    """Configuration for motion tracking action term."""
    class_type: type = MISSING
    
    asset_name: str = MISSING
    """Name of the asset in the environment."""
    
    joint_names: list[str] | str = MISSING
    """List of joint names or regex to match."""
    
    command_name: str = "motion"
    """Name of the command term that provides the reference motion."""
    
    scale: float | dict[str, float] = 1.0
    """Scale factor for the policy action."""
    
    policy_coef: float = 1.0
    """Coefficient for the policy action (Action * Scale * Policy_Coef)."""
    
    motion_coef: float = 1.0
    """Coefficient for the dataset reference motion (Ref * Motion_Coef)."""


class MotionTrackingAction(ActionTerm):
    """Action term that combines policy output with reference motion."""
    
    cfg: MotionTrackingActionCfg

    def __init__(self, cfg: MotionTrackingActionCfg, env):
        super().__init__(cfg, env)
        # 解析关节索引
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # --- 处理缩放 (Scale) ---
        # 这是一个简化版的缩放处理，支持标量和简单字典
        if isinstance(cfg.scale, (float, int)):
            self._action_scale = torch.full(
                (self._num_envs, self._num_joints), cfg.scale, device=self.device
            )
        elif isinstance(cfg.scale, dict):
            # 处理字典类型的scale (对应 G1_ACTION_SCALE)
            self._action_scale = torch.ones(
                (self._num_envs, self._num_joints), device=self.device
            )
            for joint_name, scale in cfg.scale.items():
                # 找到匹配的关节索引并赋值
                ids, _ = self._asset.find_joints(joint_name)
                # 注意：这里需要映射回 self._joint_ids 的相对索引，简化起见假设是对齐的或直接操作
                # 实际更加健壮的写法通常较长，这里假设 joint_names=[".*"] 且 scale 覆盖了所有
                for id in ids:
                    if id in self._joint_ids:
                        idx = self._joint_ids.index(id) # 这在大量关节时可能慢，但在初始化时只运行一次
                        self._action_scale[:, idx] = scale
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

    def process_actions(self, actions: torch.Tensor) -> ArticulationActions:
        # 1. 获取参考动作 (从 Command Manager)
        command = self._env.command_manager.get_term(self.cfg.command_name)
        # command.joint_pos 是 (num_envs, total_joints)，我们需要提取当前受控关节的部分
        ref_joint_pos = command.joint_pos[:, self._joint_ids]

        # 2. 计算目标位置
        # 你的逻辑: target = action * coeff1 + dataset_pos * coeff2
        # 注意: actions 通常是归一化的，需要先乘以 scale
        processed_actions = actions * self._action_scale
        
        target_pos = (processed_actions * self.cfg.policy_coef) + (ref_joint_pos * self.cfg.motion_coef)

        # 3. 保存用于 Observation 的中间变量
        self._raw_actions[:] = actions
        self._processed_actions[:] = target_pos

        # 4. 返回 IsaacLab 标准动作结构
        return ArticulationActions(
            joint_positions=target_pos,
            joint_velocities=torch.zeros_like(target_pos), # 如果需要也可以混合速度
            joint_efforts=torch.zeros_like(target_pos)
        )
    
MotionTrackingActionCfg.class_type = MotionTrackingAction