from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


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

        # --- 兼容不同 IsaacLab 版本：确保 num_envs 可用 ---
        if not hasattr(self, "_num_envs"):
            num_envs = getattr(env, "num_envs", None)
            if num_envs is None and hasattr(env, "scene"):
                num_envs = getattr(env.scene, "num_envs", None)
            if num_envs is None:
                raise AttributeError("Cannot infer num_envs from env. Expected env.num_envs or env.scene.num_envs.")
            self._num_envs = int(num_envs)

        # 解析关节索引
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)

        # 统一成 python list[int]，便于后续索引/切片
        if isinstance(self._joint_ids, torch.Tensor):
            self._joint_ids_list = [int(x) for x in self._joint_ids.tolist()]
        else:
            self._joint_ids_list = [int(x) for x in self._joint_ids]

        self._num_joints = len(self._joint_ids_list)
        self._joint_id_to_local = {jid: i for i, jid in enumerate(self._joint_ids_list)}

        # buffers (ActionTerm 抽象接口要求提供 raw/processed)
        self._raw_actions = torch.zeros((self._num_envs, self._num_joints), device=self.device)
        self._processed_actions = torch.zeros((self._num_envs, self._num_joints), device=self.device)

        # 缓存目标速度（用于扭矩计算的 qd_target）
        self._target_joint_vel = torch.zeros((self._num_envs, self._num_joints), device=self.device)

        # --- 处理缩放 (Scale) ---
        if isinstance(cfg.scale, (float, int)):
            self._scale = torch.full((self._num_envs, self._num_joints), float(cfg.scale), device=self.device)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones((self._num_envs, self._num_joints), device=self.device)
            for joint_name_expr, s in cfg.scale.items():
                ids, _ = self._asset.find_joints(joint_name_expr)
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                for jid in ids:
                    jid = int(jid)
                    if jid in self._joint_id_to_local:
                        self._scale[:, self._joint_id_to_local[jid]] = float(s)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        # 将要下发到 articulation 的 action（pos/vel/effort）
        self._actions = ArticulationActions(
            joint_positions=self._processed_actions,
            joint_velocities=self._target_joint_vel,
            joint_efforts=torch.zeros_like(self._processed_actions),
        )

    # --------- ActionTerm required interface ---------

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def apply_actions(self) -> None:
        """Write the processed actions into the simulated asset."""
        # 优先使用 IsaacLab 标准接口
        if hasattr(self._asset, "apply_action"):
            self._asset.apply_action(self._actions)
            return

        # 兼容不同版本 Articulation API（尽量写得鲁棒一点）
        if hasattr(self._asset, "set_joint_position_target"):
            self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids_list)
        if hasattr(self._asset, "set_joint_velocity_target"):
            self._asset.set_joint_velocity_target(self._target_joint_vel, joint_ids=self._joint_ids_list)
        if hasattr(self._asset, "set_joint_effort_target"):
            self._asset.set_joint_effort_target(torch.zeros_like(self._processed_actions), joint_ids=self._joint_ids_list)

    # --------- your action logic ---------

    def process_actions(self, actions: torch.Tensor) -> ArticulationActions:
        # 1) 取参考动作（来自 CommandManager 的 motion term）
        command = self._env.command_manager.get_term(self.cfg.command_name)

        # command.joint_pos/vel: (num_envs, total_joints)
        ref_joint_pos = command.joint_pos[:, self._joint_ids_list]
        ref_joint_vel = command.joint_vel[:, self._joint_ids_list]

        # 2) 位置目标：policy residual + reference
        processed_actions = actions * self._scale
        target_pos = processed_actions * self.cfg.policy_coef + ref_joint_pos * self.cfg.motion_coef

        # 3) 目标速度：直接用数据集参考速度（你要改的就是这里）
        target_vel = ref_joint_vel

        # 4) 缓存（供 observation / apply_actions / debug 使用）
        self._raw_actions[:] = actions
        self._processed_actions[:] = target_pos
        self._target_joint_vel[:] = target_vel

        # 5) 更新下发对象并返回
        self._actions.joint_positions = self._processed_actions
        self._actions.joint_velocities = self._target_joint_vel
        self._actions.joint_efforts = torch.zeros_like(self._processed_actions)
        return self._actions


# 这行可留可不留；你在 env cfg 里已经显式传 class_type 了
MotionTrackingActionCfg.class_type = MotionTrackingAction