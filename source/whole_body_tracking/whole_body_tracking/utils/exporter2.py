# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import copy

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _TorchPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand
from whole_body_tracking.utils.exporter import list_to_csv_str


def export_motion_policy_as_jit(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.jit",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _JitMotionPolicyExporter(env, actor_critic, normalizer)
    policy_exporter.export(path, filename)


def attach_jit_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.jit") -> None:
    """
    将环境配置和机器人参数作为元数据附加到 JIT 模型中。
    """
    jit_path = os.path.join(path, filename)

    # 1. 获取观测项名称
    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    # 2. 获取观测历史长度
    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    # 3. 构建元数据字典
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    # 4. 准备 extra_files
    extra_files = {}
    for k, v in metadata.items():
        extra_files[k] = list_to_csv_str(v) if isinstance(v, list) else str(v)

    # 5. 加载并重新保存
    if os.path.exists(jit_path):
        # map_location="cpu" is important to avoid needing a GPU if not available
        model = torch.jit.load(jit_path, map_location="cpu")
        model.save(jit_path, _extra_files=extra_files)
    else:
        print(f"[WARN] JIT model not found at {jit_path}, skipping metadata attachment.")




class _JitMotionPolicyExporter(_TorchPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None):
        super().__init__(actor_critic, normalizer)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        """
        Args:
            obs_prop: (1, 93) Proprioception observations [base_lin_vel, base_ang_vel, joint_pos, joint_vel, actions]
            body_pos: (1, 3) Robot base position in world frame
            body_quat: (1, 4) Robot base orientation in world frame
            time_step: (1, 1) Current time step
        """
        # Clamp time_step
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )
