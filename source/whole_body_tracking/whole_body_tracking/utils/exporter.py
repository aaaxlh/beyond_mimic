# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
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

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    """
    将环境配置和机器人参数作为元数据附加到 ONNX 模型中。

    这使得部署端（如 C++ 推理环境）可以从 ONNX 文件中直接读取关节名称、
    默认位置、观测历史长度等关键信息，无需额外的配置文件。

    Args:
        env: 包含完整配置信息的 RL 环境实例。
        run_path: WandB 的运行路径或实验标识符。
        path: ONNX 文件的保存目录。
        filename: ONNX 文件名。
    """
    onnx_path = os.path.join(path, filename)

    # 1. 获取观测项名称
    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    # 2. 获取观测历史长度 (History Length)
    # 如果全局设置了 history_length，则所有观测项使用相同长度
    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        # 否则，逐个获取每个观测项的 history_length
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            # 0 表示没有历史，即长度为 1 (当前帧)
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    # 3. 构建元数据字典
    # 这些数据将被序列化为字符串存储在 ONNX 模型中
    metadata = {
        "run_path": run_path,
        # 机器人关节物理参数
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        # 管理器配置
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        # 运动追踪任务特定参数
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    # 4. 加载现有的 ONNX 模型
    model = onnx.load(onnx_path)

    # 5. 将元数据写入模型的 metadata_props 字段
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        # 列表类型转换为 CSV 格式字符串，数值/字符串直接转换
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    # 6. 保存带有元数据的模型
    onnx.save(model, onnx_path)
