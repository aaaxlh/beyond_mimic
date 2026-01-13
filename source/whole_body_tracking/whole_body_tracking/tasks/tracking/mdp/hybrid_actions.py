"""Hybrid action term that combines pretrained model with current training model."""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


@configclass
class HybridPolicyActionCfg(ActionTermCfg):
    """Configuration for hybrid policy action term."""
    
    class_type: type = MISSING
    
    asset_name: str = MISSING
    """Name of the asset in the environment."""
    
    joint_names: list[str] | str = MISSING
    """List of joint names or regex to match."""
    
    scale: float | dict[str, float] = 1.0
    """Scale factor for the actions."""
    
    use_default_offset: bool = True
    """Whether to use default joint positions as offset."""
    
    # 混合策略参数
    pretrained_policy_coef: float = 0.5
    """Coefficient for pretrained policy output (范围 0-1)."""
    
    current_policy_coef: float = 1.0
    """Coefficient for current training policy output (范围 0-1)."""
    
    pretrained_model_path: str = ""
    """Path to pretrained model checkpoint (WandB artifact path or local path)."""
    
    pretrained_obs_group: str = "pretrained_policy"
    """Name of the observation group for pretrained model."""
    
    current_obs_group: str = "policy"
    """Name of the observation group for current training model."""


class HybridPolicyAction(ActionTerm):
    """
    Action term that combines outputs from a pretrained model and current training model.
    
    Final action = pretrained_output * pretrained_coef + current_output * current_coef
    """
    
    cfg: HybridPolicyActionCfg

    def __init__(self, cfg: HybridPolicyActionCfg, env):
        super().__init__(cfg, env)
        
        # 解析关节索引
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # 处理缩放 (Scale)
        if isinstance(cfg.scale, (float, int)):
            self._action_scale = torch.full(
                (self._num_envs, self._num_joints), cfg.scale, device=self.device
            )
        elif isinstance(cfg.scale, dict):
            self._action_scale = torch.ones(
                (self._num_envs, self._num_joints), device=self.device
            )
            for joint_name, scale in cfg.scale.items():
                ids, _ = self._asset.find_joints(joint_name)
                for id in ids:
                    if id in self._joint_ids:
                        idx = self._joint_ids.index(id)
                        self._action_scale[:, idx] = scale
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")
        
        # 获取默认关节位置偏移
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros(self._num_envs, self._num_joints, device=self.device)
        
        # 加载预训练模型
        self._pretrained_model = None
        if cfg.pretrained_model_path:
            self._load_pretrained_model(cfg.pretrained_model_path)
        
        # 存储预训练模型的输出（用于调试）
        self._pretrained_actions = torch.zeros(
            self._num_envs, self._num_joints, device=self.device
        )

    def _load_pretrained_model(self, model_path: str):
        """Load pretrained model from checkpoint."""
        import os
        
        # 检查是否是 WandB artifact 路径
        if model_path.startswith("wandb://"):
            self._load_from_wandb(model_path)
        elif os.path.exists(model_path):
            self._load_from_file(model_path)
        else:
            raise FileNotFoundError(f"Pretrained model not found: {model_path}")

    def _load_from_wandb(self, artifact_path: str):
        """Load model from WandB artifact.
        
        Args:
            artifact_path: Format "wandb://<entity>/<project>/<artifact>:<version>"
                          Example: "wandb://myteam/beyond_mimic/model-abc123:v0"
        """
        import wandb
        
        # 解析 artifact 路径
        # wandb://entity/project/artifact:version
        path_parts = artifact_path.replace("wandb://", "").split("/")
        if len(path_parts) < 3:
            raise ValueError(f"Invalid WandB artifact path: {artifact_path}")
        
        entity = path_parts[0]
        project = path_parts[1]
        artifact_name = "/".join(path_parts[2:])
        
        # 初始化 WandB（如果还没初始化）
        if wandb.run is None:
            wandb.init(entity=entity, project=project, job_type="inference")
        
        # 下载 artifact
        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()
        
        # 加载模型
        model_file = os.path.join(artifact_dir, "model.pt")
        if not os.path.exists(model_file):
            # 尝试查找 .pt 文件
            pt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
            if pt_files:
                model_file = os.path.join(artifact_dir, pt_files[0])
            else:
                raise FileNotFoundError(f"No .pt file found in artifact: {artifact_dir}")
        
        self._load_from_file(model_file)
        print(f"[HybridPolicyAction] Loaded pretrained model from WandB: {artifact_path}")

    def _load_from_file(self, file_path: str):
        """Load model from local file."""
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # 根据你的训练框架结构提取模型
        # 假设使用 RSL-RL，结构为 checkpoint["model_state_dict"] 或 checkpoint["actor"]
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "actor" in checkpoint:
            state_dict = checkpoint["actor"]
        else:
            state_dict = checkpoint
        
        # 创建模型实例（需要根据你的网络结构调整）
        from rsl_rl.modules import ActorCritic
        
        # TODO: 需要知道预训练模型的 obs_dim 和 action_dim
        # 这里假设可以从 checkpoint 中获取
        obs_dim = checkpoint.get("obs_dim", None)
        action_dim = self._num_joints
        
        if obs_dim is None:
            # 尝试从 state_dict 推断
            first_layer_key = [k for k in state_dict.keys() if "actor" in k and "weight" in k][0]
            obs_dim = state_dict[first_layer_key].shape[1]
        
        # 创建 Actor 网络
        from torch import nn
        
        class SimpleActor(nn.Module):
            def __init__(self, obs_dim, action_dim, hidden_dims=[256, 128, 64]):
                super().__init__()
                layers = []
                in_dim = obs_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    layers.append(nn.ELU())
                    in_dim = hidden_dim
                layers.append(nn.Linear(in_dim, action_dim))
                self.net = nn.Sequential(*layers)
            
            def forward(self, obs):
                return self.net(obs)
        
        self._pretrained_model = SimpleActor(obs_dim, action_dim).to(self.device)
        
        # 加载权重（只加载 actor 部分）
        actor_state_dict = {}
        for k, v in state_dict.items():
            if "actor" in k:
                # 移除 "actor." 前缀
                new_key = k.replace("actor.", "net.")
                actor_state_dict[new_key] = v
        
        self._pretrained_model.load_state_dict(actor_state_dict, strict=False)
        self._pretrained_model.eval()  # 设置为评估模式
        
        print(f"[HybridPolicyAction] Loaded pretrained model from file: {file_path}")
        print(f"[HybridPolicyAction] Model architecture: obs_dim={obs_dim}, action_dim={action_dim}")

    @torch.no_grad()
    def _get_pretrained_action(self) -> torch.Tensor:
        """Get action from pretrained model."""
        if self._pretrained_model is None:
            # 如果没有加载预训练模型，返回零动作
            return torch.zeros(self._num_envs, self._num_joints, device=self.device)
        
        # 获取预训练模型的观测
        obs_dict = self._env.observation_manager.compute()
        pretrained_obs = obs_dict.get(self.cfg.pretrained_obs_group, None)
        
        if pretrained_obs is None:
            raise ValueError(
                f"Observation group '{self.cfg.pretrained_obs_group}' not found. "
                f"Available groups: {list(obs_dict.keys())}"
            )
        
        # 前向传播
        pretrained_action = self._pretrained_model(pretrained_obs)
        
        return pretrained_action

    def process_actions(self, actions: torch.Tensor) -> ArticulationActions:
        """
        Process actions by combining current policy and pretrained policy.
        
        Args:
            actions: Current training policy output (num_envs, num_joints)
        
        Returns:
            Combined actions applied to the robot
        """
        # 1. 获取预训练模型的动作
        pretrained_actions = self._get_pretrained_action()
        self._pretrained_actions[:] = pretrained_actions
        
        # 2. 混合两个策略的输出
        combined_actions = (
            actions * self.cfg.current_policy_coef +
            pretrained_actions * self.cfg.pretrained_policy_coef
        )
        
        # 3. 应用缩放
        scaled_actions = combined_actions * self._action_scale
        
        # 4. 添加偏移（默认关节位置）
        target_positions = scaled_actions + self._offset
        
        # 5. 保存用于观测
        self._raw_actions[:] = actions
        self._processed_actions[:] = target_positions
        
        # 6. 返回关节动作
        return ArticulationActions(joint_positions=target_positions)
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term (clear action buffers)."""
        super().reset(env_ids)
        
        # 重置预训练模型的动作缓存
        if env_ids is None:
            self._pretrained_actions[:] = 0.0
        else:
            self._pretrained_actions[env_ids] = 0.0


# 注册配置类
HybridPolicyActionCfg.class_type = HybridPolicyAction
