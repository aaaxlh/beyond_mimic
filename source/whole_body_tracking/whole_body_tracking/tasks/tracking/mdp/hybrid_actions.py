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

        if hasattr(env, "num_envs"):
            self._num_envs = int(env.num_envs)
        else:
            raise AttributeError("The environment does not have a 'num_envs' attribute.")

        # 解析关节索引
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)

        # 统一 joint_ids 为 python list[int]（便于 index/in 判断）
        if isinstance(self._joint_ids, torch.Tensor):
            self._joint_ids = [int(x) for x in self._joint_ids.tolist()]
        else:
            self._joint_ids = [int(x) for x in self._joint_ids]

        self._num_joints = len(self._joint_ids)

        # ---- ActionTerm 接口要求的 buffers（必须初始化）----
        self._raw_actions = torch.zeros((self._num_envs, self._num_joints), device=self.device)
        self._processed_actions = torch.zeros((self._num_envs, self._num_joints), device=self.device)
        self._target_joint_vel = torch.zeros((self._num_envs, self._num_joints), device=self.device)

        # 预训练动作缓存（调试用）
        self._pretrained_actions = torch.zeros((self._num_envs, self._num_joints), device=self.device)

        # 处理缩放 (Scale)
        if isinstance(cfg.scale, (float, int)):
            self._action_scale = torch.full((self._num_envs, self._num_joints), float(cfg.scale), device=self.device)
        elif isinstance(cfg.scale, dict):
            self._action_scale = torch.ones((self._num_envs, self._num_joints), device=self.device)
            for joint_name, scale in cfg.scale.items():
                ids, _ = self._asset.find_joints(joint_name)
                if isinstance(ids, torch.Tensor):
                    ids = [int(x) for x in ids.tolist()]
                else:
                    ids = [int(x) for x in ids]
                for jid in ids:
                    if jid in self._joint_ids:
                        idx = self._joint_ids.index(jid)
                        self._action_scale[:, idx] = float(scale)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")

        # 获取默认关节位置偏移
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros((self._num_envs, self._num_joints), device=self.device)

        # 加载预训练模型
        self._pretrained_model = None
        if cfg.pretrained_model_path:
            self._load_pretrained_model(cfg.pretrained_model_path)

        # 给 apply_actions() 走 _asset.apply_action(...) 预留容器（可选但建议）
        self._actions = ArticulationActions(
            joint_positions=self._processed_actions,
            joint_velocities=self._target_joint_vel,
            joint_efforts=None,
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
        """Load model from WandB.

        Supported formats:
          1) Artifact:
             "wandb://<entity>/<project>/<artifact>:<version>"
             e.g. "wandb://myteam/myproj/model-abc123:v0"

          2) Run file:
             "wandb://<entity>/<project>/run-<run_id>/files/<filename>"
             e.g. "wandb://myteam/myproj/run-zcru65cr/files/model_15000.pt"
        """
        import os
        import wandb

        # wandb://entity/project/...
        path = artifact_path.replace("wandb://", "", 1)
        parts = [p for p in path.split("/") if p]
        if len(parts) < 3:
            raise ValueError(f"Invalid WandB path: {artifact_path}")

        entity, project = parts[0], parts[1]
        rest = parts[2:]

        # ---- Case 2: run file path ----
        # run-<run_id>/files/<filename>
        if len(rest) >= 3 and rest[0].startswith("run-") and rest[1] == "files":
            run_id = rest[0].replace("run-", "", 1)
            filename = "/".join(rest[2:])  # allow nested paths, if any

            api = wandb.Api()
            run = api.run(f"{entity}/{project}/{run_id}")
            wf = run.file(filename)

            cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "beyond_mimic",
                "wandb_runs",
                f"{entity}__{project}__{run_id}",
            )
            os.makedirs(cache_dir, exist_ok=True)

            local_path = wf.download(root=cache_dir, replace=True).name
            self._load_from_file(local_path)
            print(f"[HybridPolicyAction] Loaded pretrained model from W&B run file: {artifact_path}")
            print(f"[HybridPolicyAction] Downloaded to: {local_path}")
            return

        # ---- Case 1: artifact path (original behavior) ----
        # remaining part is "<artifact>:<version>" or "<name>:<version>"
        artifact_name = "/".join(rest)

        # If not in an existing run, start a minimal one so use_artifact works reliably.
        if wandb.run is None:
            wandb.init(entity=entity, project=project, job_type="inference", reinit=True)

        # Note: type must match what you logged. Keep "model" as default.
        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()

        # Try common filenames; fallback to first .pt
        model_file = os.path.join(artifact_dir, "model.pt")
        if not os.path.exists(model_file):
            pt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
            if pt_files:
                model_file = os.path.join(artifact_dir, pt_files[0])
            else:
                raise FileNotFoundError(f"No .pt file found in artifact dir: {artifact_dir}")

        self._load_from_file(model_file)
        print(f"[HybridPolicyAction] Loaded pretrained model from W&B artifact: {artifact_path}")

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
            def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
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
    
    @property
    def _scale(self) -> torch.Tensor:
        return self._action_scale

    def apply_actions(self) -> None:
        """Write the processed actions into the simulated asset."""
        if hasattr(self._asset, "apply_action"):
            self._asset.apply_action(self._actions)
            return

        if hasattr(self._asset, "set_joint_position_target"):
            self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        if hasattr(self._asset, "set_joint_velocity_target"):
            self._asset.set_joint_velocity_target(self._target_joint_vel, joint_ids=self._joint_ids)
        if hasattr(self._asset, "set_joint_effort_target"):
            self._asset.set_joint_effort_target(torch.zeros_like(self._processed_actions), joint_ids=self._joint_ids)

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

        # 5. 保存用于观测（写入私有 buffer）
        self._raw_actions[:] = actions
        self._processed_actions[:] = target_positions

        # 同步到 _actions（如果你在 apply_actions 里走 apply_action）
        self._actions.joint_positions = self._processed_actions
        self._actions.joint_velocities = self._target_joint_vel

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
