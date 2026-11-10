# Unitree G1 (29-DoF) -> Unitree H1 (19-DoF) 完整迁移报告

本报告总结了将仓库从 **Unitree G1 (29-DoF)** 全面迁移至 **Unitree H1 (19-DoF)** 所完成的全部代码新增与修改。

---

## 一、 核心维度与机器人规格变更总览

| 维度 / 常量 | 原 G1 (29-DoF) | 现 H1 (19-DoF) | 说明 |
| :--- | :---: | :---: | :--- |
| `NJ` / `ACT_DIM` | `29` | `19` | 左腿 5 + 右腿 5 + 腰部 1 (`torso_joint`) + 左臂 4 + 右臂 4 |
| `NQ` | `36` (`7 + 29`) | `26` (`7 + 19`) | 浮动基座位置 (3) + 四元数 wxyz (4) + 关节角 (19) |
| `NV` | `35` (`6 + 29`) | `25` (`6 + 19`) | 浮动基座线速度 (3) + 角速度 (3) + 关节角速度 (19) |
| `STATE_DIM` / `FEAT_DIM` | `71` | `51` | 世界模型状态向量 `[p(3), q(4), qj(19), v(3), w(3), qd(19)]` 及特征维度 |
| `DELTA_DIM` | `70` | `50` | 世界模型动力学残差输出维度 `[dv_h(3), dw_b(3), dqd(19), dp_h(3), dth_b(3), dq(19)]` |
| `XI_DIM` | `33` (`1 + 29 + 3`) | `23` (`1 + 19 + 3`) | 特权物理参数 `[friction(1), joint_bias(19), com_offset(3)]` |
| `OBS_DIM` | `93` (`3+3+29+29+29`) | `63` (`3+3+19+19+19`) | 单步本体观测 `[base_ang_vel(3), projected_gravity(3), joint_pos(19), joint_vel(19), actions(19)]` |
| `OBS_DIM * 10` (历史窗口) | `930` | `630` | 10 步历史本体观测维度 |
| `GOAL_FEAT_DIM` | `68` (`29+29+3+3+4`) | `48` (`19+19+3+3+4`) | 单帧参考目标维度 `[ref_q(19), ref_dq(19), rel_pos(3), rel_yaw(1)+pad, ...]` |
| `GOAL_FEAT_DIM * 10` | `680` (含 anchor 为 `689`) | `480` (含 anchor 为 `489`) | 10 帧未来目标窗口维度 |
| `TRACKED_BODIES` 数量 | `14` | `14` | 适配 H1 骨骼名（如 `left_ankle_link`、`left_elbow_link_ball_hand` 等） |
| 全部刚体数 (`NUM_BODIES`) | `30` | `20` | H1 除 `world` 外的 20 个刚体 link |

### H1 19 关节顺序 (`H1_JOINT_NAMES`) 与默认控制参数

| 索引 | 关节名称 | 部位 | 默认角度 `default_pos` (rad) | `Kp` (N·m/rad) | `Kd` (N·m·s/rad) | 力矩上限 `effort_limit` (N·m) | 速度上限 `velocity_limit` (rad/s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `left_hip_yaw_joint` | 左腿 | `0.0` | `150.0` | `2.0` | `200.0` | `23.0` |
| 1 | `left_hip_roll_joint` | 左腿 | `0.0` | `150.0` | `2.0` | `200.0` | `23.0` |
| 2 | `left_hip_pitch_joint` | 左腿 | `-0.4` | `150.0` | `2.0` | `200.0` | `23.0` |
| 3 | `left_knee_joint` | 左腿 | `0.8` | `200.0` | `4.0` | `300.0` | `14.0` |
| 4 | `left_ankle_joint` | 左腿 | `-0.4` | `40.0` | `2.0` | `40.0` | `9.0` |
| 5 | `right_hip_yaw_joint` | 右腿 | `0.0` | `150.0` | `2.0` | `200.0` | `23.0` |
| 6 | `right_hip_roll_joint` | 右腿 | `0.0` | `150.0` | `2.0` | `200.0` | `23.0` |
| 7 | `right_hip_pitch_joint` | 右腿 | `-0.4` | `150.0` | `2.0` | `200.0` | `23.0` |
| 8 | `right_knee_joint` | 右腿 | `0.8` | `200.0` | `4.0` | `300.0` | `14.0` |
| 9 | `right_ankle_joint` | 右腿 | `-0.4` | `40.0` | `2.0` | `40.0` | `9.0` |
| 10 | `torso_joint` | 腰部 | `0.0` | `300.0` | `3.0` | `200.0` | `23.0` |
| 11 | `left_shoulder_pitch_joint` | 左臂 | `0.0` | `100.0` | `2.0` | `40.0` | `9.0` |
| 12 | `left_shoulder_roll_joint` | 左臂 | `0.0` | `100.0` | `2.0` | `40.0` | `9.0` |
| 13 | `left_shoulder_yaw_joint` | 左臂 | `0.0` | `100.0` | `2.0` | `18.0` | `20.0` |
| 14 | `left_elbow_joint` | 左臂 | `0.0` | `50.0` | `2.0` | `18.0` | `20.0` |
| 15 | `right_shoulder_pitch_joint` | 右臂 | `0.0` | `100.0` | `2.0` | `40.0` | `9.0` |
| 16 | `right_shoulder_roll_joint` | 右臂 | `0.0` | `100.0` | `2.0` | `40.0` | `9.0` |
| 17 | `right_shoulder_yaw_joint` | 右臂 | `0.0` | `100.0` | `2.0` | `18.0` | `20.0` |
| 18 | `right_elbow_joint` | 右臂 | `0.0` | `50.0` | `2.0` | `18.0` | `20.0` |

---

## 二、 详细修改文件清单

### 1. H1 模型资产 (`h1_data/`)
- **[h1_data/h1_training.xml](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/h1_data/h1_training.xml)** & **[h1_data/h1.xml](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/h1_data/h1.xml)**
  - 提供 H1 19 自由度 MuJoCo 模型与 `meshes/h1/*.STL` 网格资产。
  - 为左右踝碰撞体补全显式命名 `left_ankle_link_collision` 与 `right_ankle_link_collision`，与训练环境摩擦随机化正则 `FOOT_GEOM_REGEX` 及足底高度计算对齐。

---

### 2. 核心库 (`src/supermimic/`)

#### (a) 机器人定义与全局维度常量
- **[src/supermimic/robot_dimensions.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/robot_dimensions.py)** *(新增)*
  - 集中定义 H1 19-DoF 的全部维度常量：`NJ = 19`、`NQ = 26`、`NV = 25`、`STATE_DIM = 51`、`FEAT_DIM = 51`、`DELTA_DIM = 50`、`ACT_DIM = 19`、`XI_DIM = 23`、`OBS_DIM = 63`、`GOAL_FEAT_DIM = 48`、`REF_DIM = 51`、`STATE_SIZES`、`DELTA_SIZES`、`OBS_TERMS`。
- **[src/supermimic/robots/__init__.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/robots/__init__.py)** & **[src/supermimic/robots/h1.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/robots/h1.py)** *(新增)*
  - 定义 `H1_XML`（指向 `h1_data/h1_training.xml`）、`H1_JOINT_NAMES`（19 关节）、`H1_ACTION_SCALE`（统一 `0.25`）、`ARM_JOINT_INDICES`（`11..18`）、`default_pos`、`stiffness`、`damping`、`effort_limit`、`velocity_limit`。
  - 实现 CPU 模型加载与校验 `load_h1_model()`、可编辑 `MjSpec` 构造 `get_h1_spec()`、基于足部网格顶点的足底最低高度计算 `foot_sole_height()`，以及 `mjlab` 位置控制实体配置 `get_h1_robot_cfg()`。

#### (b) 数据转换与录制 (`src/supermimic/data/`)
- **[src/supermimic/data/motion_format.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/data/motion_format.py)** *(新增)*
  - 定义 H1 动作数据 schema 标识（`SCHEMA_VERSION = 2`、`ROBOT_NAME = "unitree_h1"`、`EXPECTED_BODY_COUNT = 20`）及校验函数 `validate_motion()`、`validate_library()`。
- **[src/supermimic/data/motion_convert.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/data/motion_convert.py)** *(修改)*
  - 切换为加载 `H1_XML` 与 `H1_JOINT_NAMES`（保留 `G1_JOINT_NAMES`、`load_g1_model`、`g1_xml_path` 兼容别名）。
  - CSV 解析要求 `7 + 19 = 26` 列，前向运动学输出 20 个刚体状态并写入 H1 元数据。
- **[src/supermimic/data/recorder.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/data/recorder.py)** *(修改)*
  - 使用 `STATE_DIM = 51`、`ACT_DIM = 19`、`XI_DIM = 23`、`OBS_DIM = 63` 替换原硬编码的 `71 / 29 / 33 / 93`。

#### (c) 世界模型 (`src/supermimic/world_model/`)
- **[src/supermimic/world_model/state_spec.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/world_model/state_spec.py)** *(修改)*
  - 使用 `NJ = 19`、`STATE_SIZES = (3, 4, 19, 3, 3, 19)`、`DELTA_SIZES = (3, 3, 19, 3, 3, 19)` 参数化状态拆分、积分、误差计算、特权参数归一化与锁腰掩码。
- **[src/supermimic/world_model/dynamics.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/world_model/dynamics.py)** *(修改)*
  - `DynamicsCfg` 默认输入/输出维度更新为 `feat_dim = 51`、`action_dim = 19`、`xi_dim = 23`、`delta_dim = 50`。
- **[src/supermimic/world_model/heads.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/world_model/heads.py)** *(修改)*
  - `policy_obs_step`、`AnalyticHeads` 等全部切换为 `H1_JOINT_NAMES`、`H1_ACTION_SCALE`、`NQ = 26`、`NJ = 19`、`OBS_TERMS`。
- **[src/supermimic/world_model/wm_trainer.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/world_model/wm_trainer.py)** *(修改)*
  - 损失切片与分量统计改为按 `NJ = 19` 动态切片（`dq: 12..31`，`dqd: 31..50`）。

#### (d) 仿真环境与 MDP (`src/supermimic/envs/`)
- **[src/supermimic/envs/tracking_env_cfg.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/tracking_env_cfg.py)** *(修改)*
  - `robot_cfg()` 切换为 `get_h1_robot_cfg()`。
  - `TRACKED_BODIES` 更新为 H1 的 14 个关键刚体：`pelvis`、`left_hip_pitch_link`、`left_knee_link`、`left_ankle_link`、`right_hip_pitch_link`、`right_knee_link`、`right_ankle_link`、`torso_link`、`left_shoulder_roll_link`、`left_elbow_link_ball_hand`、`right_shoulder_roll_link`、`right_elbow_link_ball_hand` 等。
  - `FOOT_GEOM_REGEX` 更新为 `r"^(left|right)_ankle_link_collision$"`。
- **[src/supermimic/envs/state_io.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/state_io.py)** & **[src/supermimic/envs/sim_context.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/sim_context.py)** *(修改)*
  - 状态读写 `read_sim_state` / `write_sim_state` 及特权参数 `read_xi` 更新为 `51` 维状态与 `23` 维 `xi`。
- **[src/supermimic/envs/lock_waist.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/lock_waist.py)** *(修改)*
  - 适配 H1 单腰关节：`LOCKED_JOINT_NAMES = ("torso_joint",)`、`LOCKED_JOINT_IDS = (10,)`、`NUM_JOINTS = 19`。
- **[src/supermimic/envs/imu_model.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/imu_model.py)** *(修改)*
  - 角速度切片更新为 `s[..., 3 + NQ : 6 + NQ]`（即 `29:32`）。
- **[src/supermimic/envs/mdp/motion_library.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/mdp/motion_library.py)** & **[src/supermimic/envs/mdp/commands.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/mdp/commands.py)** & **[src/supermimic/envs/mdp/rewards.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/envs/mdp/rewards.py)** *(修改)*
  - 动作库紧凑格式改为 `51` 维 `ref`（含 `19` 维 `q` 与 `19` 维 `dq`），未来帧目标特征 `48` 维；足底高度计算调用 H1 的 `foot_sole_height`；关节力矩惩罚项切换至 H1 的 `stiffness`、`damping`、`effort_limit`、`default_pos`。

#### (e) 策略网络、想象展开、训练循环与部署导出 (`src/supermimic/policy/`, `imagination/`, `training/`, `deploy/`)
- **[src/supermimic/config.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/config.py)** *(修改)*
  - `joint_torque_jw` 默认腰部关节名更新为 `("torso_joint", 2.0)`。
- **[src/supermimic/policy/nets.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/policy/nets.py)** *(修改)*
  - `PolicyCfg` 默认维度更新为 `prop_dim = 630` (`10 * 63`)、`goal_dim = 480` (`10 * 48`)、`action_dim = 19`；`CriticCfg` 更新为 `state_dim = 51`、`goal_dim = 480`、`xi_dim = 23`。
- **[src/supermimic/imagination/compiled.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/imagination/compiled.py)** & **[src/supermimic/imagination/rollout.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/imagination/rollout.py)** *(修改)*
  - 编译展开图与观测缓冲切片更新为 `NQ = 26`、`NJ = 19`、`OBS_TERMS`。
- **[src/supermimic/training/anchor_migration.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/training/anchor_migration.py)** *(新增)* & **[src/supermimic/training/loop.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/training/loop.py)** *(修改)*
  - 支持 H1 维度下的 `anchor_obs` 权重无缝扩展与训练循环构建。
- **[src/supermimic/deploy/core.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/deploy/core.py)** & **[src/supermimic/deploy/export.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/src/supermimic/deploy/export.py)** *(修改)*
  - `DeployCore` 与 ONNX 导出模块切换为 19 自由度 H1 参数（`NUM_JOINTS = 19`、`STEP_OBS_DIM = 63`、`GOAL_FRAME_DIM = 48`、`PROP_DIM = 630`、`GOAL_DIM = 480`），并在导出的 ONNX 元数据中写入 H1 的 `joint_names`、`default_joint_pos`、`action_scale`、`kp`、`kd`、`effort_limit`。

---

### 3. 部署、数据与评测脚本 (`scripts/` & `tools/`)

- **[scripts/deploy/g1_common.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/g1_common.py)** *(修改)*
  - 全面更新为 19 关节 H1 常量：`NUM_JOINTS = 19`、`JOINT_NAMES`、`DEFAULT_JOINT_POS`、`ACTION_SCALE`、`KP`、`KD`、`EFFORT_LIMIT`、`VELOCITY_LIMIT`。
- **[scripts/deploy/check_robot.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/check_robot.py)**, **[scripts/deploy/g1_runner.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/g1_runner.py)**, **[scripts/deploy/bridge/g1_runner_bridge.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/bridge/g1_runner_bridge.py)** *(修改)*
  - 注释与日志表格适配 19 关节 H1（腿部 `0..9`、腰部 `10`、手臂 `11..18`）。
- **[scripts/deploy/sim_g1_mujoco.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/sim_g1_mujoco.py)** & **[scripts/deploy/sim2sim.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/sim2sim.py)** *(修改)*
  - 默认训练/仿真 XML 路径指向 `h1_data/h1_training.xml`，兼容无内置 `<actuator>` 的 H1 XML（支持直接通过 `qfrc_applied` 施加 PD 力矩），并将锁腰索引更新为 `(10,)`。
- **[scripts/deploy/make_combo_clip.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/make_combo_clip.py)**, **[scripts/deploy/knee_feasibility.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/knee_feasibility.py)**, **[scripts/deploy/spin_feasibility.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/spin_feasibility.py)**, **[scripts/deploy/realstate_gate.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/deploy/realstate_gate.py)** *(修改)*
  - 膝关节索引更新为 `(3, 8)`（H1 左/右膝），腰部索引更新为 `10`，动作与状态切片更新为 19 自由度。
- **[scripts/data/build_real_reset_states.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/data/build_real_reset_states.py)** & **[scripts/data/filter_bones_seed.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/data/filter_bones_seed.py)** *(修改)*
  - `foot_capsules` 与 `lowest_point` 同时支持胶囊体 (`mjGEOM_CAPSULE`) 与 H1 网格碰撞体 (`mjGEOM_MESH`)；运动过滤脚本支持 H1 的 `left_ankle_link` / `right_ankle_link` 刚体名与 19 关节维度。
- **[scripts/bench_imagination.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/bench_imagination.py)**, **[scripts/bench_sim_speed.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/bench_sim_speed.py)**, **[scripts/eval_push.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/eval_push.py)**, **[scripts/eval_benchmark.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/scripts/eval_benchmark.py)** *(修改)*
  - 基准测试与抗推力评测全部更新为 `51` 维状态、`48` 维目标特征、`23` 维 `xi` 与 `19` 维动作。
- **[tools/video2motion/g1_model.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tools/video2motion/g1_model.py)**, **[tools/video2motion/extend_full.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tools/video2motion/extend_full.py)**, **[tools/video2motion/gvhmr_to_supermimic.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tools/video2motion/gvhmr_to_supermimic.py)** *(修改)*
  - 视频转动作工具链适配 H1 19 关节模型、`26` 列 CSV、20 刚体 FK 以及 H1 足部接触高度修正。

---

### 4. 单元测试套件 (`tests/`)
已将全部测试文件迁移至 H1 (19-DoF) 规格并通过测试验证（`72 passed, 40 skipped, 0 failed`）：
- [tests/test_analytic_heads.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_analytic_heads.py)
- [tests/test_deploy_parity.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_deploy_parity.py)
- [tests/test_env_smoke.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_env_smoke.py)
- [tests/test_g1_fk.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_g1_fk.py)
- [tests/test_imu_model.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_imu_model.py)
- [tests/test_joint_torque.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_joint_torque.py)
- [tests/test_joint_vel_hinge.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_joint_vel_hinge.py)
- [tests/test_lock_waist.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_lock_waist.py)
- [tests/test_motion_convert.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_motion_convert.py)
- [tests/test_motion_library_compact.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_motion_library_compact.py)
- [tests/test_real_reset.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_real_reset.py)
- [tests/test_realstate_gate.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_realstate_gate.py)
- [tests/test_tilt_hinge.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_tilt_hinge.py)
- [tests/test_waist_coupling.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_waist_coupling.py)
- [tests/test_waist_roll.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_waist_roll.py)
- [tests/test_world_model.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_world_model.py)
- [tests/test_yaw_rate_hinge.py](file:///home/rl-4090/RL_Workspace/ljp_ws/SuperMimic-Repro-Sim2Real-main/tests/test_yaw_rate_hinge.py)

---

## 三、 常用运行指令速查

### 1. 运行单元测试
```bash
PYTHONPATH=src pytest -q
```

### 2. 将 H1 CSV 重定向数据转换为 NPZ 并打包运动库
```bash
# 批量将 26 列 H1 CSV 转换为 mjlab 兼容的 50Hz NPZ
PYTHONPATH=src python scripts/data/convert_lafan1.py \
    --src data/raw/lafan1/h1 \
    --dst data/npz/lafan1_h1 \
    --input-fps 30

# 打包为紧凑运动库 .pt
PYTHONPATH=src python scripts/data/build_library.py \
    --npz-dir data/npz/lafan1_h1 \
    --out data/library/lafan1_h1.pt
```

### 3. 启动 H1 SuperMimic 训练
```bash
PYTHONPATH=src python scripts/train.py \
    --data.library data/library/lafan1_h1.pt \
    --run-name h1_lafan1
```

### 4. 导出 H1 ONNX 部署策略与参考轨迹包
```bash
PYTHONPATH=src python scripts/deploy/export_onnx.py \
    --ckpt logs/h1_lafan1/latest.pt \
    --npz data/npz/lafan1_h1/walk1_subject1.npz \
    --out-dir exported/h1_walk1
```

### 5. MuJoCo 独立仿真验证 (Sim2Sim)
```bash
PYTHONPATH=src python scripts/deploy/sim_g1_mujoco.py \
    --policy exported/h1_walk1/policy.onnx \
    --motion exported/h1_walk1/walk1_subject1_ref.npz \
    --xml h1_data/h1_training.xml
```
