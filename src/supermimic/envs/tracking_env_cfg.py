"""SuperMimic tracking environment on mjlab (plan Phase 2.3).

Differences from mjlab's BeyondMimic re-implementation (``mjlab.tasks.tracking``):
  * observations: paper Table A.1 proprioception only (no anchor pose, no base lin vel),
    10-frame history (term-major), plus a separate ``goal`` group with K future frames;
  * action: ``NoisyJointPositionAction`` (clean action observed, noise only executed);
  * multi-clip command with return-based reset sampling and clip-end time-outs;
  * rewards: self-collision term removed, joint-limit weight -2;
  * terminations: pelvis height / tilt vs reference, no end-effector term, clip-end time-out;
  * events: no push (opt-in pelvis push window / knee effort scale: mdp/events.py); friction and encoder bias re-sampled per episode; COM at startup;
  * a ``ReplayRecorder`` streams transitions into a GPU replay buffer;
  * configured for the 19-DoF Unitree H1 robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import events as sm_ev
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import rewards as sm_rew
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg
from supermimic.envs.mdp.rewards import (
    TrackingRewardCfg,
    arm_action_target_limits,
    motion_arm_joint_position_error_exp,
    motion_heading,
    motion_planar_drift,
)
from supermimic.robots.h1 import H1_ACTION_SCALE, get_h1_robot_cfg


class QuietSimulationCfg(SimulationCfg):
    """SimulationCfg wrapper for H1 simulation."""

    def apply_wp_opt(self, wp_opt) -> None:
        super().apply_wp_opt(wp_opt)
        wp_opt.warn_overflow = 0


TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link_ball_hand",
    "right_shoulder_roll_link",
    "right_elbow_link_ball_hand",
)
ANCHOR_BODY = "torso_link"
ROOT_BODY = "pelvis"
FOOT_GEOM_REGEX = r"^(left|right)_ankle_link_collision$"


@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]
    num_envs: int = 4096
    future_frames: int = 10
    history_length: int = 10
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"  # return | uniform | start
    replay_capacity: int = 1000
    record_transitions: bool = True
    seed: int | None = 0
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8
    reward_cfg: TrackingRewardCfg | None = None
    tilt_hinge_weight: float = 0.0  # DEVIATION when != 0: reward weight of max(0, tilt_err - knee) (mdp/rewards.py)
    tilt_hinge_knee: float = 0.21
    # DEVIATION when != 0: weight of -|w| max(0, |pelvis body-z ang vel| - thr)^2 (mdp/rewards.py yaw_rate_hinge)
    yaw_rate_hinge_weight: float = 0.0
    yaw_rate_hinge_thr: float = 7.5
    # DEVIATION when > 0: saturating-IMU observation model for base_ang_vel / projected_gravity (envs/imu_model.py)
    imu_sat_prob: float = 0.0
    imu_gyro_limit: float = 8.727
    imu_accel_k_max: float = 0.5
    # DEVIATION when set (default off): per-reset PD stiffness scale and extra torso mass [kg] (sim2sim G2 robustness axes)
    dr_kp_range: tuple[float, float] | None = None
    dr_torso_mass_add: tuple[float, float] | None = None
    dr_action_delay_substeps: int = 0  # DEVIATION when > 0: per-episode command latency 0..N physics substeps (envs/actions.py)
    # DEVIATION when set (default off): per-reset effort-limit (actuator force range) scale of the knee joints
    dr_knee_effort_scale: tuple[float, float] | None = None
    # DEVIATION when dr_push_prob > 0 (default off): per-episode pelvis force/torque inside a clip-frame window
    dr_push_prob: float = 0.0
    dr_push_frames: tuple[int, int] = (690, 730)
    dr_push_steps: tuple[int, int] = (2, 10)
    dr_push_force: tuple[float, float, float] = (60.0, 60.0, 30.0)
    dr_push_torque: tuple[float, float, float] = (8.0, 8.0, 8.0)
    dr_push_clips: tuple[str, ...] = ()
    joint_limit_weight: float = -2.0
    bin_frames: int = 50
    compact_library: bool = False
    anchor_obs: bool = False  # add BeyondMimic anchor-relative pose (9) to the goal group (plan D5 alt.)
    debug_vis: bool = False  # draw the reference ghost in the viewer (scripts/play.py)
    lock_waist: bool = False
    lock_waist_refs: str = "feasible"
    locked_action_weight: float = -0.1
    waist_coupling: bool = False
    waist_coupling_k: float = 1.4
    waist_coupling_c: float = 0.95
    waist_roll_range: float | None = None
    waist_roll_model_soft_limits: bool = False
    dr_waist_roll_damping: tuple[float, float] | None = None
    dr_waist_roll_friction: tuple[float, float] | None = None
    extra_command_kwargs: dict = field(default_factory=dict)


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg:
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    if a.lock_waist:
        from supermimic.envs.lock_waist import LOCKED_JOINT_IDS, LOCKED_JOINT_NAMES, NUM_JOINTS

        assert a.lock_waist_refs in ("feasible", "original"), a.lock_waist_refs
        jnoise = lambda h: noise(tuple(0.0 if i in LOCKED_JOINT_IDS else -h for i in range(NUM_JOINTS)), tuple(0.0 if i in LOCKED_JOINT_IDS else h for i in range(NUM_JOINTS)))  # noqa: E731
        joint_obs = {
            "joint_pos": ObservationTermCfg(func=sm_obs.joint_pos_rel_full, params={"biased": True}, noise=jnoise(0.01)),
            "joint_vel": ObservationTermCfg(func=sm_obs.joint_vel_rel_full, noise=jnoise(0.5)),
        }
        locked_ids = LOCKED_JOINT_IDS
        locked_names = LOCKED_JOINT_NAMES
    else:
        joint_obs = {
            "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),
            "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),
        }
        locked_ids = ()
        locked_names = ()
    if a.imu_sat_prob > 0.0:
        from supermimic.envs.imu_model import ImuBaseAngVel, ImuProjectedGravity

        imu_params = {"prob": a.imu_sat_prob, "gyro_limit": a.imu_gyro_limit, "accel_k_max": a.imu_accel_k_max}
        ang_vel_func, gravity_func = ImuBaseAngVel, ImuProjectedGravity
    else:
        imu_params = {}
        ang_vel_func, gravity_func = base_mdp.base_ang_vel, base_mdp.projected_gravity
    policy_terms = {
        **joint_obs,
        "base_ang_vel": ObservationTermCfg(func=ang_vel_func, params=dict(imu_params), noise=noise(-0.2, 0.2)),
        "projected_gravity": ObservationTermCfg(func=gravity_func, params=dict(imu_params), noise=noise(-0.05, 0.05)),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
            history_length=a.history_length,
            flatten_history_dim=True,
        ),
        "goal": ObservationGroupCfg(
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}),
                **(
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),
                    }
                    if a.anchor_obs
                    else {}
                ),
            },
            concatenate_terms=True,
            enable_corruption=a.obs_noise,
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale={k: v for k, v in H1_ACTION_SCALE.items() if not (a.lock_waist and k in locked_names)},
            use_default_offset=True,
            noise_std=a.action_noise_std,
            locked_joint_ids=locked_ids,
            delay_substeps_max=a.dr_action_delay_substeps if a.domain_randomization else 0,
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
    commands = {
        "motion": MultiClipMotionCommandCfg(
            npz_files=[str(f) for f in a.npz_files],
            anchor_body_name=ANCHOR_BODY,
            root_body_name=ROOT_BODY,
            body_names=TRACKED_BODIES,
            entity_name="robot",
            future_frames=a.future_frames,
            bin_frames=a.bin_frames,
            pose_range=pose_range if a.reset_noise else {},
            velocity_range=vel_range if a.reset_noise else {},
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0),
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library,
            locked_joint_ids=locked_ids,
            feasible_refs=a.lock_waist and a.lock_waist_refs == "feasible",
            debug_vis=a.debug_vis,
            **a.extra_command_kwargs,
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization:
        events["base_com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
        if a.dr_kp_range is not None:
            events["pd_gains"] = EventTermCfg(
                mode="reset",
                func=dr.pd_gains,
                params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)), "kp_range": tuple(a.dr_kp_range), "kd_range": (1.0, 1.0)},
            )
        if a.dr_torso_mass_add is not None:
            events["torso_mass"] = EventTermCfg(
                mode="reset",
                func=dr.body_mass,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": tuple(a.dr_torso_mass_add)},
            )
        if a.dr_waist_roll_damping is not None:
            events["waist_roll_damping"] = EventTermCfg(
                mode="reset",
                func=dr.joint_damping,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("torso_joint",)), "operation": "abs", "ranges": tuple(a.dr_waist_roll_damping)},
            )
        if a.dr_waist_roll_friction is not None:
            events["waist_roll_friction"] = EventTermCfg(
                mode="reset",
                func=dr.joint_friction,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("torso_joint",)), "operation": "abs", "ranges": tuple(a.dr_waist_roll_friction)},
            )
        if a.dr_knee_effort_scale is not None:
            events["knee_effort"] = EventTermCfg(
                mode="reset",
                func=sm_ev.joint_effort_limit_scale,
                params={"joint_names": ("left_knee_joint", "right_knee_joint"), "scale_range": tuple(a.dr_knee_effort_scale)},
            )
        if a.dr_push_prob > 0.0:
            events["pelvis_push"] = EventTermCfg(
                mode="step",
                func=sm_ev.PelvisPushWindow,
                params={
                    "prob": a.dr_push_prob, "frames": tuple(a.dr_push_frames), "steps": tuple(a.dr_push_steps),
                    "force_max": tuple(a.dr_push_force), "torque_max": tuple(a.dr_push_torque),
                    "clip_names": tuple(a.dr_push_clips), "body_name": ROOT_BODY, "command_name": "motion",
                },
            )
    else:
        # Evaluation protocol: foot friction fixed at 1.0, no other randomization.
        events["foot_friction"] = EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs",
                "ranges": (1.0, 1.0),
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    }
    if a.reward_cfg is not None:
        r = a.reward_cfg
        if r.dt != 0.02:
            raise ValueError("H1 reward dt must match the 50 Hz control period")
        if r.planar_drift_scale <= 0 or r.w_planar_drift > 0 or r.w_heading < 0:
            raise ValueError("Global drift needs a positive scale, nonpositive cost weight and nonnegative heading weight")
        for name, weight, std in (
            ("motion_global_root_pos", r.w_root_pos, r.std_root_pos),
            ("motion_global_root_ori", r.w_root_ori, r.std_root_ori),
            ("motion_body_pos", r.w_body_pos, r.std_body_pos),
            ("motion_body_ori", r.w_body_ori, r.std_body_ori),
            ("motion_body_lin_vel", r.w_body_lin_vel, r.std_body_lin_vel),
            ("motion_body_ang_vel", r.w_body_ang_vel, r.std_body_ang_vel),
            ("action_rate_l2", r.w_action_rate, None),
            ("joint_limit", r.w_joint_limit, None),
        ):
            rewards[name].weight = weight
            if std is not None:
                rewards[name].params["std"] = std
        if r.w_arm_joint_pos:
            rewards["motion_arm_joint_pos"] = RewardTermCfg(
                func=motion_arm_joint_position_error_exp,
                weight=r.w_arm_joint_pos,
                params={"command_name": "motion", "std": r.std_arm_joint_pos},
            )
        if r.w_arm_target_limit:
            rewards["arm_target_limit"] = RewardTermCfg(
                func=arm_action_target_limits,
                weight=r.w_arm_target_limit,
                params={"command_name": "motion"},
            )
        if r.w_planar_drift:
            rewards["motion_global_planar_drift"] = RewardTermCfg(
                func=motion_planar_drift,
                weight=r.w_planar_drift,
                params={"command_name": "motion", "scale": r.planar_drift_scale},
            )
        if r.w_heading:
            rewards["motion_global_heading"] = RewardTermCfg(
                func=motion_heading,
                weight=r.w_heading,
                params={"command_name": "motion"},
            )
    if a.tilt_hinge_weight != 0.0:
        rewards["tilt_hinge"] = RewardTermCfg(func=sm_rew.tilt_hinge, weight=a.tilt_hinge_weight, params={"command_name": "motion", "knee": a.tilt_hinge_knee})
    if a.yaw_rate_hinge_weight != 0.0:
        rewards["yaw_rate_hinge"] = RewardTermCfg(func=sm_rew.yaw_rate_hinge, weight=-abs(a.yaw_rate_hinge_weight), params={"thr": a.yaw_rate_hinge_thr})
    if a.lock_waist and a.locked_action_weight != 0.0:
        rewards["locked_action_l2"] = RewardTermCfg(func=sm_rew.locked_action_l2, weight=a.locked_action_weight, params={"action_ids": locked_ids})

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),
    }

    recorders = {}
    if a.record_transitions:
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": "torso_link"})

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": robot_cfg(a.lock_waist, (a.waist_coupling_k, a.waist_coupling_c) if a.waist_coupling else None, a.waist_roll_range)}),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        recorders=recorders,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=50, ls_iterations=50)),
        decimation=4,
        episode_length_s=a.episode_length_s,
        seed=a.seed,
    )
    return cfg


def robot_cfg(lock_waist: bool = False, waist_coupling: tuple[float, float] | None = None, waist_roll_range: float | None = None):
    """H1 entity cfg."""
    if lock_waist:
        from supermimic.envs.lock_waist import get_g1_lock_waist_robot_cfg

        return get_g1_lock_waist_robot_cfg()
    return get_h1_robot_cfg()


def keep_nominal_waist_roll_soft_limits(env: ManagerBasedRlEnv) -> None:
    """No-op on H1 (which has no waist_roll_joint)."""
    return


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)
