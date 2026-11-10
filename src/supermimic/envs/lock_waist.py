"""Waist-locked H1 configuration for training (``--env.lock-waist``).

On H1 (19-DoF), waist roll and pitch are already absent from the kinematic tree; only ``torso_joint`` (yaw, index 10)
exists at the waist. When ``lock_waist=False`` (the default), all 19 joints are active. When ``lock_waist=True``,
``torso_joint`` is deleted from the MjSpec while keeping the 19-d policy/state interface.
"""

from __future__ import annotations

import mujoco
import torch

from supermimic.robots.h1 import H1_JOINT_NAMES, get_h1_robot_cfg, get_h1_spec, load_h1_model

LOCKED_JOINT_NAMES: tuple[str, ...] = ("torso_joint",)
LOCKED_JOINT_IDS: tuple[int, ...] = tuple(H1_JOINT_NAMES.index(n) for n in LOCKED_JOINT_NAMES)  # (10,)
NUM_JOINTS = len(H1_JOINT_NAMES)  # 19
ACTIVE_JOINT_IDS: tuple[int, ...] = tuple(i for i in range(NUM_JOINTS) if i not in LOCKED_JOINT_IDS)


def _short(name: str | None) -> str:
    return (name or "").split("/")[-1]


def lock_waist_spec(spec: mujoco.MjSpec | None = None) -> mujoco.MjSpec:
    """H1 MjSpec with locked waist joint(s) deleted (child bodies welded at q = 0)."""
    if spec is None:
        spec = get_h1_spec()
    for a in list(spec.actuators):
        if _short(a.target) in LOCKED_JOINT_NAMES:
            spec.delete(a)
    for s in list(spec.sensors):
        if _short(s.objname) in LOCKED_JOINT_NAMES:
            spec.delete(s)
    if any(_short(j.name) in LOCKED_JOINT_NAMES for j in spec.joints):
        for k in list(spec.keys):  # keyframes have the unlocked qpos size
            spec.delete(k)
    for j in list(spec.joints):
        if _short(j.name) in LOCKED_JOINT_NAMES:
            spec.delete(j)
    return spec


def get_g1_lock_waist_robot_cfg():
    """``get_h1_robot_cfg()`` with the locked waist model."""
    from dataclasses import replace

    cfg = get_h1_robot_cfg()
    actuators = tuple(
        a for a in cfg.articulation.actuators if not set(a.target_names_expr).issubset(set(LOCKED_JOINT_NAMES))
    )
    init_jpos = {k: v for k, v in cfg.init_state.joint_pos.items() if k not in LOCKED_JOINT_NAMES}
    return replace(
        cfg,
        spec_fn=lock_waist_spec,
        init_state=replace(cfg.init_state, joint_pos=init_jpos),
        articulation=replace(cfg.articulation, actuators=actuators),
    )


def joint_slots(entity_joint_names) -> list[int]:
    """Position in the 19-joint order of each entity joint (entity joint names may carry a prefix)."""
    names = [_short(n) for n in entity_joint_names]
    slots = [H1_JOINT_NAMES.index(n) for n in names]
    assert slots == sorted(slots), "entity joint order must follow the MJCF order"
    return slots


def is_locked_entity(robot) -> bool:
    return len(robot.joint_names) != NUM_JOINTS


def expand_joints(x: torch.Tensor, slots: list[int] | torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """(..., n_entity) -> (..., 19), locked slots set to ``fill``."""
    if x.shape[-1] == NUM_JOINTS:
        return x
    out = torch.full((*x.shape[:-1], NUM_JOINTS), fill, dtype=x.dtype, device=x.device)
    out[..., slots] = x
    return out


def full_joint_constants(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """19-joint default joint positions (19,) and soft limits (19, 2) of a (possibly locked) entity."""
    q0 = robot.data.default_joint_pos[0].detach().clone()
    soft = robot.data.soft_joint_pos_limits[0].detach().clone()
    if not is_locked_entity(robot):
        return q0, soft
    m = load_h1_model()
    rng = torch.as_tensor(m.jnt_range[1:], dtype=soft.dtype, device=soft.device)  # skip the free joint; 19 x 2
    factor = float(robot.cfg.articulation.soft_joint_pos_limit_factor)
    mid, half = rng.mean(-1), 0.5 * (rng[:, 1] - rng[:, 0])
    soft_full = torch.stack([mid - factor * half, mid + factor * half], dim=-1)
    slots = joint_slots(robot.joint_names)
    soft_full[slots] = soft
    return expand_joints(q0, slots, 0.0), soft_full
