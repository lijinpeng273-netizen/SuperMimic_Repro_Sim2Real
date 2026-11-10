"""Export a trained H1 checkpoint for sim2sim / real-robot deployment (deploy_artifacts/).

Runs in the training venv (torch + mjlab).  Subcommands:

  freeze   copy the chosen checkpoints to deploy_artifacts/ckpts/ (copy, never move)
  config   deploy_artifacts/<ckpt>/deploy_config.json from the training robot model, cross-checked
           against a live 2-env training env (compiled MjModel, GPU model, action term scale/offset)
  clips    deploy_artifacts/clips/<name>.npz fp32 goal tables (+ reference root/joint state)
  onnx     deploy_artifacts/<ckpt>/policy.onnx (obs [1,630], goal [1,480|489] -> action [1,19]),
           verified against torch with onnxruntime
  mjcf     deploy_artifacts/models/h1_train_scene.xml (+ g1_train_scene.xml alias + assets/) of the training scene
  all      everything above
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from supermimic.robot_dimensions import ACT_DIM, GOAL_FEAT_DIM, NJ, OBS_DIM

ART = Path("deploy_artifacts")
NPZ_DIR = Path("data/npz/lafan1")
COMPACT_LIB = Path("data/library/bones_seed_compact.pt")
CKPTS = ("logs/e11/ckpt_0100000.pt", "logs/e11/ckpt_0116000.pt")
SAFE_CLIPS = (
    "neutral_idle_loop_001__A081",
    "idle_loop_002__A060",
    "shoulder_clap_004__A041_M",
    "itching_left_arm_R_003__A121",
    "walk_sideway_090_stop_003__A037",
)
HARDER_CLIPS = ("wave_R_001__A431", "jog_sideway_045_loop_002__A025", "dance_hiphop_alt_stump_R_fast_001__A318_M")
TEST_CLIPS = SAFE_CLIPS + HARDER_CLIPS
FOOT_GEOM_REGEX = r"^(robot/)?(left|right)_ankle_link_collision$"


def import_ort():
    try:
        import onnxruntime as ort
    except ImportError:
        overlay = os.environ.get("SM_ORT_OVERLAY")
        if not overlay:
            raise
        sys.path.append(overlay)
        import onnxruntime as ort
    return ort


def ckpt_tag(ckpt: str | Path) -> str:
    return Path(ckpt).stem


# ---------------------------------------------------------------------------------------------- freeze


def freeze(ckpts: tuple[str, ...] = CKPTS, out: Path = ART / "ckpts") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for c in ckpts:
        dst = out / Path(c).name
        if dst.exists() and dst.stat().st_size == Path(c).stat().st_size:
            print(f"  {dst} exists ({dst.stat().st_size} B), kept")
            continue
        shutil.copy2(c, dst)
        print(f"  copied {c} -> {dst} ({dst.stat().st_size} B)")


# ---------------------------------------------------------------------------------------------- config


def constants_from_mjlab(joint_names: list[str]) -> dict[str, np.ndarray]:
    """Per-joint H1 robot constants from supermimic.robots.h1 (no env)."""
    from supermimic.robots.h1 import (
        H1_ACTION_SCALE,
        H1_JOINT_NAMES,
        damping,
        default_pos,
        effort_limit,
        get_h1_robot_cfg,
        stiffness,
        velocity_limit,
    )

    h1_cfg = get_h1_robot_cfg()
    idx_map = {n: i for i, n in enumerate(H1_JOINT_NAMES)}
    n = len(joint_names)
    out = {
        k: np.full(n, np.nan)
        for k in ("kp", "kd", "effort_limit", "armature", "velocity_limit", "action_scale", "default_joint_pos")
    }
    for i, j in enumerate(joint_names):
        short_j = j.split("/")[-1]
        assert short_j in idx_map, f"unknown H1 joint {j}"
        k = idx_map[short_j]
        act = h1_cfg.articulation.actuators[k]
        out["kp"][i] = stiffness[k]
        out["kd"][i] = damping[k]
        out["effort_limit"][i] = effort_limit[k]
        out["armature"][i] = float(act.armature)
        out["velocity_limit"][i] = velocity_limit[k]
        out["action_scale"][i] = H1_ACTION_SCALE[short_j]
        out["default_joint_pos"][i] = default_pos[k]
    return out


def _resolve_sample_npz(clip: str, npz_dir: Path = NPZ_DIR) -> str:
    candidate = npz_dir / f"{clip}.npz"
    if candidate.exists():
        return str(candidate)
    for d in (npz_dir, Path("data/npz/lafan1"), Path("data/npz/bones_seed")):
        if d.exists():
            files = sorted(d.glob("*.npz"))
            if files:
                return str(files[0])
    raise FileNotFoundError(f"Could not find any .npz clip in {npz_dir}")


def _small_env(npz: list[str], num_envs: int, device: str, training_like: bool, future_frames: int = 10, anchor_obs: bool = False):
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env

    if training_like:
        args = SuperMimicEnvCfgArgs(npz_files=npz, num_envs=num_envs, record_transitions=False, future_frames=future_frames, anchor_obs=anchor_obs)
    else:
        args = SuperMimicEnvCfgArgs(
            npz_files=npz, num_envs=num_envs, future_frames=future_frames, episode_length_s=1.0e5, action_noise_std=0.0, obs_noise=False,
            reset_noise=False, domain_randomization=False, sampling_mode="start", record_transitions=False, seed=0, anchor_obs=anchor_obs,
        )
    return make_env(args, device=device)


def _gpu_field(env, name: str, mj_value: np.ndarray) -> float:
    """max |GPU model field - CPU mj_model field| over all worlds (nan if unavailable)."""
    try:
        t = getattr(env.sim.model, name)
        arr = torch.as_tensor(t).detach().cpu().double().numpy()
        return float(np.max(np.abs(arr.reshape(-1, *mj_value.shape) - mj_value[None])))
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] GPU field {name}: {e!r}")
        return float("nan")


def config(ckpts: tuple[str, ...] = CKPTS, device: str = "cuda:0", clip: str = SAFE_CLIPS[0], lock_waist: bool = False, npz_dir: Path = NPZ_DIR) -> dict:
    """Dump deploy_config.json for each checkpoint."""
    sample_npz = _resolve_sample_npz(clip, npz_dir)
    env = _small_env([sample_npz], 2, device, training_like=True)
    try:
        robot = env.scene["robot"]
        m = env.sim.mj_model
        names = list(robot.joint_names)
        nj = len(names)
        term = env.action_manager.get_term("joint_pos")
        assert list(term.target_names) == names, ("action order != joint order", term.target_names, names)
        jid = [m.joint(f"robot/{j}").id for j in names]
        act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
        aid = [act_of_joint[j] for j in jid]
        mj_kp = m.actuator_gainprm[aid, 0].astype(np.float64)
        mj_kd = -m.actuator_biasprm[aid, 2].astype(np.float64)
        assert np.allclose(-m.actuator_biasprm[aid, 1], mj_kp), "position actuator biasprm[1] != -kp"
        mj_force = m.actuator_forcerange[aid].astype(np.float64)
        assert np.all(m.actuator_forcelimited[aid]) and np.allclose(mj_force[:, 0], -mj_force[:, 1])
        mj_range = m.jnt_range[jid].astype(np.float64)
        mj_arm = m.dof_armature[m.jnt_dofadr[jid]].astype(np.float64)
        env_q0 = robot.data.default_joint_pos[0].cpu().double().numpy()
        soft = robot.data.soft_joint_pos_limits[0].cpu().double().numpy()
        scale = (term._scale[0] if isinstance(term._scale, torch.Tensor) else torch.full((nj,), term._scale)).cpu().double().numpy()
        offset = (term._offset[0] if isinstance(term._offset, torch.Tensor) else torch.full((nj,), term._offset)).cpu().double().numpy()
        c = constants_from_mjlab(names)
        soft_factor = float(robot.cfg.articulation.soft_joint_pos_limit_factor)
        cfg = {
            "robot": "h1",
            "joint_names": names,
            "default_joint_pos": c["default_joint_pos"].tolist(),
            "action_scale": c["action_scale"].tolist(),
            "kp": mj_kp.tolist(),
            "kd": mj_kd.tolist(),
            "effort_limit": mj_force[:, 1].tolist(),
            "joint_lower": mj_range[:, 0].tolist(),
            "joint_upper": mj_range[:, 1].tolist(),
            "velocity_limit": c["velocity_limit"].tolist(),
            "soft_limit_factor": soft_factor,
            "control_dt": float(env.step_dt),
            "armature": mj_arm.tolist(),
            "sim_dt": float(m.opt.timestep),
            "decimation": int(env.cfg.decimation),
            "source": "supermimic.deploy.export config (H1 compiled training model, cross-checked)",
        }
        if lock_waist:
            from supermimic.envs.lock_waist import LOCKED_JOINT_IDS, LOCKED_JOINT_NAMES

            assert [names[i] for i in LOCKED_JOINT_IDS] == list(LOCKED_JOINT_NAMES)
            cfg["locked_joints"] = list(LOCKED_JOINT_IDS)
            cfg["robot_model"] = "h1_19dof_lock_waist (training: --env.lock-waist, feasible references)"
        # ------------------------------------------------------------- cross checks
        mid, half = 0.5 * (mj_range[:, 0] + mj_range[:, 1]), 0.5 * (mj_range[:, 1] - mj_range[:, 0])
        checks = {
            "kp: h1 const vs compiled": np.abs(c["kp"] - mj_kp).max(),
            "kd: h1 const vs compiled": np.abs(c["kd"] - mj_kd).max(),
            "effort: const vs forcerange": np.abs(c["effort_limit"] - mj_force[:, 1]).max(),
            "armature: const vs compiled": np.abs(c["armature"] - mj_arm).max(),
            "scale: const vs action term": np.abs(c["action_scale"] - scale).max(),
            "q0: keyframe vs env default": np.abs(c["default_joint_pos"] - env_q0).max(),
            "offset: action term vs q0": np.abs(offset - env_q0).max(),
            "soft limits: formula vs env": max(np.abs(mid - soft_factor * half - soft[:, 0]).max(), np.abs(mid + soft_factor * half - soft[:, 1]).max()),
            "kp: GPU model vs mj_model": _gpu_field(env, "actuator_gainprm", m.actuator_gainprm),
            "kd: GPU biasprm vs mj_model": _gpu_field(env, "actuator_biasprm", m.actuator_biasprm),
            "forcerange: GPU vs mj_model": _gpu_field(env, "actuator_forcerange", m.actuator_forcerange),
            "jnt_range: GPU vs mj_model": _gpu_field(env, "jnt_range", m.jnt_range),
            "armature: GPU vs mj_model": _gpu_field(env, "dof_armature", m.dof_armature),
        }
        print("\n  cross-checks (max abs diff):")
        for k, v in checks.items():
            print(f"    {k:34s} {v:.3e}")
        bad = {k: v for k, v in checks.items() if not (v < 1e-5)}
        if bad:
            raise RuntimeError(f"deploy config cross-check failed: {bad}")
        print(f"  control_dt {env.step_dt} (sim dt {m.opt.timestep} x decimation {env.cfg.decimation}), soft factor {soft_factor}")
        print_table(cfg)
        for ck in ckpts:
            ck_cfg = dict(cfg)
            ck_path = ART / "ckpts" / Path(ck).name
            ck_path = ck_path if ck_path.exists() else Path(ck)
            if ck_path.exists():
                ck_data = torch.load(ck_path, map_location="cpu", weights_only=False)
                ck_cfg["anchor_obs"] = bool(ck_data.get("config", {}).get("env", {}).get("anchor_obs", False))
            out = ART / ckpt_tag(ck) / "deploy_config.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(ck_cfg, indent=1))
            print(f"  wrote {out}")
        return cfg
    finally:
        env.close()


def print_table(cfg: dict) -> None:
    hdr = f"  {'#':>2} {'joint':28s} {'q0':>7} {'scale':>8} {'kp':>9} {'kd':>8} {'effort':>6} {'lower':>8} {'upper':>8} {'vel':>5} {'armature':>9}"
    print(hdr)
    for i, n in enumerate(cfg["joint_names"]):
        print(
            f"  {i:2d} {n:28s} {cfg['default_joint_pos'][i]:7.3f} {cfg['action_scale'][i]:8.5f} {cfg['kp'][i]:9.4f} {cfg['kd'][i]:8.4f} "
            f"{cfg['effort_limit'][i]:6.1f} {cfg['joint_lower'][i]:8.4f} {cfg['joint_upper'][i]:8.4f} {cfg['velocity_limit'][i]:5.1f} {cfg['armature'][i]:9.6f}"
        )


# ---------------------------------------------------------------------------------------------- clips


def _ref71_from_npz(path: Path) -> np.ndarray:
    """51-d H1 reference rows exactly as MotionLibrary builds them (fp32)."""
    with np.load(path) as d:
        r = [str(n) for n in d["body_names"]].index("pelvis")
        return np.concatenate(
            [d["body_pos_w"][:, r], d["body_quat_w"][:, r], d["joint_pos"], d["body_lin_vel_w"][:, r], d["body_ang_vel_w"][:, r], d["joint_vel"]], axis=1
        ).astype(np.float32)


def default_joint_pos() -> np.ndarray:
    names = joint_names_from_mjcf()
    return constants_from_mjlab(names)["default_joint_pos"]


def joint_names_from_mjcf() -> list[str]:
    from supermimic.robots.h1 import get_h1_spec

    m = get_h1_spec().compile()
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != 0]  # skip the free joint


def clips(names: tuple[str, ...] = TEST_CLIPS, out: Path | None = None, compare_fp16: bool = True, lock_waist: bool = False, npz_dir: Path = NPZ_DIR) -> None:
    from supermimic.envs.mdp.motion_library import MotionLibrary, goal_features, zero_joint_columns
    from supermimic.kinematics.rotations import projected_gravity

    locked: tuple[int, ...] = ()
    if lock_waist:
        from supermimic.envs.lock_waist import LOCKED_JOINT_IDS

        locked = LOCKED_JOINT_IDS
    out = out or ART / ("clips_lockwaist" if lock_waist else "clips")
    zcols = zero_joint_columns(locked)

    out.mkdir(parents=True, exist_ok=True)
    jn = joint_names_from_mjcf()
    q0 = torch.as_tensor(default_joint_pos(), dtype=torch.float32)
    pack = None
    if compare_fp16 and COMPACT_LIB.exists():
        pack = torch.load(COMPACT_LIB, map_location="cpu", weights_only=False, mmap=True)
        pack_idx = {n: i for i, n in enumerate(pack["names"])}
    for name in names:
        src = npz_dir / f"{name}.npz"
        with np.load(src) as d:
            if "joint_names" in d:
                assert [str(x) for x in d["joint_names"]] == jn, f"{name}: npz joint order != MJCF joint order"
            fps = float(np.asarray(d["fps"]).reshape(-1)[0])
            extra = {k: d[k] for k in ("segment_names", "segment_kind", "segment_start", "segment_end") if k in d.files}
        assert abs(fps - 50.0) < 1e-6, f"{name}: fps {fps}"
        ref = torch.as_tensor(_ref71_from_npz(src))
        if locked:
            ref[:, zcols] = 0.0
        rs = MotionLibrary._split(ref)
        goal = goal_features(rs, q0).numpy().astype(np.float32)
        np.savez(
            out / f"{name}.npz",
            name=np.array(name),
            fps=np.float32(fps),
            joint_names=np.array(jn),
            default_joint_pos=q0.numpy(),
            goal=goal,
            joint_pos=rs.joint_pos.numpy(),
            proj_grav=projected_gravity(rs.root_quat_w).numpy(),
            root_pos=rs.root_pos_w.numpy(),
            root_quat_wxyz=rs.root_quat_w.numpy(),
            root_lin_vel_w=rs.root_lin_vel_w.numpy(),
            root_ang_vel_w=rs.root_ang_vel_w.numpy(),
            joint_vel=rs.joint_vel.numpy(),
            locked_joints=np.array(locked, dtype=np.int64),
            **extra,
        )
        msg = f"  {name:44s} T={goal.shape[0]:4d} -> {out / (name + '.npz')}"
        if pack is not None:
            if name not in pack_idx:
                msg += " | not in compact library"
            else:
                i = pack_idx[name]
                st, ln = int(pack["starts"][i]), int(pack["lengths"][i])
                assert ln == goal.shape[0], (name, ln, goal.shape)
                ref16 = pack["ref"][st : st + ln].to(torch.float32)
                if locked:
                    ref16[:, zcols] = 0.0
                g16 = goal_features(MotionLibrary._split(ref16), q0).numpy()
                d = np.abs(g16 - goal)
                terms = {
                    "q": d[:, 0:NJ],
                    "dq": d[:, NJ : 2 * NJ],
                    "grav": d[:, 2 * NJ : 2 * NJ + 3],
                    "angb": d[:, 2 * NJ + 3 : 2 * NJ + 6],
                    "linh": d[:, 2 * NJ + 6 : 2 * NJ + 9],
                    "h": d[:, 2 * NJ + 9 : 2 * NJ + 10],
                }
                msg += " | fp16-vs-fp32 goal max " + f"{d.max():.2e} (" + " ".join(f"{k} {v.max():.1e}" for k, v in terms.items()) + ")"
        print(msg)


# ---------------------------------------------------------------------------------------------- onnx


class _Export(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.policy(obs, goal)


def real_inputs(cfg_path: Path, clip_paths: list[Path], max_per_clip: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(obs, goal) pairs built by DeployCore from the clips' reference states (robot == reference)."""
    from supermimic.deploy.core import Clip, DeployConfig, DeployCore, SafetyCfg, quat_rotate_inverse

    dc = DeployConfig.load(cfg_path)
    nj = len(dc.joint_names)
    obs, goals = [], []
    rng = np.random.default_rng(0)
    for p in clip_paths:
        z = np.load(p)
        clip = Clip.load(p)
        rec = []
        core = DeployCore(dc, lambda o, g: (rec.append((o, g)), rng.normal(0, 0.5, nj))[1], clip, SafetyCfg(max_tilt_err=np.inf, max_vel_frac=np.inf))
        core.reset(0)
        q = z["root_quat_wxyz"].astype(np.float64)
        w = z["root_ang_vel_w"].astype(np.float64)
        for t in range(min(max_per_clip, clip.num_frames - 1)):
            core.step(z["joint_pos"][t], z["joint_vel"][t], quat_rotate_inverse(q[t], w[t]), q[t])
        obs += [o for o, _ in rec]
        goals += [g for _, g in rec]
    return np.asarray(obs, np.float32), np.asarray(goals, np.float32)


def onnx(ckpts: tuple[str, ...] = CKPTS, opset: int = 18, n_random: int = 256, clips_dir: Path = ART / "clips") -> dict:
    import onnx as onnx_lib

    from supermimic.eval.benchmark import load_policy_from_checkpoint

    ort = import_ort()
    res = {}
    for ck in ckpts:
        src = ART / "ckpts" / Path(ck).name
        src = src if src.exists() else Path(ck)
        policy, pcfg = load_policy_from_checkpoint(src, "cpu")
        assert pcfg["env"]["future_frames"] == 10, pcfg["env"]
        obs_dim = int(pcfg["policy"].get("history", 10)) * OBS_DIM
        goal_dim = int(pcfg["policy"]["goal_dim"])
        act_dim = ACT_DIM
        out = ART / ckpt_tag(ck) / "policy.onnx"
        out.parent.mkdir(parents=True, exist_ok=True)
        model = _Export(policy).eval()
        ex = (torch.zeros(1, obs_dim), torch.zeros(1, goal_dim))
        with torch.no_grad():
            prog = torch.onnx.export(model, ex, dynamo=True, input_names=["obs", "goal"], output_names=["action"], opset_version=opset, external_data=False)
        prog.save(str(out), external_data=False)
        m = onnx_lib.load(str(out))
        onnx_lib.checker.check_model(m)
        io = lambda vs: [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim], v.type.tensor_type.elem_type) for v in vs]  # noqa: E731
        ins, outs = io(m.graph.input), io(m.graph.output)
        assert ins == [("obs", [1, obs_dim], 1), ("goal", [1, goal_dim], 1)] and outs == [("action", [1, act_dim], 1)], (ins, outs)
        assert not any(t.data_location == onnx_lib.TensorProto.EXTERNAL for t in m.graph.initializer), "external data"
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        g = torch.Generator().manual_seed(0)
        rnd_o = torch.randn(n_random, obs_dim, generator=g) * 0.5
        rnd_g = torch.randn(n_random, goal_dim, generator=g) * 0.5
        clip_paths = [clips_dir / f"{c}.npz" for c in TEST_CLIPS if (clips_dir / f"{c}.npz").exists()]
        if not clip_paths and clips_dir.exists():
            clip_paths = sorted(clips_dir.glob("*.npz"))[:8]
        cfg_file = ART / ckpt_tag(ck) / "deploy_config.json"
        pairs = [("random", rnd_o.numpy(), rnd_g.numpy())]
        if clip_paths and cfg_file.exists():
            real_o, real_g = real_inputs(cfg_file, clip_paths)
            pairs.append(("real", real_o, real_g))
        else:
            real_o, real_g = rnd_o.numpy(), rnd_g.numpy()
        stats = {}
        for tag, O, G in pairs:
            with torch.no_grad():
                ref = policy(torch.as_tensor(O), torch.as_tensor(G)).numpy()
            got = np.concatenate([sess.run(["action"], {"obs": O[i : i + 1], "goal": G[i : i + 1]})[0] for i in range(len(O))])
            d = float(np.abs(got - ref).max())
            stats[tag] = {"n": len(O), "max_abs_diff": d, "max_abs_action": float(np.abs(ref).max())}
        if torch.cuda.is_available():
            pol_cuda = policy.to("cuda:0")
            with torch.no_grad():
                rc = pol_cuda(torch.as_tensor(real_o, device="cuda:0"), torch.as_tensor(real_g, device="cuda:0")).cpu().numpy()
            got = np.concatenate([sess.run(["action"], {"obs": real_o[i : i + 1], "goal": real_g[i : i + 1]})[0] for i in range(len(real_o))])
            stats["real_vs_torch_cuda"] = {"n": len(real_o), "max_abs_diff": float(np.abs(got - rc).max())}
        res[ckpt_tag(ck)] = {"path": str(out), "bytes": out.stat().st_size, "opset": opset, "inputs": ins, "outputs": outs, **stats}
        print(f"  {out} ({out.stat().st_size / 1e6:.1f} MB): " + ", ".join(f"{k} max|d| {v['max_abs_diff']:.2e} (n={v['n']})" for k, v in stats.items()))
        assert stats["random"]["max_abs_diff"] < 1e-4 and ("real" not in stats or stats["real"]["max_abs_diff"] < 1e-4), stats
    return res


# ---------------------------------------------------------------------------------------------- mjcf


def mjcf(out_dir: Path = ART / "models", device: str = "cuda:0", foot_friction: float | None = 1.0, npz_dir: Path = NPZ_DIR) -> dict:
    """Write the training scene (1 env) as standalone MJCF; options copied from the compiled sim model."""
    import tempfile

    import mujoco

    sample_npz = _resolve_sample_npz(SAFE_CLIPS[0], npz_dir)
    env = _small_env([sample_npz], 1, device, training_like=False)
    from mjlab.utils.spec import export_spec

    try:
        m = env.sim.mj_model
        spec = env.scene.spec.copy()
        for ent in env.scene._entities.values():
            for mesh in ent.spec.meshes:
                p = Path(ent.spec.modelfiledir) / mesh.compiler.meshdir / mesh.file
                if mesh.file and mesh.file not in spec.assets and p.is_file():
                    spec.assets[mesh.file] = p.read_bytes()
        o = spec.option
        for f in ("timestep", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "impratio", "integrator", "cone", "solver", "jacobian", "ccd_iterations", "noslip_iterations"):
            try:
                setattr(o, f, type(getattr(o, f))(getattr(m.opt, f)))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] option {f}: {e!r}")
        o.gravity = np.array(m.opt.gravity)
        n_feet = 0
        if foot_friction is not None:
            for gm in spec.geoms:
                if re.fullmatch(FOOT_GEOM_REGEX, gm.name):
                    gm.friction[0] = foot_friction
                    n_feet += 1
        with tempfile.TemporaryDirectory() as td:
            export_spec(spec, Path(td))
            out_dir.mkdir(parents=True, exist_ok=True)
            xml = (Path(td) / "scene.xml").read_text()
            (out_dir / "h1_train_scene.xml").write_text(xml)
            (out_dir / "g1_train_scene.xml").write_text(xml)
            if (Path(td) / "assets").exists():
                shutil.copytree(Path(td) / "assets", out_dir / "assets", dirs_exist_ok=True)
        # ------------------------------------------------------------- verify with plain mujoco (CPU)
        m2 = mujoco.MjModel.from_xml_path(str(out_dir / "h1_train_scene.xml"))
        d2 = mujoco.MjData(m2)
        mujoco.mj_forward(m2, d2)
        for _ in range(10):
            mujoco.mj_step(m2, d2)
        diffs = {}
        for f in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_trnid", "dof_armature", "dof_damping", "dof_frictionloss", "jnt_range",
                  "body_mass", "body_inertia", "body_ipos", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "qpos0"):
            a, b = np.asarray(getattr(m, f), np.float64), np.asarray(getattr(m2, f), np.float64)
            diffs[f] = float(np.abs(a - b).max()) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
        fr_a, fr_b = np.asarray(m.geom_friction), np.asarray(m2.geom_friction)
        changed = np.where(np.abs(fr_a - fr_b).max(1) > 0)[0]
        opts = {f: (float(getattr(m.opt, f)), float(getattr(m2.opt, f))) for f in ("timestep", "iterations", "ls_iterations", "impratio", "integrator", "cone", "solver", "tolerance")}
        info = {
            "path": str(out_dir / "h1_train_scene.xml"),
            "n_assets": len(list((out_dir / "assets").glob("*"))) if (out_dir / "assets").exists() else 0,
            "nq": m2.nq, "nv": m2.nv, "nu": m2.nu, "nbody": m2.nbody, "ngeom": m2.ngeom,
            "opt(env, dumped)": opts,
            "field_max_abs_diff_vs_env_mj_model": diffs,
            "foot_friction_set": foot_friction, "n_foot_geoms": n_feet,
            "geoms_with_changed_friction": [m2.geom(int(i)).name for i in changed],
            "kp": m2.actuator_gainprm[:, 0].tolist(), "kd": (-m2.actuator_biasprm[:, 2]).tolist(), "forcerange": m2.actuator_forcerange[:, 1].tolist(),
            "actuator_joints": [m2.joint(int(j)).name for j in m2.actuator_trnid[:, 0]],
            "stepped_10_ok": bool(np.all(np.isfinite(d2.qpos))),
        }
        print(json.dumps({k: v for k, v in info.items() if k not in ("kp", "kd", "forcerange", "actuator_joints")}, indent=1))
        return info
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- cli


@dataclass
class Args:
    cmd: str = "all"  # freeze | config | clips | onnx | mjcf | all
    ckpts: tuple[str, ...] = CKPTS
    clip_names: tuple[str, ...] = TEST_CLIPS
    device: str = "cuda:0"
    foot_friction: float | None = 1.0
    report: Path | None = None
    lock_waist: bool = False
    npz_dir: Path = NPZ_DIR


def main(a: Args) -> None:
    rep: dict = {}
    run = lambda c: a.cmd in (c, "all")  # noqa: E731
    if run("freeze"):
        print("== freeze"); freeze(a.ckpts)  # noqa: E702
    if run("config"):
        print("== config"); rep["config"] = config(a.ckpts, a.device, lock_waist=a.lock_waist, npz_dir=a.npz_dir)  # noqa: E702
    if run("clips"):
        print("== clips"); clips(a.clip_names, lock_waist=a.lock_waist, npz_dir=a.npz_dir)  # noqa: E702
    if run("onnx"):
        print("== onnx"); rep["onnx"] = onnx(a.ckpts, clips_dir=ART / ("clips_lockwaist" if a.lock_waist else "clips"))  # noqa: E702
    if run("mjcf"):
        print("== mjcf"); rep["mjcf"] = mjcf(device=a.device, foot_friction=a.foot_friction, npz_dir=a.npz_dir)  # noqa: E702
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main(tyro.cli(Args))
