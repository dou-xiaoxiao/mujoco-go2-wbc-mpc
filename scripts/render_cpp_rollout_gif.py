"""Render a C++ headless rollout CSV as a MuJoCo GIF or viewer replay."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "cpp_trot_turn.csv"
DEFAULT_GIF_PATH = PROJECT_ROOT / "outputs" / "cpp_trot_turn.gif"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a C++ MPC/WBC rollout CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--gif-output", type=Path, default=DEFAULT_GIF_PATH)
    parser.add_argument("--viewer-replay", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")

    csv_path = absolute_path(args.csv)
    gif_path = absolute_path(args.gif_output)
    times, qpos, qvel = load_cpp_csv(csv_path)
    print(f"loaded C++ rollout: {csv_path} ({qpos.shape[0]} frames)")

    if not args.no_gif:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        render_gif(qpos, gif_path, args.width, args.height, args.fps, args.playback_speed)
        print(f"saved gif: {gif_path}")

    if args.viewer_replay:
        replay_viewer(qpos, qvel, args.fps, args.playback_speed)

    del times


def absolute_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_cpp_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    nq = model.nq
    nv = model.nv
    expected_cols = 1 + nq + nv
    if raw.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns for nq={nq}, nv={nv}, got {raw.shape[1]}")
    times = raw[:, 0]
    qpos = raw[:, 1 : 1 + nq]
    qvel = raw[:, 1 + nq : 1 + nq + nv]
    return times, qpos, qvel


def render_gif(qpos: np.ndarray, gif_path: Path, width: int, height: int, fps: float, playback_speed: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -20.0

    frames: list[Image.Image] = []
    for idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[idx]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.qpos[0:3]
        camera.lookat[2] = 0.20
        renderer.update_scene(data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))

    duration_ms = max(1, int(round(1000.0 / (fps * playback_speed))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def replay_viewer(qpos: np.ndarray, qvel: np.ndarray, fps: float, playback_speed: float) -> None:
    frame_dt = 1.0 / (fps * playback_speed)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for idx in range(qpos.shape[0]):
            if not viewer.is_running():
                break
            frame_start = time.perf_counter()
            data.qpos[:] = qpos[idx]
            data.qvel[:] = qvel[idx]
            mujoco.mj_forward(model, data)
            viewer.sync()
            elapsed = time.perf_counter() - frame_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)


if __name__ == "__main__":
    main()
