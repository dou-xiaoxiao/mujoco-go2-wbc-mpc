"""Smoke test for the new MuJoCo WBC/MPC environment."""

from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "free_body_smoke.xml"


def dense_mass_matrix(model, data):
    mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, mass)
    return mass


def main():
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"model path: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("\n=== Model dimensions ===")
    print(f"nq = {model.nq}")
    print(f"nv = {model.nv}")
    print(f"nu = {model.nu}")
    print(f"nbody = {model.nbody}")
    print(f"nsite = {model.nsite}")

    mass = dense_mass_matrix(model, data)
    print("\n=== Mass matrix ===")
    print(f"M shape = {mass.shape}")
    print(f"diag(M) = {np.round(np.diag(mass), 4).tolist()}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "front_right_foot")
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    print("\n=== Site Jacobian ===")
    print(f"front_right_foot jacp shape = {jacp.shape}")
    print(f"front_right_foot jacr shape = {jacr.shape}")
    print(f"jacp first row = {np.round(jacp[0], 4).tolist()}")

    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    print("\n=== Inverse dynamics ===")
    print(f"qfrc_inverse shape = {data.qfrc_inverse.shape}")
    print(f"qfrc_inverse = {np.round(data.qfrc_inverse, 4).tolist()}")

    print("\nMuJoCo WBC environment smoke test passed.")


if __name__ == "__main__":
    main()
