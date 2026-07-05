from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco
from src.envs.brachiation_env import BrachiationEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mujoco" / "robot.xml"


def test_mujoco_model_loads_required_contract_names() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    assert model.nu == 8
    assert model.nq >= 15

    required_sites = ["arm1_tip", "arm2_tip", "target_site"]
    for site in required_sites:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0

    required_sensors = ["arm1_touch", "arm2_touch", "base_pos", "base_quat"]
    for sensor in required_sensors:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor) >= 0

    required_keyframes = ["hanging", "wall1_grip"]
    for keyframe in required_keyframes:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe) >= 0


def test_env_reset_and_step_are_finite() -> None:
    env = BrachiationEnv(
        render_mode=None,
        initial_keyframe="wall1_grip",
        curriculum_level=8,
        max_episode_steps=5,
    )

    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        assert "walls_cleared" in info

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "total_reward" in info
    finally:
        env.close()


def test_normalized_actions_map_to_actuator_control_ranges() -> None:
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip")

    try:
        low_action = np.full(env.action_space.shape, -1.0, dtype=np.float32)
        high_action = np.full(env.action_space.shape, 1.0, dtype=np.float32)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        np.testing.assert_allclose(
            env._normalized_action_to_ctrl(low_action), env.actuator_ctrl_low
        )
        np.testing.assert_allclose(
            env._normalized_action_to_ctrl(high_action), env.actuator_ctrl_high
        )
        np.testing.assert_allclose(
            env._normalized_action_to_ctrl(zero_action),
            (env.actuator_ctrl_low + env.actuator_ctrl_high) / 2.0,
        )

        env.reset(seed=123)
        env.step(high_action)
        assert np.all(env.data.ctrl <= env.actuator_ctrl_high + 1e-9)
        assert np.all(env.data.ctrl >= env.actuator_ctrl_low - 1e-9)
    finally:
        env.close()


def test_curriculum_level_can_be_updated_between_resets() -> None:
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip", curriculum_level=0)

    try:
        _, start_info = env.reset(seed=123)
        env.set_curriculum_level(8)
        _, later_info = env.reset(seed=123)

        assert env.curriculum_level == 8
        assert later_info["initial_pos"][0] > start_info["initial_pos"][0] + 1.0
    finally:
        env.close()


def test_reset_options_can_override_curriculum_level() -> None:
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip", curriculum_level=0)

    try:
        _, info = env.reset(seed=123, options={"curriculum_level": 8})

        assert env.curriculum_level == 8
        assert info["walls_cleared"] >= 7
    finally:
        env.close()


def test_reset_starts_with_grip_contact_on_current_bar() -> None:
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip", curriculum_level=8)

    try:
        _, info = env.reset(seed=123)
        target_bar = f"bar{info['walls_cleared'] + 1}"
        contact_geoms = set()
        wall_contact_geoms = set()
        for contact_idx in range(env.data.ncon):
            contact = env.data.contact[contact_idx]
            for geom_id in (contact.geom1, contact.geom2):
                geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if geom_name is not None:
                    contact_geoms.add(geom_name)
                    if geom_name.startswith("wall"):
                        wall_contact_geoms.add(geom_name)

        assert target_bar in contact_geoms
        assert not wall_contact_geoms
        assert env._get_touch_sensor("arm1_touch") > 0.01

        action = (
            2.0
            * (env.data.ctrl - env.actuator_ctrl_low)
            / (env.actuator_ctrl_high - env.actuator_ctrl_low)
            - 1.0
        ).astype(np.float32)
        _, _, _, _, step_info = env.step(action)
        assert step_info["bar_contact_count"] > 0
        assert step_info["bar_grip_reward"] > 0
        assert step_info["grip_assist_active"] == 1.0
    finally:
        env.close()


def test_hold_action_keeps_initial_bar_grip() -> None:
    env = BrachiationEnv(render_mode=None, initial_keyframe="wall1_grip", curriculum_level=8)

    try:
        _, info = env.reset(seed=123)
        action = (
            2.0
            * (env.data.ctrl - env.actuator_ctrl_low)
            / (env.actuator_ctrl_high - env.actuator_ctrl_low)
            - 1.0
        ).astype(np.float32)
        target_bar = env._bar_center(info["walls_cleared"])
        step_info = {}
        terminated = False
        truncated = False

        for _ in range(150):
            _, _, terminated, truncated, step_info = env.step(action)

        arm1_tip = env._get_site_pos("arm1_tip")
        radial_error = np.linalg.norm((arm1_tip - target_bar)[[0, 2]])
        assert not terminated
        assert not truncated
        assert env.data.qpos[2] > 0.0
        assert radial_error < 0.03
        assert step_info["grip_assist_active"] == 1.0
        assert step_info["bar_grip_reward"] > 0
    finally:
        env.close()
