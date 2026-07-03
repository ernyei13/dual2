from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

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

        np.testing.assert_allclose(env._normalized_action_to_ctrl(low_action), env.actuator_ctrl_low)
        np.testing.assert_allclose(env._normalized_action_to_ctrl(high_action), env.actuator_ctrl_high)
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
