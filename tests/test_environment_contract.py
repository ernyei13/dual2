from __future__ import annotations

from pathlib import Path

import numpy as np

import mujoco
from src.envs.brachiation_env import BrachiationEnv

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mujoco" / "robot.xml"


def _bar_contact_bodies(env: BrachiationEnv) -> set[str]:
    bodies = set()
    for contact_idx in range(env.data.ncon):
        contact = env.data.contact[contact_idx]
        geoms = []
        for geom_id in (contact.geom1, contact.geom2):
            geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_name = mujoco.mj_id2name(
                env.model,
                mujoco.mjtObj.mjOBJ_BODY,
                env.model.geom_bodyid[geom_id],
            )
            geoms.append((geom_name, body_name))

        if geoms[0][0] is not None and geoms[0][0].startswith("bar"):
            bodies.add(str(geoms[1][1]))
        elif geoms[1][0] is not None and geoms[1][0].startswith("bar"):
            bodies.add(str(geoms[0][1]))

    return bodies


def test_mujoco_model_loads_required_contract_names() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    assert model.nu == 8
    assert model.nq >= 15
    np.testing.assert_allclose(model.opt.gravity, np.array([0.0, 0.0, -9.81]))

    required_sites = ["arm1_tip", "arm2_tip", "target_site"]
    for site in required_sites:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm1_hook_top") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm2_hook_top") < 0

    for wall_idx in range(1, 11):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wall{wall_idx}")
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0
        bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"bar{wall_idx}")
        np.testing.assert_allclose(model.geom_size[bar_id, 0], 0.008)
        np.testing.assert_allclose(model.geom_friction[bar_id], np.array([8.0, 4.0, 0.1]))

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
        contact_bodies = _bar_contact_bodies(env)
        assert "arm1_gripper" in contact_bodies
        assert "arm1_wrist_roll" in contact_bodies

        action = (
            2.0
            * (env.data.ctrl - env.actuator_ctrl_low)
            / (env.actuator_ctrl_high - env.actuator_ctrl_low)
            - 1.0
        ).astype(np.float32)
        _, _, _, _, step_info = env.step(action)
        assert step_info["bar_contact_count"] > 0
        assert step_info["bar_grip_reward"] > 0
        assert step_info["grip_assist_active"] == 0.0
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
        step_info = {}
        terminated = False
        truncated = False

        for _ in range(60):
            _, _, terminated, truncated, step_info = env.step(action)

        contact_bodies = _bar_contact_bodies(env)
        assert not terminated
        assert not truncated
        assert env.data.qpos[2] > -0.1
        assert step_info["bar_contact_count"] > 0
        assert "arm1_gripper" in contact_bodies
        assert "arm1_wrist_roll" in contact_bodies
        assert step_info["walls_cleared"] == info["walls_cleared"]
        assert step_info["grip_assist_active"] == 0.0
        assert step_info["bar_grip_reward"] > 0
    finally:
        env.close()
