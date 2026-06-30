# Dual2 Repository Assessment And Improvement Plan

Date: 2026-06-30

## Executive Summary

`dual2` is in a promising research-prototype state. The repository has a concrete MuJoCo model, CAD/URDF assets, a Gymnasium-compatible environment, PPO training scripts, a curriculum concept, and early checkpoint/log evidence. It is not yet in a reliable engineering state for repeatable robotics/RL experiments.

The immediate goal should be to make the project reproducible, measurable, and safe to iterate on before spending more training time. The current code can compile, but this checkout cannot run the runtime smoke path without installing dependencies, and the repo lacks tests, CI, pinned dependency versions, a single canonical training flow, and a clear artifact policy.

## Current State

### What Is Working

- The project has a clear domain: dual-arm brachiation over a MuJoCo wall/bar course.
- The MuJoCo XML model, mesh assets, Gymnasium environment, and PPO scripts are all present.
- The README gives a useful quick start and explains the intended curriculum training path.
- `.gitignore` already identifies generated training outputs, local environments, logs, and videos as artifacts.
- A syntax check passes for `src` and `scripts` with `python3 -m compileall src scripts`.

### Main Risks

- There is no automated test suite or CI gate, so physics, reward, observation, and script regressions are easy to miss.
- Dependencies are loosely bounded in `requirements.txt`, with no lock file or Python version contract.
- The project has multiple overlapping training entry points (`src/main.py`, `scripts/train_policy.py`, `scripts/train_grasping.py`, `scripts/train_to_goal_with_visual.py`) with different assumptions.
- Training artifacts are still tracked in git even though `.gitignore` marks checkpoints and logs as regenerated outputs.
- The environment is a large single file, currently mixing simulation setup, observation construction, reward shaping, curriculum behavior, rendering, and Gym registration.
- The curriculum callback records advancement but does not yet update the live vectorized environments.
- The action space claims normalized `[-1, 1]` controls, but `step()` clips actions and sends them directly to MuJoCo controls. This should be audited against actuator `ctrlrange` and the intended position-control semantics.
- The current training evidence does not show a solved traversal policy. The checked-in `train_v5.log` shows short evaluation episodes and negative rewards near the end of the logged run, even though grip contact appears to improve.

### Verification Performed

- Cloned `https://github.com/ernyei13/dual2.git` at `main`.
- Confirmed the checkout is clean before adding this document.
- Listed tracked files and confirmed checkpoints/logs are committed despite the artifact policy implied by `.gitignore`.
- Ran `python3 -m compileall src scripts`, which passed.
- Tried `python3 scripts/visualize.py --mode info`, which failed in this workspace because `numpy` is not installed. That is an environment bootstrap issue, not proof of model failure.

## Improvement Plan

### Phase 1: Make The Repo Reproducible

Objective: any contributor should be able to create the same dev environment and run a smoke check.

Actions:

- Add a `pyproject.toml` or equivalent project metadata file with Python version, runtime dependencies, and dev dependencies.
- Pin dependencies with a lock workflow, for example `uv.lock`, `requirements-lock.txt`, or another chosen package manager.
- Add a documented bootstrap command that creates a local environment and installs the project.
- Add a small smoke command that imports MuJoCo, loads `mujoco/robot.xml`, constructs `BrachiationEnv`, resets it, and takes a few random steps.
- Add a troubleshooting note for local MuJoCo/headless rendering setup.

Acceptance criteria:

- Fresh checkout can run a documented install command.
- Fresh checkout can run a documented smoke command.
- Smoke command verifies model load, observation shape, action shape, reset, step, no NaNs, and clean close.

### Phase 2: Add Test And CI Guardrails

Objective: make environment and model regressions visible before training.

Actions:

- Add `pytest` tests for XML/model load, keyframe existence, site/sensor/actuator names, observation shape, action shape, deterministic reset with seed, finite rewards, and termination behavior.
- Add tests for reward components at controlled states where possible.
- Add a GitHub Actions workflow for formatting/linting, compile check, and the smoke tests.
- Keep GUI/video tests out of default CI unless they are explicitly headless and stable.

Acceptance criteria:

- CI runs on every push and pull request.
- Tests fail if required MuJoCo names disappear or observation/action contracts change unexpectedly.
- Tests complete quickly enough to run during normal development.

### Phase 3: Consolidate The Training Pipeline

Objective: replace overlapping scripts with one clear training/evaluation path.

Actions:

- Choose one canonical CLI entry point for training, evaluation, and recording.
- Move experiment configuration into versioned config files for tasks like `grasping`, `single_transfer`, and `full_traversal`.
- Save every run with its config, git commit, dependency versions, seed, normalization stats, evaluation metrics, and optional video.
- Make evaluation load exactly the wrappers used during training, including `VecNormalize` and `VecFrameStack`.
- Fix `record_agent.py` so it does not default to evaluating a traversal checkpoint in `grasping` mode.

Acceptance criteria:

- One command starts training from a named config.
- One command evaluates a checkpoint and emits machine-readable metrics.
- One command records a video using the correct task, wrappers, and normalization stats.

### Phase 4: Correct Environment Semantics

Objective: make the Gym environment contract explicit and physically meaningful.

Actions:

- Audit action scaling. Either map normalized actions to each actuator `ctrlrange`, or define the action space directly in physical control units.
- Use the existing `joint_ranges` data or remove it if it is not part of the chosen action model.
- Split the environment into smaller modules for model constants, observations, rewards, termination, curriculum/reset logic, and rendering.
- Replace magic numbers in reward/reset logic with named constants or config values.
- Add a real curriculum update mechanism for vectorized environments.
- Make seeding cover NumPy, Gymnasium environment randomness, Stable-Baselines3, and PyTorch.

Acceptance criteria:

- Action and observation contracts are documented and tested.
- Curriculum level changes actually affect the next resets of all training environments.
- Reward component magnitudes are logged and bounded enough to catch reward hacking.

### Phase 5: Improve The Learning Curriculum

Objective: train the robot through measurable subskills instead of relying on one hard sparse task.

Actions:

- Define staged tasks with explicit success metrics:
  - hold grip without falling,
  - controlled swing while maintaining one grip,
  - reach and contact the next bar,
  - transfer grip to the next bar,
  - clear one bar interval,
  - traverse the full course.
- Promote the best current grasping setup into a documented pretraining stage.
- Add deterministic evaluation episodes for every stage.
- Use domain randomization only after a stage is stable under nominal physics.
- Track success rate, episode length, bars cleared, time-to-first-contact, grip loss count, and fall count.

Acceptance criteria:

- Each task has a pass/fail threshold.
- A training run report shows which stage is solved and which stage is blocking progress.
- Full traversal training begins from a checkpoint that has already passed the prerequisite transfer tasks.

### Phase 6: Clean Up Repository Hygiene

Objective: keep source code, assets, and generated artifacts clearly separated.

Actions:

- Decide whether checkpoints belong in git, Git LFS, releases, or an external experiment store.
- Remove generated checkpoints, TensorBoard event files, and raw training logs from normal git tracking after choosing the artifact destination.
- Add a lightweight artifact manifest for any checkpoint that should remain discoverable.
- Document the source of CAD, URDF, and STL assets, including how to regenerate or update them.
- Avoid scripts that rewrite `mujoco/robot.xml` by default. Make candidate-generation scripts write explicit output files or require a confirmation flag.

Acceptance criteria:

- `git status` stays clean after normal training/evaluation runs.
- Generated artifacts can be reproduced or downloaded from a documented location.
- Source model edits are intentional and reviewable.

## Recommended Next Pull Requests

1. Reproducibility and smoke tests: add project metadata, dev dependencies, smoke script, and pytest coverage for model/environment contracts.
2. CI: run compile and smoke tests on GitHub Actions.
3. Training CLI consolidation: pick the canonical entry point and migrate the other scripts behind config-driven modes or deprecate them.
4. Environment semantics: fix or document action scaling, add curriculum updates, and split reward/observation code into testable units.
5. Artifact cleanup: move tracked regenerated outputs out of normal git history going forward and document where durable experiment artifacts live.

## Definition Of Better

The repository is meaningfully better when a fresh contributor can install it, run a smoke test, train a named stage, evaluate a checkpoint, and understand from metrics whether the robot is improving. At that point, training failures become diagnosable engineering feedback instead of expensive guesswork.
