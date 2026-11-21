# DUAL2 five-rod rollout: hand-off for Cinema 4D

This folder holds the exact simulated rollout shown in the videos (`../videos/side_clean.mp4`). It includes the robot, the bars and the motion.

## Files
| File | What it is |
|---|---|
| `dual2_five_rod.fbx` | **Start here.** Animated hierarchy plus the CAD meshes and the bars. Import into C4D with *File → Merge*, keep animation on and scale 1. |
| `dual2_five_rod.blend` | The same scene as a Blender file (the source of the FBX). |
| `parts/*.obj` + `materials.mtl` | One OBJ per robot link, in that link's own frame: metres, Z-up. Use these if you rebuild the rig yourself. |
| `joint_angles.csv` | For every frame: time, base position and quaternion, and the 8 joint angles in degrees. Use it to drive your own rig. |
| `body_world_transforms.csv` | For every frame: the world position and quaternion of every link. Use it to check a rebuilt rig, or to constrain objects directly. |
| `rollout.json` | Everything in machine-readable form: the hierarchy, joint axes, anchors and limits, materials, bar dimensions, and per-frame local transforms. |

## Timing
* 50 fps, 917 frames, 18.34 s.
* Frame `k` is at time `k / 50` s and matches row `k` in the CSVs. Frame 0 is the start.
* The first 50 frames (1 s) hold the start pose, the same as in the videos. The simulation then runs from `time_s = 0`.
* Interpolation is linear, with one key per frame on every link.
* If you import the FBX into Blender, its importer starts at frame 1, so everything shifts by one frame. C4D reads FBX time directly.

## Rig / hierarchy
* `DUAL2_rig` is a static null at the root.
* `base` is the floating body. It has a world position and rotation every frame.
* Arm 1 chain: `arm1_link1` (shoulder) → `arm1_link2` (elbow) → `arm1_wrist_pitch` → `arm1_wrist_roll` (fixed jaw) → `arm1_gripper` (moving jaw).
* Arm 2 chain: `arm2_wrist_pitch` (shoulder) → `arm2_wrist_roll` (fixed jaw) → `arm2_gripper` (moving jaw).
* Each link is a null that carries its mesh (`<link>_mesh`) at zero offset.
* Below `base`, each null only rotates about its single hinge, so you can swap the nulls for C4D joints one for one.
* Joint axes and pivot points, in each link's frame, are in `rollout.json → joints`.

| Joint | Moves link | Range (deg) |
|---|---|---|
| arm1_shoulder | arm1_link1 | −168.5 … 11.5 |
| arm1_elbow | arm1_link2 | 0 … 180 |
| arm1_wrist_pitch | arm1_wrist_pitch | −173.3 … 6.7 |
| arm1_wrist_roll | arm1_wrist_roll | −171.9 … 137.5 |
| arm1_gripper | arm1_gripper | −84.0 … 6.0 |
| arm2_shoulder | arm2_wrist_pitch | −102.8 … 77.2 |
| arm2_wrist_roll | arm2_wrist_roll | 0 … 180 |
| arm2_gripper | arm2_gripper | −48.7 … 41.3 |

## Units and axes
* Source data is in metres, Z-up, with x as the travel direction. The rods run along y.
* The FBX was exported from Blender with unit scale applied and the standard FBX axis conversion. C4D should place the robot upright at real size.
* Real size means the rods are 22 cm apart and hang 55 cm above the floor.
* The robot weighs 0.89 kg. Its arms reach about 45 cm from gripper to gripper.

## Bars (metres)
| Part | Shape | Size and position |
|---|---|---|
| Rods 1–5 | Capsule, radius 0.011 (Ø 22 mm) | From y = −0.16 to +0.16 (320 mm between cap centres), at z = 0.55 and x = 0, 0.22, 0.44, 0.66, 0.88 |
| Rails (×2) | Cylinder, radius 0.012 | At y = ±0.16, z = 0.62, from x = −0.30 to 1.18. Visual only, so extend them as long as you like. |
| Drop links (×10) | Cylinder, radius 0.006 | One at each rod end, at y = ±0.16, from z = 0.55 to 0.62. Visual only. |

Only the rods existed in the physics simulation. The rails and drop links are visual dressing that holds the rods up in the renders.

## Colours used in the renders
* Green parts: `#53c81d` (darkened to about `#337a12` in `side_clean.mp4`)
* White horns: `#eaeaea`
* Servos: `#4d4d4d`
* Screws: `#9aa4ae`
* Rods: teal `#5cc8b8`, blue `#5aaef0`, amber `#f2b54f`, violet `#b48cf0`, coral `#f5826c` (rods 1 to 5)

## Regenerating
From `experiments/five_rod`:
1. Run `python export_rollout.py`. It writes the CSVs, the JSON and the OBJ files.
2. Run `/Applications/Blender.app/Contents/MacOS/Blender -b -P build_c4d.py`. It writes the FBX, the .blend and a GLB.
3. The Blender script ends with a check that compares the rebuilt motion against the simulation. The current error is 0.0007 mm.
