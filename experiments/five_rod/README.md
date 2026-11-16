# DUAL2 five-rod traversal (rods 22 cm apart, self-collision on)

## Videos (1920×1080, 25 fps, real time; all four are the same continuous run)
| File | View |
|---|---|
| `three_quarter.mp4` | Elevated three-quarter view that follows the robot |
| `side.mp4` | Telephoto side view: rods seen end-on, so you can read the swing |
| `gripper_closeup.mp4` | Close-up that follows the two grippers, for judging each catch and release |
| `triptych.mp4` | All three views in one frame |

The overlay shows simulation time, transfers completed (out of 4), the current phase, and the vertical load on each rod.
Every frame comes from the physics simulation. Nothing is keyframed for the video.

## The task
The robot starts hanging from rods 1 and 2. It moves hand over hand (2→3, 3→4, 4→5) until it hangs from rod 5.
Each transfer has to pass all of these checks:
* the new hand is closed on the target rod, with both jaws touching it, within 2.5 cm of the intended spot
* the old hand has let go
* the new hand carries more than 3 N for a full 1 s
* on the last transfer, the robot then keeps hanging from rod 5 for another 3 s
* the robot never touches the floor, and only the gripper jaws touch the rods

## What changed in this version
* **Rod spacing went from 10 cm to 22 cm.** Each swing now covers 44 cm, and the grippers sit ±4.5 cm apart sideways on the rods so the swinging arm can pass the holding arm.
* **Self-collision is on.** Each link has a massless collision box fitted to its CAD meshes. The boxes collide with each other, but not with the rods or the floor, and MuJoCo skips directly connected links. The gait keyframes are solved with zero box overlap. During motion the largest overlap is a few mm, which is the normal give of MuJoCo's soft contacts.
* **The center of mass shifts under the new hand before the old hand lets go.** This removes the big pendulum swing that used to shake the grip loose.

## What the policy takes as input (every 20 ms control step)
The policy is a phase-based feedback controller. Each of the four transfers has its own set of 16 learned parameters.

**Observations**
1. **Joint positions** of the 8 servos: arm 1 shoulder, elbow, wrist pitch, wrist roll and gripper; arm 2 shoulder, wrist roll and gripper.
2. **Gripper pose** of both hands, from forward kinematics: palm position and orientation.
3. **Whole-body center of mass** and its Jacobian. The controller uses these to predict how the body will rotate about the holding hand under gravity.
4. **Gripper–rod contact sensing**, for every hand and rod pair:
   * which jaw is touching (fixed jaw, moving jaw, or both)
   * the normal force
5. **Phase information:**
   * which transfer is running
   * which hand is swinging
   * the target rod's position (rods are 22 cm apart)
   * time within the phase

**Outputs:** 8 servo position targets. The servos use position control with ±1.5 N·m torque limits.

**How one transfer runs**
1. A reference trajectory blends through five whole-body keyframes, solved offline by IK with a self-collision penalty: lift off, mid-swing (passing beside the holding arm), below the target, on the target, then shift the center of mass.
2. A Jacobian feedback term pulls the swinging palm toward the target rod, in the frame of the holding hand. It is clipped to a learned maximum.
3. The gripper closes when the palm is within a learned distance of the rod, or when both jaws touch it.
4. After a learned number of control steps with a secure two-jaw grip, the center of mass shifts under the new hand.
5. Only then does the old gripper open.

**The 16 learned numbers per transfer:** duration, feedback-gain limit, closing distance, grip-confirm delay, 6 constant joint offsets and 6 mid-swing joint offsets.

## How it was trained
Training uses the cross-entropy method in MuJoCo. It is a black-box search, not a neural network.
1. First pass: each transfer is optimized from the nominal start.
2. Robustness pass: every candidate is scored on many perturbed start states at once, and the score rewards how many of them succeed. The perturbed starts are shoulder offsets of ±0.05 rad, plus random ±0.03 rad offsets on all joints.
3. The last transfer must also survive the 3 s hang.

## Robustness audit
Audit of the pushed parameters (`python audit3.py`): full traversal including the 3 s final hang on 8/21 shoulder-offset starts (nominal included) and 3/20 random multi-joint starts; zero floor contact; max self-overlap 6.6 mm (soft contact). Transfer 3 is the weak link; a wider robustness pass is in progress.

## Caveats
* The controller reads gripper and center-of-mass positions straight from the simulator. A real robot would need to estimate them from joint encoders, plus an IMU or the known rod positions.
* Self-collision uses fitted boxes, not exact meshes. That is good enough to stop the arms passing through each other, but it is not exact.
* At the end the robot hangs from rod 5 with one hand. The other arm stays open.

## Reproduce
In this folder:
* `python render3.py` renders all four videos
* `python audit3.py` runs the robustness audit
* `python search.py` then `python robust.py <stage> <gens> [tail]` retrains
