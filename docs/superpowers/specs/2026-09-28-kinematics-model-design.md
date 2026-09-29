# Kinematic Model and Visualization

Date: 2026-09-28
Status: Draft, awaiting review

## Purpose

Build a forward-kinematics model and 3D visualization of the custom 6-DoF arm,
so the kinematic configuration can be validated numerically and `d0_base_height`
settled before geometry is committed to CAD.

The DH table this is built from was derived by hand and is unverified. Verifying
it is the first job of this work, not an assumption behind it.

Scope of this spec: the kinematics module, its configuration file, its tests,
and the visualization. Excluded: inverse kinematics, workspace sweeps (blocked,
see Deferred), URDF export, and the existing AR4 code (`src/robot/`, `docker/`,
CI), which stays untouched.

## Input

The kinematic configuration handoff (joint architecture, DH parameters, link
dimensions, wrist geometry). Its substance is carried into
`utilities/kinematics/arm.yaml` rather than restated here, so there is one
machine-readable copy rather than two prose copies that can drift.

Joint indices are 0-based — in this document, in the config, in the code, and in
any future URDF. Never renumbered to 1-based.

## Findings: corrections to the handoff

A sensitivity probe run against the handoff's DH table before designing
established the following. These supersede the handoff's Wrist Geometry section.

| # | Finding | Evidence |
|---|---|---|
| F1 | `d4_roll_offset` occupies the **d slot of row 3**, not row 4. Renamed `d3_roll_offset` to satisfy the project's own "prefix matches DH symbol" rule. | Inspection of the table |
| F2 | `d3_roll_offset` does **not** control wrist sphericity. It translates along the J3 axis, displacing the wrist out of the arm plane. | Sweeping it 0 → 200 mm leaves the J3↔J5 axis distance at exactly 0.000 mm |
| F3 | `a4_yaw_to_roll` is the actual non-spherical offset. | The J4↔J5 axis distance equals `a4` exactly at every value tested; at `a4 = 0` all three wrist axes meet |
| F4 | The handoff's proposed check — "minimum distance between the J3 and J5 axes equals the offset" — is **pose-dependent** and cannot be used as written. | Same configuration reads 100 mm at `q = 0` and 0 mm at `q = [.3,.4,-.5,.6,.7,.2]` |
| F5 | The "two legs of one L-bracket" cannot both be perpendicular offsets between J4 and J5, because two skew lines have exactly one perpendicular distance. `a4` is one leg; the other is unrepresented in the table. | Geometric |
| F6 | What `d3_roll_offset` actually does is slide the whole wrist assembly — J4, J5 and the TCP — laterally out of the arm plane by exactly its value. It does **not** disturb the J1/J2/J3 position chain. | J4 and J5 plane excursion equals `d3` at 25/60/120 mm, in every pose tested; J2 and J3 excursion stays 0.000 throughout |

Consequences:

- The arm remains **non-spherical** — `a4_yaw_to_roll` is 100 mm, not zero — so
  the handoff's downstream conclusion stands unchanged: no closed-form IK,
  numerical IK seeded from current joint state, with a convergence-failure path.
- The parameter the handoff lists as the one open number deciding spherical vs.
  non-spherical (`d4_roll_offset`, `TBD`) is not that parameter. The number that
  decides it, `a4_yaw_to_roll`, already has an estimated value.
- **`d3_roll_offset` is removed** (decided 2026-09-28). Its only stated
  justification was the non-spherical offset, which it does not provide (F2).
  What it does provide — a lateral wrist offset (F6) — is not called for
  anywhere in the design, and a non-zero value would cost the property that a
  single 2D sketch exactly represents the chain out to the tool. Row 3 keeps
  `d: 0`; the named parameter is gone. It returns if wrist packaging ever
  demands a lateral shift, the same way `a3_wrist_pitch_to_yaw` returns if J3
  and J4 separate — with a reason attached, rather than as an unexplained
  degree of freedom.
- Whether the L-bracket's second leg (F5) needs its own slot remains a
  mechanical-design question, resolved in that conversation and not invented
  here.
- The sphericity check is replaced by a pose-invariant one (see Verification).

## Agreed definition: `a4_yaw_to_roll`

The one parameter most likely to be misunderstood between this repo, the CAD
model and the design conversation, so it is pinned down here. All four
statements below are measured, not asserted.

1. **It is the perpendicular distance between the J4 (wrist yaw) and J5 (tool
   roll) axes.** Measured J4↔J5 distance equals `a4` exactly at 0, 10, 45, 100
   and 250 mm, with the two axes at 90° in every case.
2. **It is the sole control of wrist sphericity.** At `a4 = 0` all three wrist
   axes meet at a point and the wrist is spherical, which would restore
   closed-form IK. At any non-zero value the decoupling fails and IK must be
   numerical. No other parameter in the table affects this.
3. **It is one leg of the L-bracket, and the only leg the table represents.**
   The tool-side leg runs along the J5 axis and is currently folded into
   `d5_tool_length`. Two skew lines admit exactly one perpendicular distance
   (F5), so `a4` cannot represent both.
4. **It adds to reach.** TCP distance at home goes from 461.0 mm at `a4 = 0` to
   550.0 mm at `a4 = 100`. The offset is not purely a cost; it extends the arm
   like a short link.

**Status: decision-bearing, not yet ratified.** The handoff carries `a4` as a
100 mm "estimate" while treating sphericity as an open question under a
different parameter name. In fact this value already decided it. At 100 mm
against 200 mm links — half a link length — it is large for a wrist offset, and
the cost lands as workspace asymmetry and slower IK convergence. Ratifying or
revising the magnitude belongs to the mechanical-design conversation; the
visualization exists partly to inform that call, which is why `a4` gets a
slider.

## Layout

```
utilities/
  kinematics/
    requirements.txt        # numpy, matplotlib; roboticstoolbox-python as dev extra
    arm.yaml                # the kinematic configuration; the single source of truth
    config.py               # load + validate arm.yaml into a typed ArmConfig
    kinematics.py           # forward kinematics; pure numpy, no I/O, no plotting
    plot_arm.py             # interactive 3D visualization; entry point
    test_config.py          # loader behaviour
    test_kinematics.py      # FK correctness
    test_verification.py    # verifies the DH table itself, not the code
```

Scripts live under `utilities/`, never in `requirements/`, matching the
convention set by the requirements-system spec. The venv is independent of the
ROS Docker image.

## Configuration (`arm.yaml`)

The DH table is data, not code. Rows reference parameters by name, so a
parameter can be changed, or the table restructured, without touching
kinematics.

```yaml
units: {length: mm, angle: deg}      # angles converted to radians on load
parameters:
  d0_base_height:  {value: 200.0, status: placeholder, rationale: "To be explored; sweep it"}
  a1_link_upper:   {value: 200.0, status: settled,     rationale: "Equal-length links as starting point"}
  a2_link_forearm: {value: 200.0, status: settled,     rationale: "Equal to upper arm; maximizes dexterous workspace"}
  a3_wrist_pitch_to_yaw: {value: 0.0, status: assumed, rationale: "J3/J4 coincident; revisit when wrist is packaged"}
  a4_yaw_to_roll:  {value: 100.0, status: estimate,    rationale: "THE non-spherical offset (F3); decision-bearing, awaiting ratification"}
  d5_tool_length:  {value: 112.0, status: settled,     rationale: "Hiwonder gripper, flange to grasp point"}
dh:
  convention: classical        # Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha)
  rows:
    - {joint: 0, d: d0_base_height, a: 0,               alpha:  90}
    - {joint: 1, d: 0,              a: a1_link_upper,   alpha:   0}
    - {joint: 2, d: 0,              a: a2_link_forearm, alpha:   0}
    - {joint: 3, d: 0,              a: a3_wrist_pitch_to_yaw, alpha:  90}
    - {joint: 4, d: 0,              a: a4_yaw_to_roll,  alpha: -90}
    - {joint: 5, d: d5_tool_length, a: 0,               alpha:   0}
```

A `null` value loads as 0.0 and emits a warning naming the parameter and its
status. A `suspect` or `open` parameter must never fail silently.

`status` values: `settled`, `estimate`, `assumed`, `placeholder`, `open`,
`suspect`.

The `theta` column is absent from `rows` because it is never a constant: row `i`
takes joint variable `q[i]`. Once a home pose is chosen, a constant zero offset
per joint is added here as `theta_offset` and applied as `q[i] + theta_offset`.
Until then it is identically zero and the distinction does not arise.

**Units.** Lengths are millimetres throughout, config and runtime alike. Angles
are degrees **in the config only**, for readability, and are converted on load;
every angle crossing a Python interface — `q`, the returned frames, anything in
`kinematics.py` — is radians. The conversion happens once, in `config.py`.

## Forward kinematics (`kinematics.py`)

Pure numpy. No file I/O, no plotting, no globals.

- `forward_kinematics(cfg: ArmConfig, q: ndarray[6]) -> Pose` where `Pose`
  exposes `frames` (7 × 4×4: base then one per joint), `joint_origins` (7 × 3),
  `joint_axes` (6 × 3, unit vectors), and `tcp` (4×4).
- Joint `i`'s axis is the z axis of the frame preceding it, per the classical
  convention; the module exposes this mapping rather than leaving callers to
  rederive it. The off-by-one here is the single most likely source of a silent
  error, so it is written once and tested directly.
- `axis_distance(p1, d1, p2, d2)` — minimum distance between two lines, handling
  the parallel case. Used by both the tests and the visualization.

## Verification (`test_kinematics.py`)

Tests are written before the implementation. Four classes:

**1. Pose-independent invariants.** These hold in every configuration and do not
depend on the joint zero offsets, which are still unassigned:

- `|origin(J1) − origin(J2)| == a1_link_upper`, and likewise `a2` for J2→J3.
- J1, J2, J3 axes mutually parallel (cross products vanish) in all poses.
- The J0 axis is world Z in all poses.
- The TCP lies on the J5 axis — the handoff requires this, and violating it
  makes J5 swing the tool through a circle instead of spinning it in place.

**2. Sphericity, pose-invariantly.** The wrist is spherical iff the J3, J4 and J5
axes share a common point. Asserted via the perpendicular distances between
consecutive wrist axes, over a set of sampled poses, not the handoff's
J3↔J5 check (F4). Expected result with the current config: non-spherical, with
the separation equal to `a4_yaw_to_roll`.

**3. Parameter sensitivity.** Each parameter must measurably move the quantity it
is named for; a parameter that provably cannot influence its own quantity is
wrong regardless of its value. This is the check that produced F2, generalized
into a permanent guard.

**4. Independent cross-check.** `roboticstoolbox-python` built from the same
table, compared against the hand-written FK across random poses. Marked
`skipif` on import failure — a dev-only dependency, never imported by the tool.

Known-by-inspection poses (arm straight up, arm horizontal) are deliberately
**not** used as the primary test: they depend on joint zero offsets, which are an
open item. They are added once a home pose is chosen.

## Visualization (`plot_arm.py`)

Matplotlib 3D. Launched directly; no notebook.

- Stick figure through the joint origins, markers at each joint, TCP
  distinguished.
- A short quiver at each joint showing its axis direction — the wrist offset is
  invisible in a top view and cannot be confirmed by looking at a model, so the
  axes are drawn rather than inferred.
- A plane representing the work surface.
- Sliders: the six joint angles, plus `d0_base_height` and `a4_yaw_to_roll`.
  `d0` is the value this work exists to settle; `a4` is included because F3
  makes it the sphericity control, and watching the wrist axes converge as
  `a4 → 0` is the clearest demonstration that the model behaves correctly.
- A readout of the current TCP position and the wrist axis separation.

Equal aspect ratio on all three axes. A 200 mm arm drawn in a stretched box
misleads about reach.

## Deferred

- **Workspace sweep.** Blocked: joint limits are undefined for all six joints,
  and a sweep without them produces a plot that looks authoritative and means
  nothing. Unblocked by the joint-limits open item, not by more code.
- Inverse kinematics (numerical, per the handoff's constraint).
- URDF export.
- The printed XYZ axis indicator as a second tool transform.

## Open items inherited from the handoff

Unresolved; the script treats them as variables, not constants.

- `a4_yaw_to_roll` — magnitude not ratified. 100 mm is an estimate that
  silently decided sphericity; confirm or revise it.
- The L-bracket's second leg (F5) — unrepresented in the table; confirm whether
  `d5_tool_length` absorbing it is correct.
- `d0_base_height` — placeholder; settling it is the point of this work.
- Joint limits — undefined for all six joints. Blocks the workspace sweep.
- Joint zero offsets — unassigned. Pick a home pose, then back out the constants.
- Mounting arrangement — undecided; changes what base height is measured from.
- Work surface dimensions — undefined, so there is no coverage target yet.

## Implementation order

1. Commit this spec on `kinematics-model` (branched off `develop`).
2. Write the implementation plan.
3. Config loader, FK module and tests, built by a `senior-engineer` subagent.
   Acceptance: the full test suite passes, including the sensitivity class.
4. Visualization, built by a `senior-engineer` subagent. Acceptance: the plot
   launches, sliders drive it, and the wrist axis separation readout goes to
   zero as `a4_yaw_to_roll → 0`.
