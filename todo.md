# Midterm Project: Digital Twin & Robot Simulation

**Deadline: 2026-05-11**
**Scene: hospital.usd** (Isaac Sim built-in: Isaac/Environments/Hospital/hospital.usd)
**Robots: Nova Carter (AMR) + Franka Panda (arm)**
**Object: Standard cube**

---

## 1. Scene Setup

- [ ] Load hospital.usd environment in Isaac Sim
- [ ] Import Nova Carter AMR into the scene
- [ ] Import Franka Panda manipulator (mounted on Nova Carter or standalone at goal)
- [ ] Add standard cube object with physics collider
- [ ] Define Point A (pick location) and Point B (place location)
- [ ] Verify physics simulation runs correctly in hospital scene

## 2. Navigation & Obstacle Avoidance (30%)

**Architecture:** three pluggable components selected via `config.yaml`
(see `midterm_project/core/`):

- `Planner` — global path planning (scene-aware, re-planable)
- `Navigator` — AMR base execution + reactive avoidance
- `Manipulator` — arm control with obstacle avoidance

All components read their tuning from `config.yaml` — no hardcoded values.
This makes robot swaps (Robot Diversity +5%) and randomized runs (Section 4)
a one-line config change.

### Global path planning (`RRTStarPlanner`)

- [ ] Define abstract `Planner` interface (`core/planner.py`)
      with `build(world)`, `plan(start_xy, goal_xy)`, `is_valid(xy)`
- [ ] Build a 2D occupancy grid of the hospital once (raycast sampling
      of static geometry) — used only as a fast collision oracle
- [ ] Implement RRT*: sample → nearest → steer → collision-check → **rewire**
- [ ] Apply shortcut-smoothing on the final path
- [ ] **All RRT\* tuning from `config.yaml` → `planner.rrt_star.*`**
      (`max_iter`, `step_size`, `goal_bias`, `rewire_radius`, `goal_tolerance`, `smooth`)
- [ ] Expose `is_valid(xy)` so the Randomizer can reject unreachable points

### AMR base execution (`WaypointNavigator`)

- [ ] Define abstract `Navigator` interface (`core/navigator.py`)
- [ ] Adapt `examples/hand_on_1_amr.py` pattern for Nova Carter
      (uses `WheelBasePoseController` + `DifferentialController`)
- [ ] Consume the Planner's waypoint list; drive through them in sequence
- [ ] **Re-plan on blockage**: if stuck > N ticks or LIDAR detects obstacle
      closer than threshold, call `planner.plan()` from the current pose
- [ ] Verify AMR reaches the goal reliably for randomized (start, goal) pairs

### Arm obstacle avoidance (`FrankaRMPflowManipulator`)

- [ ] Define abstract `Manipulator` interface (`core/manipulator.py`)
- [ ] Implement with RMPflow (adapt `examples/hand_on_3_rmpflow.py`,
      YAML tuning from `configs/rmpflow/franka_rmpflow_common.yaml`)
- [ ] Register static pick/place obstacles (tables, walls near the workspace)
- [ ] Verify arm avoids obstacles for randomized pick/place positions

### Optional — Cortex orchestration (polish / robustness)

- [ ] Implement `CortexOrchestrator` (adapt `examples/hand_on_4_cortex.py`)
      — reactive replanning if the cube is disturbed or dropped
- [ ] Same `Orchestrator.run()` contract as the default async FSM, so swap
      is a one-line change: `orchestrator.type: cortex` in `config.yaml`

### Why NOT pure RMPflow or pure Cortex for the base

- RMPflow in Isaac Sim is **arm-only** — no built-in wheeled-robot variant
- Cortex is an **orchestration layer**, not a motion planner; it still needs
  a concrete Navigator underneath
- Real mobile-manipulation stacks (including NVIDIA's own Nova Carter demos)
  separate **base navigation** from **arm manipulation** — we mirror that

## 3. Arm Operation - Pick & Place (30%)

- [x] Set up Franka Panda with pick-and-place controller (`core.manipulator.FrankaPickPlaceManipulator`)
- [x] Implement grasp logic for the standard cube (via Isaac Sim's `PickPlaceController`)
- [x] Implement place logic at the destination (reachable dropoff near pickup)
- [x] Verify successful grasp of the object (tested: cube lifts to z=0.31 m at t≈500)
- [x] Verify successful placement of the object (cube settles at dropoff within reach)
- [x] Integrate arm operation with AMR navigation (`apps/run_pipeline.py`: nav → pick → place)

**Known limitation — Franka is a separate articulation ("station mode"), NOT
mounted on the AMR.** Isaac Sim's `PickPlaceController` caches the articulation
root pose at construction, so a mobile (pose-synced) Franka's IK plans from the
stale spawn frame. Tried:
  1. Pose-sync callback + `rebase()` before each pick — arm follows but IK
     succeeds only when rebased; pose-sync then needs careful toggling.
  2. `RobotAssembler.assemble_rigid_bodies()` to merge the two articulations —
     strips both ArticulationRootAPIs and uses relative body paths; non-functional
     in Isaac Sim 5.1 runtime context.

For the current green demo, the Franka is parked at `[1.5, 1.0, 0]` (near the
cube). The AMR still drives to Point A for the navigation half of the demo but
the arm is not on it. See `config.yaml` manipulator section for details.

## 4. Domain Randomization (20%)

- [ ] Randomize AMR start position (within valid hospital locations)
- [ ] Randomize AMR goal position
- [ ] Randomize object (cube) spawn location
- [ ] Randomize placement target location
- [ ] Verify the full pipeline works across randomized configurations

## 5. Technical Report & Video (20%)

- [ ] Write system architecture overview
- [ ] Document coordinate setup (hospital scene coordinate frame, robot frames)
- [ ] Take Isaac Sim setup screenshots
- [ ] Document domain randomization implementation
- [ ] Write troubleshooting log
- [ ] Record demo video with >= 3 randomized runs
- [ ] Enable Physics Collider visualization in video
- [ ] Compile report as PDF

## 6. Deliverables Checklist

- [ ] Project Report (PDF)
- [ ] Demo Video (>= 3 randomized runs, collider viz enabled)
- [ ] Code: Python scripts (path planning, control, randomization logic)
- [ ] Assets: scene USD file

---

## Bonus Items (optional)

### 3DGS Scene Reconstruction (+3%)

**Target: A room/lab at school**

#### Step 1: Film the scene with smartphone
- [ ] Pick a well-lit room/lab (avoid reflective surfaces, glass, and moving objects)
- [ ] Film a smooth, slow walkthrough video (1-2 min) covering the full room
  - Walk in a circle around the room, keeping the camera pointed inward
  - Move slowly and steadily — no sudden turns or shaky motion
  - Overlap coverage: every surface should appear in multiple frames from different angles
  - Film at 1080p or 4K, 30fps is fine
  - Keep consistent exposure (lock AE/AF on your phone camera app)
- [ ] Also capture a few extra loops at different heights (waist-level, chest-level)
- [ ] Transfer video to your computer

#### Step 2: Extract frames from video
- [ ] Use ffmpeg to extract frames: `ffmpeg -i video.mp4 -qscale:v 2 -vf "fps=2" frames/%04d.jpg`
  - fps=2 means 2 frames/sec; adjust for density (more frames = better but slower)
  - Aim for 100-300 frames total
- [ ] Delete blurry or redundant frames manually (optional but helps quality)

#### Step 3: Run COLMAP for camera pose estimation
- [ ] Install COLMAP (`sudo apt install colmap` or from source)
- [ ] Run feature extraction: `colmap feature_extractor --database_path db.db --image_path frames/`
- [ ] Run matching: `colmap exhaustive_matcher --database_path db.db`
- [ ] Run sparse reconstruction: `colmap mapper --database_path db.db --image_path frames/ --output_path sparse/`
- [ ] Verify sparse point cloud looks reasonable in COLMAP GUI

#### Step 4: Train 3D Gaussian Splatting model
- [ ] Clone a 3DGS implementation (e.g., `gsplat`, `nerfstudio`, or original `gaussian-splatting`)
- [ ] Convert COLMAP output to the expected format (most tools have a converter script)
- [ ] Train the model (~15-30 min on a decent GPU)
- [ ] Preview the result in the viewer to check quality

#### Step 5: Export and import into Isaac Sim
- [ ] Export the trained model as a mesh (PLY or USD)
  - Most 3DGS tools support mesh extraction via marching cubes or similar
  - Alternatively, export as point cloud and convert to mesh with Open3D/MeshLab
- [ ] Clean up the mesh in MeshLab/Blender if needed (remove floaters, close holes)
- [ ] Convert to USD format (use NVIDIA's `omni.kit` tools or Blender USD exporter)
- [ ] Import the USD into Isaac Sim as a static environment reference
- [ ] Add collision meshes (simplified convex decomposition of the scene mesh)
- [ ] Run the AMR simulation within the reconstructed scene
- [ ] Take screenshots for the report

> **Note:** Reconstruction quality is NOT evaluated — they just want to see the pipeline works.

---

### Grasp 3DGS-Reconstructed Object (+2%)

**Target: Small household item (e.g., mug, bottle, toy)**

#### Step 1: Film the object with smartphone
- [ ] Place the object on a plain, non-reflective surface (table with matte cloth)
- [ ] Film the object by slowly orbiting around it at ~30cm distance
  - Do 2-3 full orbits: one at table-level, one from ~45 degrees above, one from steeper angle
  - Each orbit should take ~20-30 seconds (slow and steady)
  - Total video: ~1-2 min
- [ ] Ensure the entire object is always in frame
- [ ] Good lighting from multiple directions (avoid harsh single-source shadows)

#### Step 2: Extract frames & run COLMAP
- [ ] Extract frames: `ffmpeg -i object_video.mp4 -qscale:v 2 -vf "fps=3" obj_frames/%04d.jpg`
  - Use higher fps=3 since the object is small and needs more coverage
  - Aim for 60-150 frames
- [ ] Run COLMAP pipeline (same as scene: feature_extractor -> exhaustive_matcher -> mapper)
- [ ] Verify sparse reconstruction captures the object shape

#### Step 3: Train 3DGS & extract mesh
- [ ] Train 3DGS model on the object images
- [ ] Export as mesh (PLY/OBJ)
- [ ] Clean up in MeshLab: remove background/table geometry, keep only the object
- [ ] Scale the mesh to match real-world dimensions (measure the real object)

#### Step 4: Prepare for Isaac Sim grasping
- [ ] Convert cleaned mesh to USD format
- [ ] Set up collision proxy: generate a **convex hull** approximation of the mesh
  - In Isaac Sim: right-click the mesh prim -> Physics -> Set Collision -> Convex Hull
  - Or use MeshLab/trimesh to compute convex hull and import as collision mesh
- [ ] Add rigid body physics properties (mass, friction, etc.)
- [ ] Replace the standard cube in your pick-and-place script with the reconstructed object
- [ ] Tune grasp parameters if needed (the convex hull may have different grasp points)
- [ ] Verify: successfully grasp & place the reconstructed object
- [ ] Take screenshots of the collision proxy setup for the report

> **Tip:** Choose an object that is:
> - Graspable by Franka's parallel gripper (3-10 cm wide)
> - Not transparent or highly reflective (bad for 3DGS)
> - Simple-ish shape (mug, small box, fruit, toy block)
> - A mug or small bottle works well — recognizable and easy to reconstruct

---

### Robot Diversity (+5%) — optional

- [ ] Replace standard robot chassis (e.g., quadruped base)
- [ ] Or use custom end-effector / gripper
- [ ] Verify pick & place still works with custom robot

---

## Notes

- Evaluation uses official hospital.usd + Nova Carter + Franka Panda
- Existing repo examples to reference:
  - `examples/hand_on_1_amr.py` - AMR navigation pattern (uses JetBot, adapt for Nova Carter)
  - `examples/hand_on_2_franka.py` - Franka pick-and-place
  - `examples/hand_on_3_rmpflow.py` - RMPflow obstacle avoidance
  - `examples/hand_on_4_cortex.py` - Cortex decider networks
- Scripts run via `run_in_isaac.py` TCP socket to Isaac Sim Docker container
