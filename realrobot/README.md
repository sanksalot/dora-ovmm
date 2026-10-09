# `realrobot/`

Everything run against the HSR lab bag of 2026-08-11.

Grouped by what each script talks to: the bag, the recording, or the robot.

| Folder | Talks to | What it does |
| --- | --- | --- |
| [`dataprep/`](dataprep/) | the bag | bag → Nav2 map → keyframes → Boxer boxes → scene graph. Steps 1–8 and their checks: [dataprep/realrobot_dataprep.md](dataprep/realrobot_dataprep.md) |
| [`offline/`](offline/) | the recording | the finished map, graph and keyframes, no robot needed: path overlays, footprint checks, next-best-view replay |
| [`live/`](live/) | the real HSR | seed its localizer so Nav2 can drive; drive to a furniture piece; check what a grasp needs before anything moves |
| [`visualizations/`](visualizations/) | the real HSR | one RGB-D frame, SAM3's mask and GraspGenX's best grasps drawn on the frame and on the cloud, each a file for slides: `rgbd_frame_capture.py`. `view_grasps.py` turns the cloud and hands round in Open3D (on a machine with a display); the run's `graspgenx/` folder opens in GraspGenX's own viewer via `docker/graspgenx/run_demo.sh`. `rgbd_recorder.py` records the RGB and depth streams to `rgb.mp4` and `depth.mp4`. `scene_graph_figure.py` draws the scene graph as stacked map, object, furniture and room layers; `scene_graph_3d.py` draws it as boxes in the fused cloud, or opens it in Open3D with `--view` |

Navigation on the robot -- the run sequence, every parameter changed and why,
and each failure diagnosed so far: **[realrobot_nav_setup.md](realrobot_nav_setup.md)**.

---

## `realrobot/live/`

| File | Does |
| --- | --- |
| `localize.py` | seeds the robot's localizer by matching the live `/scan` to the map, so Nav2 can plan. Run it **before** Nav2 |
| `goto.py` | drive to one scene-graph furniture piece, named outright: the mission's navigation leg alone |
| `grasp_preflight.py` | checks the robot, vision transport RX and the model servers answer, before a pick moves anything |

Bring-up, the measurements and the failure modes:
[launch/realrobot/navigation.readme](../launch/realrobot/navigation.readme).

---

## Drive to one piece of furniture

The whole chain from a fresh scan to the base moving, no reasoner and no
DeepSeek anywhere in it. Chair (107) is the worked example: it stands in the
west room, 16.3 m from the desks, so the drive crosses the corridor.

### 1. Build the graph · GPU host, ~20 min

```bash
bash docker/boxer/build_scene_graph.sh
```

Steps 1-3 of [docker/boxer/README.md](../docker/boxer/README.md): keyframes →
3D boxes → `config/realrobot/scene_graph/lab_20260811.json`, in the map frame.
Skip it if the graph is already built.

A different recording? `export RECORDING=<name>` moves every path here and in
every other real-robot tool ([recording.py](../core/utils/recording.py)); the
`--map`, `--graph` and `--keyframes` flags still override it. Starting from a
bag, `bash docker/boxer/build_scene_graph.sh --bag bags/<name> --rooms <name>`
runs data prep and the graph in one go
([data prep](dataprep/realrobot_dataprep.md#all-of-it-in-one-command)).

### 2. Look at what it found · anywhere, no ROS

```bash
python3 realrobot/offline/plan_overlay.py --survey
```

Every furniture footprint and floor object on the map, labelled with its node
ID — which is how you pick the ID for step 3. Read this first whenever a query
reports *no observation poses*: it shows whether the piece is where you expect
and whether the laser ever mapped the floor around it.

With rooms assigned, each footprint is coloured by its room and the room names
are written on the map. That is how you check a room division: each room should
be one contiguous patch of colour, and a piece in one room's colour sitting
inside another's patch is a real error. See
[the Boxer README](../docker/boxer/README.md#rooms-come-from-the-map-not-the-labels).

### 3. Check the path before the robot moves · needs ROS, no robot

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
python3 realrobot/offline/plan_overlay.py --furniture 107 --start 2.0 -1.5
```

`--start` is roughly where the robot stands; without it a random free cell is
drawn. Real `planner_server`, real Theta*, on the map with the graph stamped in.

- [ ] Prints a goal and a path length, and writes `outputs/realrobot/lab_20260811/nav/chair_107.png`.
- Nothing usable? The `[blocked]` table names the reason per pose. Try
  `--clearance 0.25` or `--no-blockers`.

### 4. Seed the localizer · once per boot, before Nav2

```bash
python3 realrobot/live/localize.py
```

`laser_2d_localizer` boots at the map origin, 35 cm off the walls, and every
plan then aborts with *"Either of the start or goal pose are an obstacle"*.

Search the whole map, as above, unless you have a reason not to. `--near X Y`
restricts the match to a guess, which disambiguates a symmetric corridor but
will happily seed a **wrong** pose if the guess itself is wrong -- and a wrong
seed is worse than none, because everything downstream then agrees with it.
`--dry-run` reports without publishing.

- [ ] `/laser_2d_localizer/score` reads about **-0.01**, not -0.07.
- [ ] median beam-to-wall in single-figure cm, most beams within 10 cm.
- [ ] The line comparing the match against what the stack already believes
      reads a few centimetres, not metres.

### 5. Nav2 on top of the robot's stack

```bash
ros2 launch /home/ws/launch/realrobot/nav2_real.launch.py
```

No AMCL: the robot publishes `map->odom` itself and the saved map is in that
frame. See [core/navigation/README.md](../core/navigation/README.md#real-robot).

With Toyota's `navigation.py` stopped, two arguments become necessary rather
than optional -- AMCL takes the free `map->odom`, and Nav2 must reach the wheels
itself because the velocity relays stopped with that launch:

```bash
ros2 launch /home/ws/launch/realrobot/nav2_real.launch.py \
  localization:=true cmd_vel_topic:=/omni_base_controller/cmd_vel
```

Why each, and what the second one costs:
[realrobot_nav_setup.md](realrobot_nav_setup.md).

### 6. Drive

```bash
python3 realrobot/live/goto.py --furniture 107 --dry-run   # poses and paths only
python3 realrobot/live/goto.py --furniture 107             # actually drives
```

`--furniture` takes a node ID, an instance name or a label, and must match
exactly one piece. `actions.plan` rings the piece with observation poses, drops
the ones Nav2's live costmap calls blocked, orders them nearest-first, and the
base drives to the first one `compute_path_to_pose` can reach — the same
sequence `mission_tree` runs, minus the query and the grasp. `--look` turns the
head towards the piece on arrival.

Nothing here moves the arm, so Ctrl-C is safe: it cancels the Nav2 goal and the
base stops.

### Every pose says *no path*

Theta* refuses a plan whose **start** is an obstacle, and `planner_server` then
logs *"Either of the start or goal pose are an obstacle!"* for every goal, however
clear the goal is. So a base parked inside the inflation of a desk cannot plan
anywhere, and the symptom looks like the target is unreachable when it is not.

`goto.py` checks the base's own cell against the live global costmap and says so
before it lists the poses. With `robot_radius: 0.3` and `inflation_radius: 0.35`
in [nav2_params_real.yaml](../config/realrobot/nav2/nav2_params_real.yaml), the
gap between two desks in the lab is not much wider than the inflation, so this is
easy to hit. In order:

1. **A wrong pose first, always.** A mis-localised base paints `/scan` onto the
   map somewhere the robot is not, and those obstacles land exactly where the
   stack believes the robot is standing -- so the start cell is blocked by the
   robot's own laser. Clearing the costmap does nothing, because the next sweep
   repaints it.

   ```bash
   python3 realrobot/live/localize.py --dry-run
   ```

   It prints the scan match, then how far that is from the pose the stack is
   already acting on. **Metres apart with a good match means this is your bug**,
   and no amount of costmap clearing or target-picking will help. Publish the
   match (drop `--dry-run`), *then* clear the costmap, in that order.

2. **Stale marks.** With localization sound, the obstacle layer can still hold
   what `/scan` saw earlier:

   ```bash
   ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap {}
   ```

   Real obstacles come straight back; an old pose's smear does not. Not while
   the base is mid-drive.

3. **Genuinely wedged.** `goto.py` prints the nearest free cell and its bearing.
   Drive the base there by hand, with the robot's own teleop, and rerun.

A failed `drive_to` can leave the base a few centimetres further in than it
started, which is why a run that planned once can refuse every pose on the next
try.

### The local and global costmaps pull apart

The local costmap lives in **odom**, the global one in **map**, and the only
thing holding them together is the `map->odom` correction. On this robot that
comes from Toyota's `laser_2d_localizer`; `nav2_real.launch.py` starts no AMCL
(`localization` defaults to false) precisely so there is exactly one publisher
of it. If that localizer stops correcting, `map->odom` freezes and the two
costmaps separate by however far the wheels have drifted since.

Measure it rather than guess:

```bash
python3 realrobot/live/localize.py --watch
```

Every 2 s it scores the live `/scan` against the map **at the pose the stack
believes**, and prints `map->odom` beside it. It searches nothing and publishes
nothing, so it is safe to leave running in a second terminal during a drive.

```text
  time        believed x,y         map->odom   median   <10cm
     0      1.732,   0.011     0.221, -0.147     6 cm     74%
    12      0.884,  -1.402     0.221, -0.147    19 cm     38%
    24     -0.310,  -2.688     0.221, -0.147    41 cm     11%
```

A median climbing as the base moves while `map->odom` never changes is the
localizer not correcting: the pose is pure odometry from the last correction on,
and it drifts without limit. A median that stays in single-figure centimetres is
localization doing its job, and the problem is elsewhere.

Who else might be publishing:

```bash
ros2 node list | grep -i local
ros2 run tf2_ros tf2_echo map odom          # small corrections, not frozen, not jumping
```

Two publishers of `map->odom` -- AMCL from `localization:=true` alongside the
robot's own -- make the base jump between two answers rather than drift from
one. Only run AMCL with Toyota's localization stopped.

### It drove partway and stopped

Not a connection loss, if `goto.py` printed a start pose and returned rather
than hanging: `robot_xy()` rejects TF older than 2 s and raises, and `drive_to`
waits the full `--timeout` before giving up. A drive that reports back promptly
is Nav2 deciding to stop, not the link dying. To rule the link out anyway:
`python3 realrobot/live/dds_check.py` and `ros2 topic hz /scan`.

What usually stops it, from
[nav2_params_real.yaml](../config/realrobot/nav2/nav2_params_real.yaml):

| Setting | Value | What it does |
| --- | --- | --- |
| `SimpleProgressChecker` | 0.5 m / 10 s | Under half a metre in ten seconds and the controller aborts |
| `behavior_plugins` | `["spin", "wait"]` | **No backup.** The rear laser sector is unsensed, so a base wedged nose-first cannot reverse out by itself |
| `robot_radius` / `inflation_radius` | 0.3 / 0.35 | A doorway or a desk gap can be narrower than MPPI will thread |

So the usual sequence is: MPPI slows in a tight gap, the progress checker fires,
the BT spins and waits, and the goal aborts with the base parked in the gap —
where its own cell is inside the inflation, so every following plan fails too.

`goto.py` prints where it stopped, how far it got, whether that cell is blocked
and which way the nearest free one lies. The reason itself is only in the
`controller_server` log, in the Nav2 terminal.

Getting it out: teleop, since no recovery behaviour reverses. Then rerun.

---

## `realrobot/offline/`

`plan_overlay.py` and `footprint_check.py` read the two artefacts step 1 and step 7 leave behind,
`config/realrobot/map/lab_20260811/map.{pgm,yaml}` and
`config/realrobot/scene_graph/lab_20260811.json`, and need nothing else running.

| File | Does |
| --- | --- |
| `plan_overlay.py` | query a target in the scene graph, ask Nav2 for a path to it, draw the path on the map. `--survey` draws the graph alone |
| `footprint_check.py` | measure the scene graph's footprints against the laser map |
| `planner.launch.py` | `map_server` + `planner_server` alone, no robot. `plan_overlay.py` starts it itself |
| `lab_planner.yaml` | their parameters, taken from `config/nav2/nav2_params.yaml` minus the obstacle layer |
| `nbv_replay.py` | replay recorded keyframes through the next-best-view policy and score what it would have chosen |
| `target_tsdf.py` | keyframes + a text prompt → a TSDF cube around the detected target |

`plan_overlay.py` needs ROS sourced; `footprint_check.py` does not, and neither
does `plan_overlay.py --survey`, which skips the query and the planner and just
draws every furniture footprint and floor object on the map, labelled and
coloured by room:

```bash
python3 realrobot/offline/plan_overlay.py --survey
```

That is the picture to read first when a query says *no observation poses*: it
shows whether the piece is where you expect and whether the laser ever mapped
the floor around it.

```bash
cd /home/ws
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

### Commands run on 2026-09-16, and what each one wrote

Every figure below is reproducible from the command next to it. Each run takes
about 40 s, most of it Nav2 coming up.

| # | Command | Generates |
| --- | --- | --- |
| 1 | `python3 realrobot/offline/footprint_check.py` | stdout only |
| 2 | `python3 realrobot/offline/plan_overlay.py --target laptop --seed 2` | `outputs/realrobot/lab_20260811/nav/laptop.png` · `nav/lab_20260811_blockers.{pgm,yaml}` |
| 3 | `python3 realrobot/offline/plan_overlay.py --furniture 8 --seed 4 --no-blockers --output outputs/realrobot/lab_20260811/nav/desk_8_laser_only.png` | `outputs/realrobot/lab_20260811/nav/desk_8_laser_only.png` |
| 4 | `python3 realrobot/offline/plan_overlay.py --furniture 4 --start -0.45 -4.7 --no-blockers --output outputs/realrobot/lab_20260811/nav/desk_4_laser_only.png` | `outputs/realrobot/lab_20260811/nav/desk_4_laser_only.png` |
| 5 | `python3 realrobot/offline/plan_overlay.py --furniture 4 --start -0.45 -4.7 --output outputs/realrobot/lab_20260811/nav/desk_4_with_blockers.png` | `outputs/realrobot/lab_20260811/nav/desk_4_with_blockers.png` |
| 6 | `python3 realrobot/offline/plan_overlay.py --furniture 8 --seed 4` | nothing: prints why every pose around desk (8) is unusable |

- 4 and 5 are the same start and the same target on the two maps, so the paths can be compared. 3 and 6 are the same query, and only the laser-map run has anywhere to stand.
- `--seed` is reproducible for a fixed map, `--clearance` and `--blockers`; change any of them and the same seed picks a different start, because the set of free cells it draws from changes.
- `nav/lab_20260811_blockers.{pgm,yaml}` is rewritten by every `--blockers` run. It is a derived file under `outputs/`, not a map to drive on.

### Options

| Option | Default | Use it to |
| --- | --- | --- |
| `--target <object>` | — | query an object: `laptop`, `coffee maker`, `trash can` |
| `--furniture <name\|id>` | — | query a piece of furniture; the name has to be unique, so `8` rather than `desk` |
| `--start X Y` | random | fix the start instead of drawing one |
| `--seed N` | `0` | pick a different random start |
| `--min-distance M` | `4.0` | how far the random start must be from the target |
| `--attempts N` | `6` | random starts to try before giving up |
| `--clearance M` | `0.3` | free space a pose needs; the `global_costmap` `robot_radius` |
| `--no-blockers` | off | plan on the laser map alone, without the scene graph stamped in |
| `--no-sight-line` | off | keep poses with a wall between them and the target |
| `--full-map` | off | draw all 50 × 28 m instead of the area around the path |
| `--graph`, `--map`, `--output`, `--domain-id` | lab bag, `91` | point at another run |

### How it works

1. **Resolve the query.** `core.pipeline.actions.targets` turns `--target`/`--furniture` into a `Location`. An object already in the graph is answered from memory, so no `DEEPSEEK_API_KEY` is needed.
2. **Turn it into a goal.** `core.pipeline.actions.plan` rings the furniture with observation poses and orders them — the same call `mission_tree` drives. The first pose that survives the filters becomes the Nav2 goal, so the picture shows the leg the mission would ask for, not a straight line to the object.
3. **Plan.** `ComputePathToPose` with `use_start`, because there is no robot and no localisation. A path that stops more than 10 cm short of the goal counts as a refusal, the same check `core.navigation.nav2_client.reachable` makes.
4. **Draw.** The laser map underneath, the scene graph over it, the path on top.

`planner.launch.py` parks a static `map -> base_footprint` transform at the origin only because `Costmap2DROS` will not activate without one. Nothing reads that pose. The stack runs on `ROS_DOMAIN_ID=91` so a sim or a mission on the devcontainer's domain 5 cannot answer these goals.

### What the figures show

**Bounding boxes.** Run 1 measures every furniture footprint against the laser map:

```
11 furniture pieces, mean occupied 10%, free 80%, unknown 10%
walls within 6 m of the furniture run at 22° mod 90 (664 fitted cells);
8 of 11 footprint yaws sit within 10° of that, median 16°
```

- **Placement is good.** Ten of eleven pieces touch an occupied cell (`edge` 0.00); only chair (6) floats, 18 cm off. Two independent sensors — RGB-D at head height and a laser at 19 cm — agree on where things are.
- **Orientation is good.** The room is not square to the map; it is turned about 22°. The footprint fit recovered that from the point clouds alone, and 8 of 11 pieces land within 10° of it. The two chairs that do not are round-ish, where a fitted rectangle has no strong angle to find.
- **80% of the furniture area is free floor in the PGM.** That is the point of stamping them in: the laser sees a desk as four thin legs, so a base planning on the laser map alone drives through the middle of one.
- **Objects standing on the floor are a different story.** All three (trash can, bin, computer tower) span the 19 cm laser plane, and the map already has 35% of their area occupied. Stamping them in sharpens a box the laser saw one side of; it does not add something new.

**Where the scene graph changes the path.** Runs 4 and 5, same start and goal: 5.88 m on the laser map, 5.91 m with the graph stamped in. Around desk (4) the two maps barely differ, because the wall behind it already forces the same route. The graph earns its place on the poses it rules out, not on the metres it adds.

**A goal the geometry gets wrong.** Run 6, desk (8): `standoff.candidates` rings a piece at 0.8 m and filters only against other furniture boxes and free cells. Desk (8) stands against a wall and its fitted box overhangs it by about 0.3 m, so most of the ring lands in the corridor on the far side — free cells that Nav2 happily reaches, from which the robot would be looking at the desk through a wall. `plan_overlay.py` adds a sight-line test on the occupancy grid that `core/` does not have, which drops those poses. What is left:

| Map | Poses clear of the map | Of those, in sight of desk (8) | Result |
| --- | --- | --- | --- |
| laser only (run 3) | 8 | 1, at (1.68, −1.63) | robot enters the room and parks north of the desk |
| scene graph stamped in (run 6) | 7 | 0 | nothing to drive to |

The one good pose has 0.47 m of clearance on the laser map and 0.29 m once office chair (11) and desk (8) are stamped in — a centimetre under the 0.30 m the base needs. So with the graph as blockers, desk (8) is genuinely unreachable for viewing, and run 6 says so instead of parking in the corridor. Rooms would not have fixed this on their own: both the good pose and the bad ones are in the same room as the desk. Two things would: a sight-line or same-enclosure test (this script's version is the crude one), and a tighter box on desk (8).

### Known limits

- The sight line is a ray to one aim point on the target, not a visibility test over the whole surface, and it ignores the head's field of view and tilt.
- `SKIN = 0.15 m` around the target decides which occupied cells are the target's own laser return. Furniture whose box is off by more than that reads its own wall as an obstruction.
- Unknown cells count as blocked, matching `global_costmap`'s `track_unknown_space`. 86% of this map is unknown, so a target in an unscanned corner has nowhere to stand: bookshelf (40) fails this way, and the script prints the breakdown.
- `allow_unknown: false` in `lab_planner.yaml` differs from the simulation stack, which allows it. A path through unmapped space would not mean anything here.
- The stack has no obstacle layer, because there is no live `/scan`. The saved map is all it knows.
