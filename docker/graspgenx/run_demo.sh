#!/bin/bash
# Run real GraspGenX inference on bundled sample point clouds, for a
# registered gripper (default: hsrc_hand). Run ON THE HOST, after
# run_wizard.sh has registered the gripper. Opens a browser viewer.
#
# Usage:
#   ./run_demo.sh                        # inference on hsrc_hand, gripper mesh shown
#   ./run_demo.sh --vis-top-grasp-meshes --num-top-grasp-meshes 10   # more than 5
#   SAMPLE_DATA=<dir of *.json> ./run_demo.sh --planner diffusion   # your own clouds:
#       a run's graspgenx/ folder, written by realrobot/visualizations/rgbd_frame_capture.py
#       and by every pick (outputs/detections/<run>/graspgenx/)
# The demo defaults to --planner graspmoe, which adds rule-placed top-down
# candidates; app.py serves diffusion only. Pass --planner diffusion to see
# the grasps pick.py actually gets.
set -e

DATA_DIR=/opt/graspgenx/assets/sample_data/object_pc
MOUNT=()
if [ -n "${SAMPLE_DATA:-}" ]; then
    MOUNT=(-v "$(realpath "$SAMPLE_DATA"):/data:ro")
    DATA_DIR=/data
fi

cd "$(dirname "$0")"
# Repo root as the build context: the Dockerfile COPYs docker/graspgenx/app.py
# and core/utils/zenoh_rpc.py, the same paths compose.yaml builds it with.
docker build -t hrl/graspgenx:latest -f Dockerfile ../..
docker run --gpus all -it --rm \
    --net=host \
    -v "$(pwd)/checkpoints:/opt/graspgenx/ext/graspgenx_checkpoints" \
    -v "$(pwd)/x_grippers:/opt/graspgenx/assets/x_grippers" \
    "${MOUNT[@]}" \
    -e GRASPGENX_GRIPPER_CFG_DIR=/opt/graspgenx/assets \
    hrl/graspgenx:latest \
    uv run python3 scripts/demo_object_pc.py \
        --sample_data_dir "$DATA_DIR" \
        --gripper_name hsrc_hand --plot_top_mesh "$@"
# Watch the terminal output for the viser URL (usually http://localhost:8080).
