#!/usr/bin/env bash

# CHECKPOINT="${1:-logs/skrl/tiago_lift/2026-04-27_20-38-57_ppo_torch/checkpoints/best_agent.pt}"
# /home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task Template-Tiago-Lift-Play-v0 --checkpoint "$CHECKPOINT" --num_envs 1 --video --video_length 480

CHECKPOINT="${1:-logs/skrl/tiago_lift_vision/2026-04-28_23-38-36_ppo_torch/checkpoints/best_agent.pt}"
/home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task Template-Tiago-Lift-Vision-Play-v0 --checkpoint "$CHECKPOINT" --num_envs 1 --enable_cameras --video --video_length 480
