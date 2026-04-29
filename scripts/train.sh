#!/usr/bin/env bash
# State-based PPO training for the TIAGo lift task.

#/home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Tiago-Lift-v0 --headless

# --- headless full training, VISION (1024 envs) — fresh start ---
/home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Tiago-Lift-Vision-v0 --headless --enable_cameras

#/home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Tiago-Lift-Vision-v0 --headless --enable_cameras --checkpoint logs/skrl/tiago_lift_vision/2026-04-27_23-40-38_ppo_torch/checkpoints/best_agent.pt

# ---  ---
# /home/mohamed/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Tiago-Lift-v0 --num_envs 2
