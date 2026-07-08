#!/usr/bin/env bash
# 方案 B 一键起：retarget 计算节点 + MuJoCo 仿真订阅节点（同一个 ROS2 图）
# 用法: ./run_ros2_teleop.sh [view|video]
set -e
cd "$(dirname "$0")"
source /opt/ros/kilted/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
MODE=${1:-view}
PY=./venv312/bin/python
$PY ros2_retarget_node.py --side right --source replay --rate 120 & PUB=$!
trap "kill $PUB 2>/dev/null" EXIT
sleep 4
$PY ros2_mujoco_hand_node.py --side right --mode "$MODE" --out teleop_ros2.mp4 --frames 600
