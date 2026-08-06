# wuji-glove-teleop-mujoco 运行环境
#
# 与 wuji-hand-teleop 的镜像（Ubuntu 22.04 / ROS2 Humble / Python 3.10）不兼容：
# 本仓的 ros2_realhand_node.py 依赖 wuji-sdk（走以太网 zenoh/UDP 连二代手），而
# wuji-sdk 的 retargeting 与二代手支持需 cp312 wheel。ROS2 里 rclpy 必须与 Python
# 版本严格匹配。因此整套栈上移到：
#   Ubuntu 24.04 (Noble) · ROS2 Kilted · Python 3.12
# —— 三者天然配套（Kilted 的 Tier-1 平台就是 24.04，系统 Python 就是 3.12），
# 本镜像即在真机上验证通过的环境（连二代手 → 驱动 → 回发 joint_states ~100Hz）。
#
# 备选：想要 LTS 可把基础镜像换成 ros:jazzy-ros-base（同样 24.04 / Python 3.12，
# 支持周期更长）；节点代码无需改动。
#
# 构建：  docker build -t wuji-teleop:kilted -f docker/Dockerfile .
# 运行：  docker run --rm -it --network host wuji-teleop:kilted
#         └─ 必须 --network host：二代手走以太网(zenoh/UDP)发现 + DDS 都需要主机网络。
#         带 USB 手套/一代手再加：  --device /dev/bus/usb
FROM ros:kilted-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# 系统依赖：
#  - python3-pip：装 pip 依赖（24.04 系统 Python 就是 3.12，与 Kilted 的 rclpy 对齐）
#  - libusb-1.0-0：wuji-sdk USB transport（手套 / 一代手；二代手走以太网不需要）
#  - libgl1/libegl1/libglib2.0-0/ffmpeg：mujoco / opencv / imageio-ffmpeg 渲染与编码
#  - ros-kilted-sensor-msgs / geometry-msgs：节点用到的消息（ros-base 通常已含，显式确保）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip git git-lfs \
        libusb-1.0-0 libgl1 libegl1 libglib2.0-0 ffmpeg \
        ros-kilted-sensor-msgs ros-kilted-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖装进系统 Python 3.12（容器内可 --break-system-packages 绕过 PEP668）。
# 源 setup.bash 后，python3 同时可见 /opt/ros 的 rclpy 与这里装的 wuji-sdk。
#   wuji-sdk：2026.7.21/8.3 起 retargeting 原生化，顶层 from wuji_sdk import
#             HandModel, RetargetSession，无需 [retarget] extra。
RUN pip install --no-cache-dir --break-system-packages \
        wuji-sdk \
        numpy \
        ruckig \
        mujoco imageio imageio-ffmpeg opencv-python

# 放入仓库代码（也可在 run 时用 -v $PWD:/ws 覆盖为本地开发目录）
WORKDIR /ws
COPY . /ws

# 每次进 shell 自动 source ROS2；自检环境
RUN echo 'source /opt/ros/kilted/setup.bash' >> /etc/bash.bashrc \
    && echo 'echo -n "env check: "; python3 -c "import rclpy, wuji_sdk, sensor_msgs.msg; print(\"rclpy + wuji_sdk + sensor_msgs OK (py\"+__import__(\"sys\").version.split()[0]+\")\")"' >> /etc/bash.bashrc

SHELL ["/bin/bash", "-lc"]

# 默认进交互 shell。跑二代手驱动节点示例：
#   source /opt/ros/kilted/setup.bash
#   python3 ros2_realhand_node.py --side right --hand-sn <SN> --prefix "" --diagnostics
CMD ["bash"]
