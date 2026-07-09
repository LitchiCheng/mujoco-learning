# mujoco-learning

## Installation

### 1. System dependencies (Ubuntu / WSL2)

```bash
# 基础库 & GUI 支持
sudo apt update && sudo apt install -y \
    libgl1-mesa-glx libgl1-mesa-dri mesa-utils \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-sync1 \
    libxcb-xfixes0 libxkbcommon-x11-0 libqt5gui5 libqt5widgets5

# 编译 PyKDL / Pinocchio 所需工具
sudo apt install -y make cmake g++ unzip python3-dev libboost-all-dev
```

### 2. Python 环境

```bash
uv venv && source .venv/bin/activate
uv sync                          # 安装主要依赖 (pyproject.toml + uv.lock)
```

### 3. Pinocchio（可选，用于 MPC / IK）

```bash
bash scripts/install_pinocchio.sh    # 源码编译带 CasADi 的 Pinocchio
```

安装后直接使用即可。

### 4. PyKDL + kdl_parser（可选，用于 URDF 运动学）

```bash
# uv sync 会安装 kdl_parser Python 包
uv sync

# 运行脚本：编译 PyKDL C++ 库 + 自动修复兼容性问题
bash scripts/install_pykdl.sh        # 第一次需要编译，之后会自动检查跳过
```

之后 `import kdl_parser` / `from kdl_parser.urdf import ...` 即可使用。

---

## Running Examples

所有示例脚本已添加自动路径设置，从项目根目录运行：

```bash
# 从项目根目录运行任何示例
python examples/kinematics/ik_kdl_panda.py
python examples/control/pbvs_mpc.py
python examples/robots/panda/panda_viewer.py
python hardware/joystick_so100.py
```

---

## Examples

### 🤖 Kinematics & Path Planning

| File | Description |
|------|-------------|
| `examples/kinematics/ik_kdl_panda.py` | KDL 逆运动学 FK/IK 演示 |
| `examples/kinematics/ik_casadi_panda.py` | CasADi + Pinocchio 逆运动学 (Panda) |
| `examples/kinematics/ik_casadi_ur5e.py` | CasADi + Pinocchio 逆运动学 (UR5e) |
| `examples/kinematics/null_space.py` | 冗余机械臂零空间运动 |
| `examples/kinematics/check_fk_match_with_mujoco.py` | KDL vs Pinocchio FK 一致性验证 |
| `examples/planning/path_plan_ompl_rrtconnect.py` | OMPL RRTConnect 关节空间规划 |
| `examples/planning/rrt_obstacle.py` | OMPL RRT 避障路径规划 |
| `examples/planning/trajectory_plan_toppra.py` | TOPPRA 时间最优轨迹规划 |

### 🎯 Control & Dynamics

| File | Description |
|------|-------------|
| `examples/control/control_joint_pos.py` | 关节空间位置控制 |
| `examples/control/control_ee_with_pinocchio.py` | Pinocchio CLIK 末端姿态控制 |
| `examples/control/pbvs_mpc.py` | PBVS 视觉伺服 + MPC |
| `examples/dynamics/impedance_control.py` | 关节空间阻抗控制 |
| `examples/dynamics/joint_impedance_control.py` | 关节阻抗控制示例 |

### 🦾 Robot-Specific Examples

**Panda / Franka:**
| File | Description |
|------|-------------|
| `examples/robots/panda/panda_viewer.py` | 基础可视化 demo |
| `examples/robots/panda/panda_pbvs.py` | PBVS 基于位置视觉伺服 |
| `examples/robots/panda/panda_dynamics_admittance.py` | 笛卡尔空间导纳控制 |
| `examples/robots/panda/panda_dynamics_drag.py` | 拖动示教 (Drag Teaching) |
| `examples/robots/panda/panda_dynamics_hold.py` | 阻抗力矩保持位置 |
| `examples/dynamics/panda_get_torque.py` | Panda 关节力矩获取 |

**SO-ARM100:**
| File | Description |
|------|-------------|
| `examples/robots/so100/so100_real_control.py` | sim2real 实机控制 |
| `examples/robots/so100/control_ee_with_pinocchio_so100.py` | Pinocchio+CasADi IK (SO-ARM100) |
| `examples/robots/so100/ik_pyroboplan_so_arm100.py` | PyRoboPlan IK (SO-ARM100) |

**Hardware Interface:**
| File | Description |
|------|-------------|
| `hardware/joystick_so100.py` | 手柄遥控仿真末端位姿 |
| `hardware/joystick_sim_and_real_so100.py` | 手柄驱动仿真 + 实机双场景 |

### 🦾 Reinforcement Learning

| File | Description |
|------|-------------|
| `examples/rl/rl_panda_pickup_cube.py` | PPO 强化学习 Pick and Place |
| `examples/rl/rl_panda_obstacle_high_profile.py` | PPO 强化学习避障绕障 |
| `examples/rl/rl_panda_reach_target_high_profile.py` | PPO 末端路径规划到达目标 |

### 👁️ Perception & Sensors

| File | Description |
|------|-------------|
| `examples/perception/test_apriltag.py` | AprilTag 识别与位姿获取 |
| `examples/perception/get_apriltag_pos.py` | SolvePnP 获得 AprilTag 位姿 |
| `examples/perception/camera_calibration.py` | 棋盘格标定相机内参 |
| `examples/perception/sensordata.py` | 传感器数据 (SensorData) 获取 |
| `examples/perception/get_ee_wrench.py` | 末端执行器 Wrench 检测 |
| `examples/perception/contact_detect.py` | 碰撞/接触检测 |

### 📦 Scene Setup & Environment

| File | Description |
|------|-------------|
| `examples/scene_setup/add_random_obstacles.py` | 动态添加随机碰撞障碍物 |
| `examples/scene_setup/add_random_geoms.py` | 动态添加可视化元素（画轨迹/目标点） |
| `examples/scene_setup/move_obstacles.py` | RL 中动态/随机更改障碍物位置 |
| `examples/scene_setup/pickup_cube.py` | 抓取方块（摩擦力参数设置） |
| `examples/scene_setup/move_ball.py` | 键盘控制球体可视化 |
| `examples/scene_setup/mocap_panda.py` | Mocap 动捕接口操控机械臂 |

### 🧪 Utilities & Analysis

| File | Description |
|------|-------------|
| `examples/utilities/get_workspace.py` | 蒙特卡洛采样可达工作空间分析 |
| `examples/utilities/get_body_pos.py` | 获取 body 位置（带可视化） |
| `examples/utilities/mujoco_get_all_body.py` | 获取所有 body name/id/位姿 |
| `examples/utilities/set_and_get_qvel.py` | 关节角速度记录与可视化 |

### 🧪 Tests & Debug

| File | Description |
|------|-------------|
| `tests/test_pinocchio.py` | Pinocchio 安装验证 |
| `tests/test_pyroboplan.py` | PyRoboPlan 库快速测试 |
| `tests/debug/test_joystick.py` | 手柄输入测试 |

### 📁 Conversions & Models

- `conversions/mjcf2usd.py` - MJCF 转 USD 格式
- `model/` - Franka Panda, SO-ARM100 等模型目录

---

## Video Tutorials (Bilibili)

| File | Video |
|------|-------|
| `rl_panda_pickup_cube.py` | [Mujoco RL Pick and Place](https://www.bilibili.com/video/BV1YKTG6GEqj/) |
| `sensordata.py` | [传感器数据获取](https://www.bilibili.com/video/BV1CZovBSEvT/) |
| `pbvs_mpc.py` | [MPC 视觉伺服 PBVS](https://www.bilibili.com/video/BV1UtSoBPEdr/) |
| `pickup_cube.py` | [摩擦力参数设置](https://www.bilibili.com/video/BV1dZAuzLEh2/) |
| `get_apriltag_pos.py` | [SolvePnP AprilTag 位姿](https://www.bilibili.com/video/BV1PTwTzUEq5/) |
| `camera_calibration.py` | [相机内参标定](https://www.bilibili.com/video/BV1eVNMz5Eku/) |
| `test_apriltag.py` | [AprilTag 识别识别](https://www.bilibili.com/video/BV1FgZyBqEpS/) |
| `rl_panda_obstacle_high_profile.py` | [RL 避障绕障](https://www.bilibili.com/video/BV1bd6sBpEvT/) |
| `move_obstacles.py` | [动态障碍物位置控制](https://www.bilibili.com/video/BV1FXzQBjEk1/) |
| `panda_dynamics_admittance.py` | [导纳控制 Admittance](https://www.bilibili.com/video/BV1FhkxBjEvr/) |
| `get_ee_wrench.py` | [末端 Wrench 检测](https://www.bilibili.com/video/BV1ynirB2Ezy/) |
| `panda_dynamics_drag.py` | [拖动示教 Drag Teaching](https://www.bilibili.com/video/BV1k9v6B3EQS/) |
| `panda_dynamics_hold.py` | [逆动力学维持位置](https://www.bilibili.com/video/BV1nVqDBTEma/) |
| `null_space.py` | [零空间 Null Space 运动](https://www.bilibili.com/video/BV1bUmFBiEv1/) |
| `get_workspace.py` | [蒙特卡洛工作空间分析](https://www.bilibili.com/video/BV1tDmFBNE1d/) |
| `add_random_obstacles.py` | [随机添加碰撞物体](https://www.bilibili.com/video/BV1Zs2WBjEyK/) |
| `add_random_geoms.py` | [动态添加可视化元素](https://www.bilibili.com/video/BV16J2WBSE5Z/) |
| `rrt_obstacle.py` | [OMPL RRT 避障规划](https://www.bilibili.com/video/BV1fuSiBdExx/) |
| `check_fk_match_with_mujoco.py` | [KDL vs Pinocchio FK 一致性验证](https://www.bilibili.com/video/BV19xSKBYEoX/) |
| `ik_kdl_panda.py` | [KDL IK/FK末端移动控制](https://www.bilibili.com/video/BV1GKSNBtEvJ/) |
| `urdf_match_with_mjcf.py` | [URDF 与 MJCF 对齐](https://www.bilibili.com/video/BV1HzSFB8EyS/) |
| `panda_pbvs.py` | [PBVS 视觉伺服思路](https://www.bilibili.com/video/BV18zC5BNEt6/) |
| `rl_panda_reach_target_high_profile.py` | [PPO 末端路径规划到达目标](https://www.bilibili.com/video/BV1DAskzmEPZ/) |
| `joystick_sim_and_real_so100.py` | [SO-ARM100 手柄仿真 + 实机双场景](https://www.bilibili.com/video/BV1RCp1zFE2v/) |
| `joystick_so100.py` | [SO-ARM100 手柄遥控仿真](https://www.bilibili.com/video/BV1fyYLzVEbW/) |
| `so100_real_control.py` | [sim2real 实机控制](https://www.bilibili.com/video/BV1gHeHz7ETT/) |
| `control_ee_with_pinocchio_so100.py` | [Pinocchio+CasADi IK (SO-ARM100)](https://www.bilibili.com/video/BV1o38gzSE9h/) |
| `kdl_urdf_test.py` | [不装 ROS 用 PyKDL 解析 URDF](https://www.bilibili.com/video/BV1RWMHzREg4/) |
| `get_torque.py` | [qfrc 三种力与拖拽交互](https://www.bilibili.com/video/BV1kH79zUEAc/) |
| `joint_impedance_control.py` | [关节空间阻抗控制 Impedance](https://www.bilibili.com/video/BV1UK5czMEQr/) |
| `rl_panda.py` | [PPO 强化学习逆运动学 IK](https://www.bilibili.com/video/BV1mHLVzzEMj) |
| `pid_torque_and_get.py` | [PID 力矩控制到达位置](https://www.bilibili.com/video/BV1MbL6zSEAY) |
| `get_body_pos.py` | [末端位置实时追踪 + 可视化](https://www.bilibili.com/video/BV1gaXxYaEnv) |
| `control_joint_pos.py` | [关节空间运动控制](https://www.bilibili.com/video/BV1pWoBYcETJ) |
| `test_pinocchio.py` | [Pinocchio 安装教程](https://www.bilibili.com/video/BV1UFoRYDEfF) |
| `control_ee_with_pinocchio.py` | [Pinocchio+MuJuCo CLIK 闭环控制](https://www.bilibili.com/video/BV1aAZYYAE5f) |
| `move_ball.py` | [键盘控制球体可视化记录](https://www.bilibili.com/video/BV1oTZrYaE2h) |
| `trajectory_plan_toppra.py` | [TOPPRA 时间最优轨迹规划](https://www.bilibili.com/video/BV1fndxYSEui) |
| `path_plan_ompl_rrtconnect.py` | [OMPL Panda RRT 关节空间规划](https://www.bilibili.com/video/BV1EJd5YQExw) |
| `test_pyroboplan.py` | [PyRoboPlan 库测试](https://www.bilibili.com/video/BV1Rod6YHET2) |
| `path_plan_pyroboplan_rrt.py` | [PyRoboPlan RRT 路径规划 + 轨迹优化](https://www.bilibili.com/video/BV1tZo7YjEgd) |
| `path_plan_pyroboplan_rrt_draw_trajectory.py` | [末端轨迹可视化](https://www.bilibili.com/video/BV1B2ocYSE7r) |
| `ik_path_paln_trajectory_pyroboplan.py` | [IK+ 路径规划 + 轨迹优化成功率提升](https://www.bilibili.com/video/BV1qA5EzPEFh) |
| `mocap_panda.py` | [Mocap 动捕操控机械臂](https://www.bilibili.com/video/BV1k651zXEeN) |
| `set_and_get_qvel.py` | [关节角速度记录与可视化](https://www.bilibili.com/video/BV1kSLdznEMd) |
| `get_camera_pic.py` | [相机视角调整获取图片实时显示](https://www.bilibili.com/video/BV1THGSzvE6t) |
| `test_why_continuous_2q.py` | [continuous 关节问题解析](https://www.bilibili.com/video/BV1tvVrzmEgx) |
| `contact_detect.py` | [物体碰撞接触检测](https://www.bilibili.com/video/BV12WfFYYE4T) |

---

## Project Structure

```
mujoco-learning/
├── examples/                    # Main example scripts
│   ├── control/                 # Joint/EE pose control, MPC
│   ├── dynamics/                # Torque, wrench sensing
│   ├── kinematics/              # FK/IK, null space motion
│   ├── perception/              # AprilTag, camera, sensors
│   ├── planning/                # RRT, TOPPRA trajectory
│   ├── rl/                      # PPO reinforcement learning
│   ├── robots/panda/            # Panda-specific demos
│   ├── robots/so100/            # SO-ARM100 examples
│   ├── scene_setup/             # Environment construction
│   └── utilities/               # Workspace analysis, misc tools
├── hardware/                    # Real robot interfaces (joystick)
├── tests/                       # Unit tests and debug scripts
├── conversions/                 # Model conversion utilities  
├── src/                         # Shared modules (mujoco_viewer, kinematics, utils...)
│   ├── mujoco_viewer.py
│   ├── kdl_kinematic.py
│   ├── pinocchio_kinematic.py
│   └── utils.py
├── model/                       # Robot models (MJCF/URDF)
├── scripts/                     # Build/install scripts
│   ├── install_pinocchio.sh
│   └── install_pykdl.sh
├── pyproject.toml
├── uv.lock
└── yaml.py                      # Config helpers
```
---

## Environment Setup Notes (aarch64/ARM64)

本项目的虚拟环境在激活时会自动配置以下环境变量：

```bash
source .venv/bin/activate
# 自动设置:
# - QT_QPA_PLATFORM=offscreen (pyqtgraph 无头模式运行)  
# - LD_LIBRARY_PATH (PyKDL / Pinocchio 动态库路径)
```

**使用方式：**
```bash
# 每次使用时先激活环境（环境变量会自动加载）
source .venv/bin/activate

# 然后直接运行脚本
python examples/kinematics/ik_kdl_panda.py
python examples/control/pbvs_mpc.py   # pyqtgraph 可在无头模式工作
```

---
