# mujoco-learning

## Related package

```
# uv 
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# install PyKDL
bash install_pykdl.sh

# install pinocchio
bash install_pinocchio.sh

```

## Tutorials
|file|url|
|----|-------|
|rl_panda_obstacle_high_profile.py|[Mujoco 开源机械臂 RL 强化学习避障、绕障](https://www.bilibili.com/video/BV1bd6sBpEvT/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|move_obstacles.py|[Mujoco 如何在RL中动态或随机更改障碍物位置、开关碰撞计算以及碰撞信息](https://www.bilibili.com/video/BV1FXzQBjEk1/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|panda_dynamics_admittance.py|[Mujoco 末端（笛卡尔空间）导纳控制（Admittance）仿真及代码讲解](https://www.bilibili.com/video/BV1FhkxBjEvr/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|get_ee_wrench.py|[Mujoco 仿真 Dynamics 动力学获得末端执行器 Wrench](https://www.bilibili.com/video/BV1ynirB2Ezy/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|panda_dynamics_drag.py|[Mujoco 仿真动力学拖动示教(Drag Teaching)](https://www.bilibili.com/video/BV1k9v6B3EQS/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|panda_dynamics_hold.py|[Mujoco 使用 Pinocchio 进行逆动力学及阻抗力矩控制维持当前位置](https://www.bilibili.com/video/BV1nVqDBTEma/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|null_space.py|[Mujoco 冗余机械臂零空间 Null Space 运动仿真](https://www.bilibili.com/video/BV1bUmFBiEv1/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|get_workspace.py|[Mujoco 蒙特卡洛采样统计机械臂可达工作空间（非Matlab）](https://www.bilibili.com/video/BV1tDmFBNE1d/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|add_random_obstacle.py|[Mujoco 随机添加碰撞物体（位置、类型、大小、颜色）以及注意点](https://www.bilibili.com/video/BV1Zs2WBjEyK/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|add_random_geoms.py|[Mujoco 动态添加可视化元素（可用于画轨迹或目标点）](https://www.bilibili.com/video/BV16J2WBSE5Z/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|rrt_obstacle.py|[Mujoco 机械臂 OMPL 进行 RRT 关节空间路径规划避障、绕障](https://www.bilibili.com/video/BV1fuSiBdExx/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|check_fk_match_with_mujoco.py|[Mujoco 检验 KDL 和 Pinocchio 运动学 FK 是否一致](https://www.bilibili.com/video/BV19xSKBYEoX/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|ik_kdl_panda.py|[Mujoco 使用 KDL 对机械臂进行 IK 和 FK 运动学控制末端移动](https://www.bilibili.com/video/BV1GKSNBtEvJ/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|mujoco_get_all_bady.py|[Mujoco 基础：获取模型中所有 body 的 name, id 以及位姿](https://www.bilibili.com/video/BV16cSFBxEYy/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|urdf_match_with_mjcf.py|[Mujoco 模型 MJCF 和 URDF 如何手动对齐（Pinocchio验证）](https://www.bilibili.com/video/BV1HzSFB8EyS/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|panda_pbvs.py|[Mujoco 机械臂进行 PBVS 基于位置的视觉伺服思路](https://www.bilibili.com/video/BV18zC5BNEt6/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|rl_panda_reach_target_high_profile.py|[Mujoco 仿真 PPO 强化学习机械臂末端路径规划到达指定位置](https://www.bilibili.com/video/BV1DAskzmEPZ/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|joystick_sim_and_real_so100.py|[SO-ARM100 双场景演示：手柄驱动 Mujoco 仿真 + 实机控制](https://www.bilibili.com/video/BV1RCp1zFE2v/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|joystick_so100.py|[北通手柄遥控 + Mujoco 仿真 SO-ARM100 机械臂末端位姿](https://www.bilibili.com/video/BV1fyYLzVEbW/?share_source=copy_web&vd_source=98d79df50a14f07106c58e9e50f70c68)|
|so100_real_control.py|[sim2real！so-arm100 机械臂 Mujoco 仿真与实机控制](https://www.bilibili.com/video/BV1gHeHz7ETT/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|control_ee_with_pinocchio_so100.py|[Pinocchio 结合 CasADi 进行 IK 逆运动学及 Mujoco 仿真](https://www.bilibili.com/video/BV1o38gzSE9h/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|install_pinocchio.sh|[Pinocchio 导入 CasADi 失败？源码编译保姆级教程，一步到位解决！](https://www.bilibili.com/video/BV1mSghz1EUx/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|kdl_urdf_test.py|[不装 ROS 也能用 PyKDL！使用kdl_parser解析URDF并进行IK](https://www.bilibili.com/video/BV1RWMHzREg4/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|get_torque.py|[MuJoCo 解析 qfrc 三种力！带你测试鼠标拖拽物理交互效果](https://www.bilibili.com/video/BV1kH79zUEAc/?vd_source=5ba34935b7845cd15c65ef62c64ba82f)|
|joint_impedance_control.py|[MuJoCo 机械臂关节空间阻抗控制Impedance实现（附代码）](https://www.bilibili.com/video/BV1UK5czMEQr/?vd_source=5ba34935b7845cd15c65ef62c64ba82f#reply262516173552)|
|rl_panda.py|[MuJoCo 机械臂 PPO 强化学习逆向运动学（IK）](https://www.bilibili.com/video/BV1mHLVzzEMj?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|pid_torque_and_get.py|[MuJoCo 机械臂 PID 控制器输出力矩控制到达指定位置（附代码）](https://www.bilibili.com/video/BV1MbL6zSEAY?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|get_body_pos.py|[MuJoCo 仿真 Panda 机械臂！末端位置实时追踪 + 可视化（含缩放交互）](https://www.bilibili.com/video/BV1gaXxYaEnv?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|control_joint_pos.py|[MuJoCo 仿真 Panda 机械臂关节空间运动｜含完整代码](https://www.bilibili.com/video/BV1pWoBYcETJ?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|test_pinocchio.py|[Pinocchio 安装教程｜机器人学的必备库](https://www.bilibili.com/video/BV1UFoRYDEfF?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|control_ee_with_pinocchio.py|[【逆解机械臂】Pinocchio+MuJuCo 仿真 CLIK 闭环控制！附代码](https://www.bilibili.com/video/BV1aAZYYAE5f?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|move_ball.py|[MuJoCo 可视化键盘控制球体及位姿实时记录，附代码！](https://www.bilibili.com/video/BV1oTZrYaE2h?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|trajectory_plan_toppra.py|[MuJoCo 仿真 + TOPPRA 最优时间轨迹规划！机械臂运动效率拉满（附代码）](https://www.bilibili.com/video/BV1fndxYSEui?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|path_plan_ompl_rrtconnect.py|[MuJoCo + OMPL 进行Panda机械臂关节空间的RRT路径规划](https://www.bilibili.com/video/BV1EJd5YQExw?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|test_pyroboplan.py|[PyRoboPlan 库，给 panda 机械臂微分 IK 上大分，关节限位、碰撞全不怕](https://www.bilibili.com/video/BV1Rod6YHET2?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|path_plan_pyroboplan_rrt.py|[MuJoCo 机械臂关节路径规划+轨迹优化+末端轨迹可视化（附代码）](https://www.bilibili.com/video/BV1tZo7YjEgd?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|path_plan_pyroboplan_rrt_draw_trajectory.py|[MuJoCo 画出机械臂末端轨迹进行可视化（附代码）](https://www.bilibili.com/video/BV1B2ocYSE7r?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|ik_path_paln_trajectory_pyroboplan.py|[MuJoCo 提高机械臂笛卡尔空间IK+路径规划+轨迹优化的成功率及效率](https://www.bilibili.com/video/BV1qA5EzPEFh?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|mocap_panda.py|[MuJoCo 动捕接口 Mocap 直接操控机械臂（附代码）](https://www.bilibili.com/video/BV1k651zXEeN?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|set_and_get_qvel.py|[MuJoCo 关节角速度记录与可视化，监控机械臂运动状态](https://www.bilibili.com/video/BV1kSLdznEMd?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|get_camera_pic.py|[MuJoCo 相机图片怎么拿？视角调整获取物体图片及实时显示（附代码）](https://www.bilibili.com/video/BV1THGSzvE6t?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|test_why_continuous_2q.py|[Pinocchio导入URDF关节为continuous的问题及详细解释](https://www.bilibili.com/video/BV1tvVrzmEgx?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
|contact_detect.py|[MuJoCo 机械臂物体碰撞、接触检测方式一](https://www.bilibili.com/video/BV12WfFYYE4T?vd_source=5ba34935b7845cd15c65ef62c64ba82f&spm_id_from=333.788.videopod.sections)|
