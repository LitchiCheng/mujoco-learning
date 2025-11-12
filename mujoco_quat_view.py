import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco.viewer

roll = -np.pi / 2
pitch = 0.0
yaw = 0.0

cr, sr = np.cos(roll), np.sin(roll)
cp, sp = np.cos(pitch), np.sin(pitch)
cy, sy = np.cos(yaw), np.sin(yaw)
R_des = np.array([
    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
    [-sp,   cp*sr,            cp*cr],
])

rot = R.from_matrix(R_des)
quat = rot.as_quat()  # (x, y, z, w)
mujoco_quat = [quat[3], quat[0], quat[1], quat[2]]  # 转换为MuJoCo格式
mujoco_pos = [0.1, 0.1, 0.5]  # 姿态主体位置

xml = f"""
<mujoco model="pose_visualization">
  <worldbody>
    <!-- 地面 -->
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    
    <!-- 姿态主体（应用目标旋转） -->
    <body name="target_pose" pos="{mujoco_pos[0]} {mujoco_pos[1]} {mujoco_pos[2]}" quat="{mujoco_quat[0]} {mujoco_quat[1]} {mujoco_quat[2]} {mujoco_quat[3]}">
      <!-- X轴箭头（红色）：圆柱体（箭杆）+胶囊体（箭头） -->
      <geom name="x_axis杆" type="cylinder" fromto="0 0 0 0.4 0 0" size="0.01" rgba="1 0 0 1"/>  <!-- 杆长0.4，半径0.01 -->
      <geom name="x_axis头" type="capsule" fromto="0.4 0 0 0.5 0 0" size="0.02" rgba="1 0 0 1"/>  <!-- 箭头长0.1，半径0.02（胶囊体模拟箭头） -->
      
      <!-- Y轴箭头（绿色） -->
      <geom name="y_axis杆" type="cylinder" fromto="0 0 0 0 0.4 0" size="0.01" rgba="0 1 0 1"/>
      <geom name="y_axis头" type="capsule" fromto="0 0.4 0 0 0.5 0" size="0.02" rgba="0 1 0 1"/>
      
      <!-- Z轴箭头（蓝色） -->
      <geom name="z_axis杆" type="cylinder" fromto="0 0 0 0 0 0.4" size="0.01" rgba="0 0 1 1"/>
      <geom name="z_axis头" type="capsule" fromto="0 0 0.4 0 0 0.5" size="0.02" rgba="0 0 1 1"/>
    </body>

    <body name="world_frame" pos="0 0 0">
      <geom name="X" type="cylinder" size="0.005" fromto="0 0 0 0.3 0 0" rgba="1 0 0 1" contype="0"
            conaffinity="0"/>
      <geom name="Y" type="cylinder" size="0.005" fromto="0 0 0 0 0.3 0" rgba="0 1 0 1" contype="0"
            conaffinity="0"/>
      <geom name="Z" type="cylinder" size="0.005" fromto="0 0 0 0 0 0.3" rgba="0 0 1 1" contype="0"
            conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# 4. 编译模型并可视化
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()