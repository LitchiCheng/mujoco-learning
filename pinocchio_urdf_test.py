import pinocchio as pin
import numpy as np

# 加载 URDF 文件
urdf_path = "model/franka_panda_urdf/robots/panda_arm.urdf"
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# 计算正运动学
q = np.zeros(model.nq)
for i in range(model.nq):
    q[i] = 0.1
try:
    frame_id = model.getFrameId("panda_link7") 
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # pin.ReferenceFrame.LOCAL              # 表达在末端局部坐标系（body frame）
    # pin.ReferenceFrame.WORLD              # 表达在世界/基坐标系（world frame）
    # pin.ReferenceFrame.LOCAL_WORLD_ALIGNED # 原点在末端，但方向与世界对齐
    J = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    print(J.round(3))
except Exception as e:
    print(e)

qvel = []
for i in range(model.nq):
    qvel.append(0.2)
ee_vel = J @ qvel
print("末端速度:", ee_vel.round(3))

# 打印frame_id对应的位姿
print(data.oMf[frame_id])
for i in range(model.nq):
    print("panda_link"+str(i+1))
    print(data.oMf[model.getFrameId("panda_link" + str(i+1))])



