from pathlib import Path
from sys import argv
 
import pinocchio as pin
import numpy as np
 
# Load the urdf model
# model = pin.RobotWrapper.BuildFromMJCF("model/trs_so_arm100/so_arm100.xml").model
model = pin.RobotWrapper.BuildFromMJCF("model/franka_emika_panda/panda_tau.xml").model
# model = pin.buildModelFromUrdf("model/so_arm100_description/so100.urdf")
print("model name: " + model.name)
print("lowerLimits: " + str(model.lowerPositionLimit))
print("upperLimits: " + str(model.upperPositionLimit))
 
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration
q = pin.randomConfiguration(model)
print(f"q: {q.T}")
 
# Perform the forward kinematics over the kinematic tree
pin.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

print("=== 从URDF解析的连杆惯性参数 ===")
for joint_id in range(model.njoints):
    joint_name = model.names[joint_id]
    inertia = model.inertias[joint_id]  # 惯性参数对象

    # 只打印有质量的连杆（忽略固定关节/空连杆）
    if inertia.mass > 1e-6:  
        print(f"\n连杆 {joint_name} (ID: {joint_id}):")
        print(f"  质量: {inertia.mass:.3f} kg")
        print(f"  质心位置（局部坐标系）: {inertia.lever.T} m")
        print(f"  转动惯量矩阵（局部坐标系）:\n{inertia.inertia}")

# ====================== 提取URDF中的惯性参数（验证加载结果） ======================
print("=== 从URDF解析的连杆惯性参数 ===")
for joint_id in range(model.njoints):
    joint_name = model.names[joint_id]
    inertia = model.inertias[joint_id]  # 惯性参数对象

    # 只打印有质量的连杆（忽略固定关节/空连杆）
    if inertia.mass > 1e-6:  
        print(f"\n连杆 {joint_name} (ID: {joint_id}):")
        print(f"  质量: {inertia.mass:.3f} kg")
        print(f"  质心位置（局部坐标系）: {inertia.lever.T} m")
        print(f"  转动惯量矩阵（局部坐标系）:\n{inertia.inertia}")

# 定义机器人状态（关节角度q、速度v、加速度a）
q = pin.neutral(model)  # 中性位形（初始关节角度）
v = np.zeros(model.nv)  # 关节速度（nv为机器人自由度）
a = np.zeros(model.nv)  # 关节加速度

# 3.1 逆动力学（RNEA）：给定q/v/a，计算所需关节力矩
model.gravity.linear = np.array([0, 0, 9.81])  # 沿+z轴（可根据坐标系调整）
tau = pin.rnea(model, data, q, v, a)
print("\n=== 逆动力学结果（关节力矩）===")
print(f"关节力矩 τ: {tau} N·m")

tau_gravity = pin.computeGeneralizedGravity(model, data, q)  # 仅重力项
tau_coriolis = pin.computeCoriolisMatrix(model, data, q, v) @ v  # 仅科氏/离心项
mass_matrix = pin.crba(model, data, q)
tau_inertia = mass_matrix @ a  # 仅惯性项
print(f"重力项 τ_g: {tau_gravity} N·m")
print(f"科氏/离心项 τ_c: {tau_coriolis} N·m")
print(f"惯性项 τ_i: {tau_inertia} N·m")

# 正动力学（ABA）：给定q/v/τ，计算关节加速度
a_forward = pin.aba(model, data, q, v, tau)
print("\n=== 正动力学结果（关节加速度）===")
print(f"关节加速度 a: {a_forward} m/s² (rad/s²)")

# 机器人总质心和总质量
pin.forwardKinematics(model, data, q, v, a)  # 先更新几何数据
total_mass = pin.computeTotalMass(model)     # 总质量（累加所有连杆质量）
com = pin.centerOfMass(model, data, q)       # 总质心（世界坐标系）
print("\n=== 机器人整体惯性属性 ===")
print(f"总质量: {total_mass:.3f} kg")
print(f"总质心位置（世界坐标系）: {com.T} m")