import numpy as np

def quat2rotmat(quat):
    """
    四元数（[w, x, y, z]）转换为旋转矩阵（3x3）。
    
    参数：
        quat: 长度为4的数组，格式为[w, x, y, z]（需归一化）
    
    返回：
        R: 3x3旋转矩阵
    """
    w, x, y, z = quat
    # 计算旋转矩阵各元素
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def euler2rotmat(roll, pitch, yaw):
    """
    将欧拉角（roll, pitch, yaw）转换为旋转矩阵（3x3）。
    旋转顺序：Z-Y-X（先绕Z轴yaw，再绕Y轴pitch，最后绕X轴roll），与用户提供的公式一致。
    
    参数：
        roll: 绕X轴旋转角度（弧度）
        pitch: 绕Y轴旋转角度（弧度）
        yaw: 绕Z轴旋转角度（弧度）
    
    返回：
        R: 3x3旋转矩阵
    """
    # 计算各角度的正弦和余弦
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # 根据Z-Y-X旋转顺序构造旋转矩阵（公式来自用户提供的代码）
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])
    return R

def transform2mat(x, y, z, roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp,   cp*sr,            cp*cr,            z],
        [0,     0,                0,                1]
    ])

def mat2transform(mat):
    x, y, z = mat[0:3, 3]
    roll, pitch, yaw = np.arctan2(mat[2, 1], mat[2, 2]), np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2)), np.arctan2(mat[1, 0], mat[0, 0])
    return x, y, z, roll, pitch, yaw

def euler2quat(roll, pitch, yaw):
    """
    将欧拉角（roll, pitch, yaw）转换为四元数（[w, x, y, z]）。    
    参数：
        roll: 绕X轴旋转角度（弧度）
        pitch: 绕Y轴旋转角度（弧度）
        yaw: 绕Z轴旋转角度（弧度）
    
    返回：
        quat: 长度为4的数组，格式为[w, x, y, z]
    """
    # 计算半角的正弦和余弦
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = -cy * sp * sr + sy * cp * cr
    return np.array([w, x, y, z])

def quat2euler(quat):
    """
    将四元数（[w, x, y, z]）转换为欧拉角（roll, pitch, yaw），弧度制。
    对应关系：与euler2quat完全匹配，欧拉角为X轴(roll)-Y轴(pitch)-Z轴(yaw)旋转顺序。
    
    参数：
        quat: 长度为4的数组/列表，四元数格式为[w, x, y, z]（w为实部，x/y/z为虚部）
    
    返回：
        roll: 绕X轴旋转的角度（弧度）
        pitch: 绕Y轴旋转的角度（弧度）
        yaw: 绕Z轴旋转的角度（弧度）
    """
    # 提取四元数各分量，确保为浮点数类型
    w, x, y, z = map(float, quat)
    # 计算俯仰角（pitch，Y轴）的正弦值，并用clip限制范围（避免浮点误差导致超出[-1,1]）
    sin_pitch = 2 * (w * y - x * z)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    # 常规情况（非万向锁状态）：pitch ≠ ±π/2
    if np.abs(sin_pitch) < 0.9999999:
        # 计算滚转角（roll，X轴）
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        # 计算俯仰角（pitch，Y轴）
        pitch = np.arcsin(sin_pitch)
        # 计算偏航角（yaw，Z轴）
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    # 万向锁状态（pitch ≈ π/2 或 ≈ -π/2），此时roll与yaw耦合，固定roll=0
    else:
        # 俯仰角取极值（π/2 或 -π/2）
        pitch = np.pi / 2 if sin_pitch > 0 else -np.pi / 2
        # 计算偏航角（yaw），roll固定为0
        yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (x**2 + z**2))
        roll = 0.0  # 万向锁状态下滚转角无法唯一确定，固定为0
    return roll, pitch, yaw

def dampedPinv(J, lambda_d=0.1):
    J_T = J.T
    damping = lambda_d ** 2 * np.eye(J.shape[0])
    J_pinv_damped = np.dot(J_T, np.linalg.inv(np.dot(J, J_T) + damping))
    return J_pinv_damped