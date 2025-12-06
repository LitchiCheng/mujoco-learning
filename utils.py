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