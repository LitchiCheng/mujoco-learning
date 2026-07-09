import math

# 定义角度
angles = [170, -190, 180, -180, 180, 540]

# 遍历角度列表
for theta in angles:
    # 将角度转换为弧度
    theta_rad = math.radians(theta)

    # 计算 cos 和 sin 值
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # 输出结果
    print(f"当 theta = {theta} 度时:")
    print(f"cos(theta) = {cos_theta}")
    print(f"sin(theta) = {sin_theta}")
    print()
    