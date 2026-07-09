import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH

# MDH参数 [α, a, d, θ偏移]
links = [
    RevoluteMDH(alpha=0, a=0.0452,      d=0.0165,      qlim=[-160, 160]*np.pi/180),  # Joint 1
    RevoluteMDH(alpha=-np.pi/2,      a=0.0306, d=0,      qlim=[-45, 225]*np.pi/180),   # Joint 2
    RevoluteMDH(alpha=0, a=0.11257, d=0, qlim=[-225, 45]*np.pi/180),  # Joint 3
    RevoluteMDH(alpha=0, a=0.1349,      d=0, qlim=[-110, 170]*np.pi/180),  # Joint 4
    RevoluteMDH(alpha=-np.pi/2, a=0,      d=-0.0601,   offset=-np.pi/2,   qlim=[-100, 100]*np.pi/180),  # Joint 5
    RevoluteMDH(alpha=np.pi/2,      a=0.0202,      d=0,      qlim=[-266, 266]*np.pi/180)   # Joint 6
]

# 创建机器人模型
test = DHRobot(links, name="so-arm100")
print(test)