import mujoco
import numpy as np
import glfw
import mujoco.viewer
from scipy.optimize import minimize

import numpy as np
import pinocchio
from numpy.linalg import norm, solve

ARM_XML_PATH = 'model/trs_so_arm100/so_arm100.xml'
SCENE_XML_PATH = 'model/trs_so_arm100/scene.xml'

def inverse_kinematics(current_q, target_dir, target_pos):
    
    model = pinocchio.RobotWrapper.BuildFromMJCF(ARM_XML_PATH).model
    # 为模型创建数据对象，用于存储计算过程中的中间结果
    data = model.createData()

    # 指定要控制的关节 ID
    JOINT_ID = 6
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-4
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 1000
    # 定义积分步长，用于更新关节角度
    DT = 1e-2
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-12

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(model, data, q)
        # 计算目标位姿到当前位姿之间的变换
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            break

        # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        # 使用阻尼最小二乘法求解关节速度
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        # 根据关节速度更新关节角度
        q = pinocchio.integrate(model, q, v * DT)

        # 每迭代 10 次打印一次当前的误差信息
        if not i % 10:
            print(f"{i}: error = {err.T}")
        # 迭代次数加 1
        i += 1

    # 根据算法是否收敛输出相应的信息
    if success:
        print("Convergence achieved!")
    else:
        print(
            "\n"
            "Warning: the iterative algorithm has not reached convergence "
            "to the desired precision"
        )

    # 打印最终的关节角度和误差向量
    print(f"\nresult: {q.flatten().tolist()}")
    print(f"\nfinal error: {err.T}")
    # 返回最终的关节角度向量（以列表形式）
    return q.flatten().tolist()

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.pos = 0.0001

        # 找到末端执行器的 body id
        self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'Moving_Jaw')
        print(f"End effector ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            print("Warning: Could not find the end effector with the given name.")

        # 初始关节角度
        self.initial_q = data.qpos[:6].copy()
        print(f"Initial joint positions: {self.initial_q}")
        theta = np.pi
        self.R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        self.x = 0.1
        self.new_q = self.initial_q

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport
    
    def run_loop(self):
        status = 0
        while self.is_running():
            mujoco.mj_forward(model, data)
            if self.x < 0.2: 
                self.x += 0.001
                new_q = inverse_kinematics(self.initial_q, self.R_x, [self.x, 0.1, 0.2])
            data.qpos[:6] = new_q
            mujoco.mj_step(model, data)
            self.sync()

viewer = CustomViewer(model, data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30
viewer.run_loop()