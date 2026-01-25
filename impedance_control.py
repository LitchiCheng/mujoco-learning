import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import time
import src.mujoco_viewer as mujoco_viewer
import numpy as np

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        # 阻抗控制参数
        # 确保 Kp 和 Kd 的维度与实际控制的关节数量匹配
        self.Kp = np.diag([100] * self.model.nu)  # 刚度矩阵
        self.Kd = np.diag([10] * self.model.nu)  # 阻尼矩阵

        # 目标关节角度
        # self.q_desired = np.zeros(self.model.nu)
        self.q_desired = [0.0, -0.991, 0.196, 0.662, -0.88, 0.66]

        # 仿真参数
        self.total_time = 30  # 总仿真时间（秒）
        self.dt = self.model.opt.timestep  # 仿真时间步长
        self.num_steps = int(self.total_time / self.dt)

        # 存储数据
        self.q_history = np.zeros((self.num_steps, self.model.nu))
        self.qdot_history = np.zeros((self.num_steps, self.model.nu))
        self.torque_history = np.zeros((self.num_steps, self.model.nu))
        self.index = 0
    
    def runFunc(self):
        # 读取当前关节角度和速度
        q = self.data.qpos[:self.model.nu]
        qdot = self.data.qvel[:self.model.nu]

        # 计算阻抗控制扭矩
        error = self.q_desired - q
        print(self.index, self.num_steps, self.model.nu, error)
        torque = self.Kp @ error - self.Kd @ qdot

        # 设置控制输入
        self.data.ctrl[:] = torque

        # 存储数据
        self.q_history[self.index] = q
        self.qdot_history[self.index] = qdot
        self.torque_history[self.index] = torque
        self.index += 1

        if self.index >= self.num_steps:
            # # 绘制结果
            time = np.arange(0, self.total_time, self.dt)

            plt.figure(figsize=(12, 8))

            # 绘制关节角度
            plt.subplot(3, 1, 1)
            for j in range(self.model.nu):
                plt.plot(time, self.q_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Angles')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (rad)')
            plt.legend()

            # 绘制关节速度
            plt.subplot(3, 1, 2)
            for j in range(self.model.nu):
                plt.plot(time, self.qdot_history[:, j], label=f'Joint {j+1}')
            plt.title('Joint Velocities')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.legend()

            # 绘制控制扭矩
            plt.subplot(3, 1, 3)
            for j in range(self.model.nu):
                plt.plot(time, self.torque_history[:, j], label=f'Joint {j+1}')
            plt.title('Control Torques')
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (N.m)')
            plt.legend()

            plt.tight_layout()
            plt.show()

test = Test("./model/trs_so_arm100/scene_without_position.xml")
test.run_loop()

