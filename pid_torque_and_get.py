import mujoco_viewer
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time,math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path
    
    def runBefore(self):
        kp = 10.0
        ki = 1.0
        kd = 5.0
        self.pid = PIDController(kp, ki, kd)
        self.target_position = 3.14 / 4
        self.dt = self.model.opt.timestep
        self.positions = []
        self.torques = []
       
    def runFunc(self):
        wait_time = 0.01
        time.sleep(wait_time)
        # 获取当前关节位置
        current_position = self.data.qpos[0]
        # 计算误差
        error = self.target_position - current_position
        # 使用 PID 控制器计算扭矩
        torque = self.pid.update(error, self.dt + wait_time)
        print(f"error: {error}, Current Position: {current_position}, Target Position: {self.target_position}, Torque: {torque}")
        # 设置关节扭矩
        self.data.ctrl[0] = torque
        # 记录数据
        self.positions.append(current_position)
        self.torques.append(torque)
        if math.fabs(error) < 0.001:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(self.positions, label='Joint Position')
            plt.axhline(y=self.target_position, color='r', linestyle='--', label='Target Position')
            plt.xlabel('Time Step')
            plt.ylabel('Position')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(self.torques, label='Torque')
            plt.xlabel('Time Step')
            plt.ylabel('Torque')
            plt.legend()

            plt.show()
            self.target_position = 0.0
        

if __name__ == "__main__":
    test = Test("model/franka_emika_panda/scene_tau.xml")
    test.run_loop()

    