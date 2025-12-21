import mujoco_viewer
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time,math

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path
    
    def runBefore(self):
        # 存储关节力矩的列表
        self.torque_history = []
        self.time_history = []
        self.initial_pos = self.model.key_qpos[0][:7]  # 仅取7个关节维度
        # 初始化机械臂到初始位置（仅一次）
        self.data.qpos[:7] = self.initial_pos
       
    def runFunc(self):
        if True:
            self.data.qpos[:7] = self.initial_pos
            self.time_history.append(self.data.time)
            # self.torque_history.append(self.data.qfrc_actuator.copy())  # 存储关节力矩
            # print(self.data.qfrc_actuator)
            print(f"Torque: {np.round(self.data.qfrc_actuator[:7], 2)}")
            # if len(self.torque_history) > 20000:
            #     torque_history = np.array(self.torque_history)
            #     # 绘制关节力矩曲线
            #     plt.figure(figsize=(10, 6))
            #     for i in range(torque_history.shape[1]):
            #         plt.subplot(torque_history.shape[1], 1, i + 1)
            #         plt.plot(self.time_history, torque_history[:, i], label=f'Joint {i + 1} Torque')
            #         plt.xlabel('Time (s)')
            #         plt.ylabel('Torque (N·m)')
            #         plt.title(f'Joint {i + 1} Torque Over Time')
            #         plt.legend()
            #         plt.grid(True)
            #     # plt.tight_layout()
            #     plt.show()

if __name__ == "__main__":
    test = Test("./model/franka_emika_panda/scene_pos.xml")
    test.run_loop()

    