import src.mujoco_viewer as mujoco_viewer as mujoco_viewer
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
       
    def runFunc(self):
        if True:
            self.time_history.append(self.data.time)
            self.torque_history.append(self.data.qfrc_actuator.copy())  # 存储关节力矩
            if len(self.torque_history) > 20000:
                torque_history = np.array(self.torque_history)
                # 绘制关节力矩曲线
                plt.figure(figsize=(10, 6))
                for i in range(torque_history.shape[1]):
                    plt.subplot(torque_history.shape[1], 1, i + 1)
                    plt.plot(self.time_history, torque_history[:, i], label=f'Joint {i + 1} Torque')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Torque (N·m)')
                    plt.title(f'Joint {i + 1} Torque Over Time')
                    plt.legend()
                    plt.grid(True)
                # plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    test = Test("model/trs_so_arm100/scene.xml")
    test.run_loop()

    