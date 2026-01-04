import src.mujoco_viewer as mujoco_viewer
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time,math
import src.ploter as ploter

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
        # self.plotter = ploter.Ploter()

    def runFunc(self):
        if True:
            self.data.ctrl[:7] = self.initial_pos
            # self.data.qfrc_applied[0] = 5.0
            # self.time_history.append(self.data.time)
            # print(f"Torque: {self.data.qfrc_actuator[:7].round(2)}")
            print(f"Torque applied: {self.data.xfrc_applied[:7].round(4)}")
            # for i in range(7):
            #     self.plotter.update_data(0, self.data.qfrc_actuator[i].round(2), label="qfrc_actuator_"+str(i))

if __name__ == "__main__":
    test = Test("./model/franka_emika_panda/scene.xml")
    test.run_loop()

    