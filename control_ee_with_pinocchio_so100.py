import mujoco
import numpy as np
import mujoco.viewer
import numpy as np
import src.pinocchio_kinematic as pinocchio_kinematic
import time

SCENE_XML_PATH = 'model/trs_so_arm100/scene.xml'
ARM_XML_PATH = 'model/trs_so_arm100/so_arm100.xml'

model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.x = 0.3
        self.y = 0.0
        self.z = 0.1
        self.arm = pinocchio_kinematic.Kinematics("Jaw")
        self.arm.buildFromMJCF(ARM_XML_PATH)

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
            theta = np.pi
            self.z = self.z + 0.001
            if self.z > 0.3:
                self.z = 0.3
            print(f"Current position: x={self.x}, y={self.y}, z={self.z}")
            tf = np.array([
                    [1, 0, 0, self.x],
                    [0, np.cos(theta), -np.sin(theta), self.y],
                    [0, np.sin(theta), np.cos(theta), self.z],
                ])
            tf = np.vstack((tf, [0, 0, 0, 1]))
            self.dof, info = self.arm.ik(tf)
            print(f"DoF: {self.dof}, Info: {info}")
            data.qpos[:6] = self.dof[:6]
            mujoco.mj_step(model, data)
            self.sync()
            time.sleep(0.01)

viewer = CustomViewer(model, data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30
viewer.run_loop()