import mujoco
import numpy as np
import src.mujoco_viewer as mujoco_viewer
import src.pinocchio_kinematic as pinocchio_kinematic
import src.utils as utils
import src.key_listener as key_listener
from pynput import keyboard

# 按键状态字典
key_states = {
    keyboard.Key.up: False,
    keyboard.Key.down: False,
    keyboard.Key.left: False,
    keyboard.Key.right: False,
    keyboard.Key.page_up: False,
    keyboard.Key.page_down: False,
    keyboard.Key.ctrl_l: False,  # 夹爪开
    keyboard.Key.ctrl_r: False   # 夹爪收
}

class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path):
        super().__init__(scene_path, distance=2.0, azimuth=-135, elevation=-20)
        self.scene_path = scene_path
        self.arm_path = arm_path

        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
        
        self.arm = pinocchio_kinematic.Kinematics("ee_center_body")
        self.arm.buildFromMJCF(arm_path)
        self.last_dof = None
        
    def runBefore(self):
        self.model.opt.timestep = 0.005
        self.ee_cmd_x = 0.1
        self.ee_cmd_y = 0.0
        self.ee_cmd_z = 0.4
        self.grip_value = 0
        self.last_dof = self.data.qpos[:9].copy()
    
    def runFunc(self):
        if key_states[keyboard.Key.up]:
            self.ee_cmd_z += 0.001
        if key_states[keyboard.Key.down]:
            self.ee_cmd_z -= 0.001
        if key_states[keyboard.Key.left]:
            self.ee_cmd_x -= 0.001
        if key_states[keyboard.Key.right]:
            self.ee_cmd_x += 0.001
        if key_states[keyboard.Key.page_up]:
            self.ee_cmd_y += 0.001
        if key_states[keyboard.Key.page_down]:
            self.ee_cmd_y -= 0.001
        if key_states[keyboard.Key.ctrl_l]:
            self.grip_value += 5
        if key_states[keyboard.Key.ctrl_r]:
            self.grip_value -= 5

        tf = utils.transform2mat(self.ee_cmd_x, self.ee_cmd_y, self.ee_cmd_z, np.pi, 0, 0)
        self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
        self.last_dof = self.dof
        self.data.ctrl[:7] = self.dof[:7]
        self.data.ctrl[7] = self.grip_value

        geom_id = self.getGeomIdByName("cube_geom")
        friction = self.model.geom_friction[geom_id]
        condim = self.model.geom_condim[geom_id]
        print(f"condim: {condim} sliding: {friction[0]} torsional: {friction[1]} rolling: {friction[2]}")

if __name__ == "__main__":
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_with_cube.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda.xml'
    robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()
