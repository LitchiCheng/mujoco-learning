import mujoco_viewer 
import mujoco
import src.kdl_kinematic as kdl_kinematic
import src.pinocchio_kinematic as pinocchio_kinematic

class CheckFk(mujoco_viewer.CustomViewer):
    def __init__(self, rendor_path, arm_path):
        super().__init__(rendor_path, 3, azimuth=-45, elevation=-30)
        self.arm_path = arm_path

        self.ee_body_name = "link7"
        # # 初始化逆运动学
        self.arm2 = kdl_kinematic.Kinematics(self.ee_body_name)
        urdf_file = "model/franka_panda_urdf/robots/panda_arm.urdf"
        self.arm2.buildFromURDF(urdf_file, "link0")

        self.arm1 = pinocchio_kinematic.Kinematics(self.ee_body_name)
        self.arm1.buildFromMJCF(self.arm_path)

    def runBefore(self):
        self.model.opt.timestep = 0.005
        # 设定初始位置
        self.initial_pos = self.model.key_qpos[0]  
        print("Initial position: ", self.initial_pos)
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]

        

    def runFunc(self):
        # self.data.qpos[:7] = self.initial_pos[:7]
        print("Mujoco body position: ")
        print(self.getBodyPosByName(self.ee_body_name))
        self.fk_tf1 = self.arm1.fk(self.data.qpos)
        self.fk_tf2 = self.arm2.fk(self.data.qpos)
        print("FK methed 1 result: ")
        print(self.fk_tf1)
        print("FK methed 2 result: ")
        print(self.fk_tf2)


if __name__ == '__main__':
    SCENE_XML_PATH = 'model/franka_emika_panda/scene_pos.xml'
    ARM_XML_PATH = 'model/franka_emika_panda/panda_pos.xml'
    robot = CheckFk(SCENE_XML_PATH, ARM_XML_PATH)
    robot.run_loop()