import src.mujoco_viewer as mujoco_viewer
import pinocchio as pin
import src.utils as utils

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        
        self.initial_pos = self.model.key_qpos[0]
        self.data.qpos = self.initial_pos
        
    def runBefore(self):
        self.frame_name = "link7"
        self.pin_model = pin.RobotWrapper.BuildFromMJCF(self.arm_xml).model
        self.pin_data = self.pin_model.createData()
        self.ee_id = self.pin_model.getFrameId(self.frame_name)

    def getJacobian(self, model, data, q, end_effector_id):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        J = pin.computeFrameJacobian(
            model, data, q, end_effector_id, pin.LOCAL_WORLD_ALIGNED
        )
        return J
    
    def runFunc(self):
        j = self.getJacobian(self.pin_model, self.pin_data, self.data.qpos, self.ee_id)
        tau = self.data.qfrc_actuator + self.data.qfrc_passive + self.data.qfrc_applied
        F_mujoco = utils.dampedPinv(j.T, lambda_d=10e-5) @ tau
        print("F_mujoco:", F_mujoco.round(3))
        self.data.qpos = self.initial_pos

if __name__ == "__main__":
    SCENE_XML = "model/franka_emika_panda/scene_pos.xml"
    ARM_XML = "model/franka_emika_panda/panda_pos.xml"
    env = PandaEnv(SCENE_XML, ARM_XML)
    env.run_loop()