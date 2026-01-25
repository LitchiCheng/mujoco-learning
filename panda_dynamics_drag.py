import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pinocchio as pin

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        
        self.initial_pos = self.model.key_qpos[0]
        self.data.qpos[:7] = self.initial_pos[:7]
        self.data.qvel[:7] = np.zeros(7)

        self.step = 0
        self.step_list = []
        self.dynamics_tau_list = []
        self.damping_tau_list = []

    def runBefore(self):
        self.pin_model = pin.RobotWrapper.BuildFromMJCF(self.arm_xml).model
        self.pin_data = self.pin_model.createData()
        self.last_v = np.zeros(7)

    def runFunc(self):
        q = self.data.qpos[:7]
        v = self.data.qvel[:7]
        a = (v - self.last_v) / self.model.opt.timestep
        self.last_v = v.copy()
        # a = np.zeros(7)
        v = np.concatenate((v, np.zeros(2)))
        q = np.concatenate((q, np.zeros(2)))
        a = np.concatenate((a, np.zeros(2)))
        dynamics_tau = pin.rnea(self.pin_model, self.pin_data, q, v, a)
        DAMPING = -100
        damping_tau = DAMPING * v
        tau = dynamics_tau + damping_tau
        
        self.data.ctrl[:7] = tau[:7]
        print(f"Total Torque: {np.round(tau[:7], 2)}")
        print(f"damping_tau: {np.round(damping_tau[:7], 2)}")

        self.step += 1
        self.step_list.append(self.step)
        self.dynamics_tau_list.append(dynamics_tau[:7].copy())
        self.damping_tau_list.append(damping_tau[:7].copy())
        # if self.step >= 2000:
        #     self.plotTorque()

    def plotTorque(self):
        steps = np.array(self.step_list)
        dynamics_tau = np.array(self.dynamics_tau_list)
        damping_tau = np.array(self.damping_tau_list)
        fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
        fig.suptitle('dynamics_tau & damping_tau', fontsize=14, fontweight='bold')
        joint_names = ['j1', 'j2', 'j4', 'j5', 'j6', 'j7', 'j8']
        for i in range(7):
            axes[i].plot(steps, dynamics_tau[:, i], label=f'dynamics_tau', color='blue', linewidth=1.5)
            axes[i].plot(steps, damping_tau[:, i], label=f'damping_tau', color='red', linewidth=1.5)
            axes[i].set_ylabel(f'{joint_names[i]} Tau (N·m)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right')
        axes[-1].set_xlabel('step')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    SCENE_XML = "model/franka_emika_panda/scene_tau.xml"
    ARM_XML = "model/franka_emika_panda/panda_tau.xml"
    env = PandaEnv(SCENE_XML, ARM_XML)
    env.run_loop()