import mujoco
import src.mujoco_viewer as mujoco_viewer
import src.key_listener as key_listener
from pynput import keyboard
key_states = {
    keyboard.Key.up: False,
    keyboard.Key.down: False,
    keyboard.Key.left: False,
    keyboard.Key.right: False,
    keyboard.Key.page_up: False,
    keyboard.Key.page_down: False,
}

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
    
    def runBefore(self):
        self.initial_pos = self.model.key_qpos[0]
        self.obs_idx = self.getGeomIdByName("obstacle_0")
        self.obs_x = self.model.geom_pos[self.obs_idx][0]
        self.obs_y = self.model.geom_pos[self.obs_idx][1]
        self.obs_z = self.model.geom_pos[self.obs_idx][2]
        print("obs_pos", [self.obs_x, self.obs_y, self.obs_z])
        print("obs_contact", self.model.geom_conaffinity[self.obs_idx], self.model.geom_contype[self.obs_idx])
        self.model.geom_contype[self.obs_idx] = 1
        self.model.geom_conaffinity[self.obs_idx] = 1

    def runFunc(self):
        for i in range(self.model.nq):
            self.data.qpos[i] = self.initial_pos[i]
        
        if key_states[keyboard.Key.up]:
            self.obs_z += 0.001
        if key_states[keyboard.Key.down]:
            self.obs_z -= 0.001
        if key_states[keyboard.Key.left]:
            self.obs_x -= 0.001
        if key_states[keyboard.Key.right]:
            self.obs_x += 0.001
        if key_states[keyboard.Key.page_up]:
            self.obs_y += 0.001
        if key_states[keyboard.Key.page_down]:
            self.obs_y -= 0.001

        self.setGeomPositionByName("obstacle_0", [self.obs_x, self.obs_y, self.obs_z])  

        if  self.data.ncon > 0:
            print(self.getContactInfo())

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene_pos_with_obstacles.xml")
    env.run_loop()
