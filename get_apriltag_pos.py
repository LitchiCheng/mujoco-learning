import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import math
import src.solvepnp as solvepnp
import numpy as np
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
        self.initial_pos = self.model.qpos0.copy()
        TAG_SIZE = 0.1
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
        fovy = self.model.cam_fovy[camera_id]
        width = 640
        height = 480
        f = 0.5 * height / math.tan(fovy * math.pi / 360)
        # CAMERA_MATRIX = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # 格式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # fx/fy: 焦距, cx/cy: 主点坐标
        CAMERA_MATRIX = np.array([
            [414.42263304, 0.0, 318.91934938],
            [0.0, 414.314660431, 239.2895262],
            [0.0, 0.0, 1.0] 
        ], dtype=np.float32)
        DIST_COEFFS = np.array([0.005498773 ,-0.00174292 ,-0.0002786  ,-0.00070906 ,-0.00284597], dtype=np.float32)
        # DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32);
        self.spnp = solvepnp.SolvePnp(TAG_SIZE, CAMERA_MATRIX, DIST_COEFFS)
        self.apriltag_pos_x = 0.5
        self.apriltag_pos_y = 0
        self.apriltag_pos_z = 0.05

    def runFunc(self):
        if key_states[keyboard.Key.up]:
            self.apriltag_pos_z += 0.01
        if key_states[keyboard.Key.down]:
            self.apriltag_pos_z -= 0.01
        if key_states[keyboard.Key.left]:
            self.apriltag_pos_x -= 0.01
        if key_states[keyboard.Key.right]:
            self.apriltag_pos_x += 0.01
        if key_states[keyboard.Key.page_up]:
            self.apriltag_pos_y += 0.01
            print(self.apriltag_pos_y)
        if key_states[keyboard.Key.page_down]:
            self.apriltag_pos_y -= 0.01
        self.setMocapPosition("apriltag_0", [self.apriltag_pos_x, self.apriltag_pos_y, self.apriltag_pos_z])
        image = self.getFixedCameraImage(fix_elevation=-90)
        print(self.model.cam_ipd)
        if image is None:
            pass
        else:
            _,_,_,transform = self.spnp.compute(image, 0)
            self.spnp.show()
            print(transform)

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene_with_apriltag.xml")
    env.run_loop()
