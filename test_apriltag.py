import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import cv2
from pupil_apriltags import Detector

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        self.initial_pos = self.model.qpos0.copy()
        # 初始化 AprilTag 检测器
        self.detector = Detector(families="tag36h11",
                            nthreads=1,
                            quad_decimate=1.0,
                            refine_edges=1)

    def runFunc(self):
        image = self.getTrackingCameraImage(fix_elevation=-90)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray_image)
        for tag in tags:
            print(f"Detected tag ID: {tag.tag_id}, Center: {tag.center}")

if __name__ == "__main__":
    env = PandaEnv("./model/franka_emika_panda/scene_with_apriltag.xml")
    env.run_loop()
