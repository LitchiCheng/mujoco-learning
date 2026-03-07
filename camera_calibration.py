import src.mujoco_viewer as mujoco_viewer
import time
import mujoco
import math
import src.solvepnp as solvepnp
import numpy as np
import src.key_listener as key_listener
from pynput import keyboard
import cv2
from typing import Tuple, List

# 按键状态字典
key_states = {
    keyboard.Key.up: False,
    keyboard.Key.down: False,
    keyboard.Key.left: False,
    keyboard.Key.right: False,
    keyboard.Key.page_up: False,
    keyboard.Key.page_down: False,
    keyboard.Key.space: False,  # 空格键用于采集标定图像
    keyboard.Key.enter: False   # 回车键用于执行标定
}

# 标定棋盘格参数
CALIB_BOARD_SIZE = (8, 5)      # 棋盘格内角点数量 (宽, 高)
CALIB_SQUARE_SIZE = 0.02       # 棋盘格每个方格的物理尺寸 (米)
# 生成棋盘格世界坐标
objp = np.zeros((np.prod(CALIB_BOARD_SIZE), 3), np.float32)
objp[:, :2] = np.mgrid[0:CALIB_BOARD_SIZE[0], 0:CALIB_BOARD_SIZE[1]].T.reshape(-1, 2)
objp *= CALIB_SQUARE_SIZE

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
        self.key_listener = key_listener.KeyListener(key_states)
        self.key_listener.start()
        
        # 标定相关初始化
        self.calib_object_points = []  # 存储标定用的世界坐标点
        self.calib_image_points = []   # 存储标定用的图像坐标点
        self.calib_images = []         # 存储采集的标定图像
        self.camera_matrix = None      # 标定后的相机内参矩阵
        self.dist_coeffs = None        # 标定后的畸变系数
        self.calib_done = False        # 标定完成标志
    
    def runBefore(self):
        """初始化运行前的参数"""
        self.initial_pos = self.model.qpos0.copy()
        self.checker_board_x = 0.5
        self.checker_board_y = 0
        self.checker_board_z = 0.4

    def collect_calib_image(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 去抗锯齿噪声
        gray = cv2.Canny(gray, 50, 150)          # 提取边缘，强化角点
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, CALIB_BOARD_SIZE,
            # 允许更快检测，接受边缘模糊的角点
            cv2.CALIB_CB_ADAPTIVE_THRESH 
            + cv2.CALIB_CB_NORMALIZE_IMAGE 
            + cv2.CALIB_CB_FAST_CHECK  # 没有角点直接返回，提升速度
        )
        
        if ret:
            # 降低精度要求，加快速度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            self.calib_object_points.append(objp)
            self.calib_image_points.append(corners2)
            self.calib_images.append(image)
            img_with_corners = cv2.drawChessboardCorners(image.copy(), CALIB_BOARD_SIZE, corners2, ret)
            cv2.imshow('Calibration Image', img_with_corners)
            cv2.waitKey(300)
            print(f"成功采集第 {len(self.calib_images)} 张标定图像")
            return True
        else:
            # 调试：显示当前灰度图
            cv2.imshow('Calibration Image', gray)
            cv2.waitKey(100)
            print("未检测到角点，请调整棋盘格位置/视角")
            return False
        
    def calibrate_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.calib_images) < 5:
            print(f"标定图像数量不足（当前{len(self.calib_images)}张，建议至少5张）")
            return None, None
        
        print("\n开始相机标定...")
        gray = cv2.cvtColor(self.calib_images[0], cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.calib_object_points,
            self.calib_image_points,
            (w, h),
            None,
            None
        )
        
        if ret:
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # 计算重投影误差
            total_error = 0
            for i in range(len(self.calib_object_points)):
                img_points2, _ = cv2.projectPoints(
                    self.calib_object_points[i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(self.calib_image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            
            mean_error = total_error / len(self.calib_object_points)
            
            print("\n标定完成")
            print(f"平均重投影误差: {mean_error:.6f} (越小越好，建议小于<0.1)")
            print(f"相机内参矩阵:\n{mtx}")
            print(f"畸变系数:\n{dist}")
            print(f"优化后相机矩阵:\n{new_mtx}")
            
            self.camera_matrix = new_mtx
            self.dist_coeffs = dist
            self.calib_done = True
        
            return new_mtx, dist
        else:
            print("标定失败")
            return None, None

    def runFunc(self):
        if key_states[keyboard.Key.up]:
            self.checker_board_z += 0.01
        if key_states[keyboard.Key.down]:
            self.checker_board_z -= 0.01
        if key_states[keyboard.Key.left]:
            self.checker_board_x -= 0.01
        if key_states[keyboard.Key.right]:
            self.checker_board_x += 0.01
        if key_states[keyboard.Key.page_up]:
            self.checker_board_y += 0.01
        if key_states[keyboard.Key.page_down]:
            self.checker_board_y -= 0.01
        
        self.setMocapPosition("calib_checkerboard", [self.checker_board_x, self.checker_board_y, self.checker_board_z])
        image = self.getFixedCameraImage(fix_elevation=-90, show=True)
    
        if image is not None:
            # 转换图像格式（MuJoCo RGBA转为RGB）
            if image.shape[-1] == 4:
                image = image[..., :3]
            
            # 空格键采集标定图像
            if key_states[keyboard.Key.space]:
                self.collect_calib_image(image)
                key_states[keyboard.Key.space] = False  # 防止重复采集
            
            # 回车键执行标定
            if key_states[keyboard.Key.enter]:
                self.calibrate_camera()
                key_states[keyboard.Key.enter] = False  # 防止重复标定
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
                fovy = self.model.cam_fovy[camera_id]
                width = 640
                height = 480
                f = 0.5 * height / math.tan(fovy * math.pi / 360)

                CAMERA_MATRIX = np.array([
                    [f, 0, width / 2],
                    [0, f, height / 2],
                    [0, 0, 1]
                ], dtype=np.float32)

                DIST_COEFFS = np.zeros(5, dtype=np.float32)  # 虚拟相机无畸变
                print("Mujoco FOV直接计算的内参：")
                print(CAMERA_MATRIX)

if __name__ == "__main__":
    cv2.namedWindow('Calibration Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration Image', 640, 480)
    env = PandaEnv("./model/franka_emika_panda/scene_with_checkerboard.xml")
    try:
        env.run_loop()
    finally:
        cv2.destroyAllWindows()
        env.key_listener.stop()