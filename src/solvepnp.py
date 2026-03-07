import cv2
import numpy as np
from pupil_apriltags import Detector

import signal
import sys

# 全局变量，用于控制主循环
is_running = True

def signal_handler(sig, frame):
    """
    捕获Ctrl+C信号的处理函数
    :param sig: 信号类型
    :param frame: 栈帧
    """
    global is_running
    is_running = False

# 注册信号处理函数，捕获Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, signal_handler)

class SolvePnp:
    def __init__(self, tag_size=0.1, camera_intrinsic=None, camera_distort=None):
        self.setTagSize(tag_size)
        self.setCameraInstrinsic(camera_intrinsic)
        self.setCameraDistort(camera_distort)
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            refine_edges=1
        )

    def setCameraInstrinsic(self, intrinsic):
        self.camera_intrinsic = intrinsic
    
    def setCameraDistort(self, distort):
        self.camera_distort = distort

    def setTagSize(self, size):
        self.tag_size = size
        # 定义AprilTag在世界坐标系中的3D角点 (假设标签位于z=0平面，中心在原点)
        # 顺序：左上、右上、右下、左下 (与detect返回的corners顺序一致)
        self.obj_points = np.array([
            [-size/2, size/2, 0],   # 左上
            [size/2, size/2, 0],    # 右上
            [size/2, -size/2, 0],   # 右下
            [-size/2, -size/2, 0]   # 左下
        ], dtype=np.float32)

    def compute(self, bgr_img, tag_id):
        self.last_id = tag_id
        self.last_img = bgr_img
        self.gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        self.tags = self.detector.detect(self.gray_img)
        find_tag = False
        for tag in self.tags:
            if tag.tag_id == tag_id:
                find_tag = True
                # 打印基础检测信息
                print(f"\n=== 检测到标签 ID: {tag.tag_id} ===")
                print(f"中心坐标 (像素): {tag.center}")
                
                # 提取2D图像角点坐标
                img_points = np.array(tag.corners, dtype=np.float32)
                
                # 使用solvePnP计算位姿 (rvec: 旋转向量, tvec: 平移向量)
                # SOLVEPNP_IPPE_SQUARE 适合正方形标签，精度更高
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, 
                    img_points, 
                    self.camera_intrinsic, 
                    self.camera_distort,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    # print(f"旋转向量 (rvec): \n{rvec}")
                    # print(f"平移向量 (tvec): \n{tvec}")
                    
                    # # 将旋转向量转换为旋转矩阵
                    rmat, _ = cv2.Rodrigues(rvec)
                    # print(f"旋转矩阵 (rmat): \n{rmat}")
                    
                    # 在图像上绘制坐标轴
                    axis_length = self.tag_size / 2
                    imgpts, _ = cv2.projectPoints(
                        np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]),
                        rvec, tvec, self.camera_intrinsic, self.camera_distort
                    )
                    imgpts = np.int32(imgpts).reshape(-1,2)
                    
                    # 绘制标签角点和中心
                    center = tuple(map(int, tag.center))
                    for corner in tag.corners:
                        corner = tuple(map(int, corner))
                        cv2.circle(self.last_img, corner, 5, (0, 255, 0), -1)
                    cv2.circle(self.last_img, center, 5, (255, 0, 0), -1)
                    
                    # 绘制坐标轴 (X:红, Y:绿, Z:蓝)
                    cv2.line(self.last_img, center, tuple(imgpts[0]), (0,0,255), 2)  # X轴
                    cv2.line(self.last_img, center, tuple(imgpts[1]), (0,255,0), 2)  # Y轴
                    cv2.line(self.last_img, center, tuple(imgpts[2]), (255,0,0), 2)  # Z轴
                    
                    # 在图像上标注标签ID
                    cv2.putText(
                        self.last_img, 
                        f"ID: {tag.tag_id}", 
                        (center[0]-20, center[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255,255,0), 
                        2
                    )
                    homogeneous_matrix = np.eye(4, dtype=np.float64)
                    homogeneous_matrix[:3, :3] = rmat
                    homogeneous_matrix[:3, 3] = tvec.ravel()
                    return tag, rvec, tvec, homogeneous_matrix
                else:
                    print("compute failed")
                    return None
        if not find_tag:
            # print("No tag id %d " % tag_id)
            return None,None,None,None    
            
    def show(self):
        if self.last_img is not None:
            title = "AprilTag Detection with Pose"
            cv2.imshow(title, self.last_img)
            # while True:
            # 10ms检测一次，如果q或ESC按下或窗口消失，则退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                # break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                # break

    def save(self):
        if self.last_img is not None:
            cv2.imwrite("result_" + str(self.last_id), self.last_img)

if __name__ == '__main__':
    image_path = "../model/misc/textures/tag36h11_0.png"
    TAG_SIZE = 0.1
    # 格式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # fx/fy: 焦距, cx/cy: 主点坐标
    CAMERA_MATRIX = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"can not read image: {image_path}")
    
    spnp = SolvePnp(TAG_SIZE, CAMERA_MATRIX, DIST_COEFFS)
    _,_,_,transform = spnp.compute(image, 0)
    print(transform)
    spnp.show()