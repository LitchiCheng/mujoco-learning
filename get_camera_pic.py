
import mujoco
import numpy as np
import glfw
import cv2

resolution = (640, 480) 
# 创建OpenGL上下文（离屏渲染）
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
glfw.make_context_current(window)

model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene_withcamera.xml')
data = mujoco.MjData(model)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# 设置相机参数
camera_name = "rgb_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
camera = mujoco.MjvCamera()
# 使用固定相机
# camera.type = mujoco.mjtCamera.mjCAMERA_FIXED  
# 设置相机为跟踪模式
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
if camera_id != -1:
    print("camera_id", camera_id)
    camera.fixedcamid = camera_id
    # camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body")

# 创建帧缓冲对象
framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

while True:
    mujoco.mj_step(model, data)
    tracking_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body")
    camera.trackbodyid = tracking_body_id
    camera.distance = -0.15  # 相机与目标的距离
    camera.azimuth = 0    # 水平方位角（度）
    camera.elevation = -90 # 俯仰角（度）
    viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), 
                         mujoco.MjvPerturb(), camera, 
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    # 转换颜色空间 (OpenCV使用BGR格式)
    bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    cv2.imshow('MuJoCo Camera Output', bgr)
    if cv2.waitKey(1) == 27:
        break

cv2.imwrite('debug_output.png', bgr)
cv2.destroyAllWindows()
glfw.terminate()
del context
del scene