import mujoco
import numpy as np
import glfw
from scipy.optimize import minimize

def scroll_callback(window, xoffset, yoffset):
    global cam
    # 调整相机的缩放比例
    cam.distance *= 1 - 0.1 * yoffset

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def main():
    global cam
    # 加载模型
    model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # 打印所有 body 的 ID 和名称
    print("All bodies in the model:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"ID: {i}, Name: {body_name}")

    # 初始化 GLFW
    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, 'Panda Arm Control', None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 设置鼠标滚轮回调函数
    glfw.set_scroll_callback(window, scroll_callback)

    # 初始化渲染器
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    scene = mujoco.MjvScene(model, maxgeom=10000)

    # 找到末端执行器的 body id
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    print(f"End effector ID: {end_effector_id}")
    if end_effector_id == -1:
        print("Warning: Could not find the end effector with the given name.")
        glfw.terminate()
        return

    # 初始关节角度
    initial_q = data.qpos[:7].copy()
    print(f"Initial joint positions: {initial_q}")

    while not glfw.window_should_close(window):

        # 获取当前末端执行器位置
        mujoco.mj_forward(model, data)
        end_effector_pos = data.body(end_effector_id).xpos

        initial_q[0] = initial_q[0] + 0.1
        initial_q[0] = limit_angle(initial_q[0])
        new_q = initial_q
        # 设置关节目标位置
        data.qpos[:7] = new_q

        # 模拟一步
        mujoco.mj_step(model, data)

        # 更新渲染场景
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, con)

        # 交换前后缓冲区
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 清理资源
    glfw.terminate()


if __name__ == "__main__":
    main()
    