import mujoco
import numpy as np
import mujoco_viewer
import casadi_ik
import time
import pygame
import os

# 设置环境变量以确保正确访问游戏杆设备
os.environ["SDL_JOYSTICK_DEVICE"] = "/dev/input/js0"

SCENE_XML_PATH = 'model/trs_so_arm100/scene.xml'
ARM_XML_PATH = 'model/trs_so_arm100/so_arm100.xml'

class XboxController:
    """Xbox手柄控制器类，负责处理所有手柄输入"""
    
    def __init__(self):
        # 初始化位置参数
        self.x = 0.0
        self.y = -0.3
        self.z = 0.15
        
        # 位置限制
        self.x_min, self.x_max = -0.3, 0.3
        self.y_min, self.y_max = -0.4, 0.0
        self.z_min, self.z_max = 0.05, 0.3
        
        # 控制灵敏度
        self.sensitivity = 0.005
        
        # 死区阈值（过滤摇杆中心微小漂移）
        self.deadzone = 0.1
        
        # 存储dof[5]的目标值
        self.dof5_target = None
        
        # 跟踪按钮状态，避免重复触发
        self.button1_pressed = False
        self.button2_pressed = False
        
        # 初始化手柄
        self.controller = self.init_controller()
        
    def init_controller(self):
        """初始化Xbox手柄（通过pygame访问js设备）"""
        
        # 初始化pygame和游戏杆模块
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("未检测到任何游戏杆设备")
            return None
            
        # 对应/dev/input/js0
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"检测到手柄: {joystick.get_name()}")
        print(f"按钮数量: {joystick.get_numbuttons()}")
        return joystick
        
    def is_connected(self):
        """检查手柄是否连接"""
        return self.controller is not None
        
    def handle_input(self, arm):
        """处理手柄输入并更新控制参数"""
        if not self.is_connected():
            return
            
        # 处理pygame事件（必须调用，否则无法更新轴值和按钮状态）
        pygame.event.pump()
        
        # 读取左摇杆X轴（0号轴）
        x_axis = self.controller.get_axis(1)
        # 应用死区过滤
        if abs(x_axis) < self.deadzone:
            x_axis = 0.0
            
        # 读取左摇杆Y轴（1号轴）
        y_axis = self.controller.get_axis(0)
        if abs(y_axis) < self.deadzone:
            y_axis = 0.0
            
        # 读取右摇杆Y轴（4号轴）
        z_axis = -self.controller.get_axis(4)
        if abs(z_axis) < self.deadzone:
            z_axis = 0.0
        
        # 根据轴值更新位置
        self.x = np.clip(self.x + x_axis * self.sensitivity, 
                       self.x_min, self.x_max)
        self.y = np.clip(self.y + y_axis * self.sensitivity, 
                       self.y_min, self.y_max)
        self.z = np.clip(self.z + z_axis * self.sensitivity, 
                       self.z_min, self.z_max)
        
        # 检测按钮1（假设是0号按钮，通常是A键）
        button1 = self.controller.get_button(0)
        if button1 and not self.button1_pressed:
            print("Button1 pressed - 设置dof[5]为满量程")
            # 设置dof[5]为满量程值
            self.dof5_target = arm.model.upperPositionLimit[5]
            self.button1_pressed = True
        elif not button1:
            self.button1_pressed = False
            
        # 检测按钮2（假设是1号按钮，通常是B键）
        button2 = self.controller.get_button(1)
        if button2 and not self.button2_pressed:
            print("Button2 pressed - 设置dof[5]为最小值")
            # 设置dof[5]为最小值
            self.dof5_target = arm.model.lowerPositionLimit[5]
            self.button2_pressed = True
        elif not button2:
            self.button2_pressed = False

    def get_position_target(self):
        return self.x, self.y, self.z
        
    def get_dof5_target(self):
        return self.dof5_target
        
    def cleanup(self):
        pygame.quit()


class RobotController(mujoco_viewer.CustomViewer):    
    def __init__(self, scene_path, arm_path, controller):
        super().__init__(scene_path, distance=1.5, azimuth=135, elevation=-30)
        self.scene_path = scene_path
        self.arm_path = arm_path
        self.controller = controller
        
        # 初始化逆运动学
        self.arm = casadi_ik.Kinematics("Wrist_Roll")
        self.arm.buildFromMJCF(arm_path)
        
        self.last_dof = None
        
    def runBefore(self):
        pass
    
    def runFunc(self):
        """主循环中执行的函数"""
        # 处理控制器输入
        self.controller.handle_input(self.arm)
        
        # 获取目标位置
        x, y, z = self.controller.get_position_target()
        print(f"当前位置: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        
        # 构建变换矩阵（保持末端执行器姿态不变）
        tf = self.build_transform_simple(x, y, z, np.pi / 4, 0, 0)
        
        # 求解逆运动学
        self.dof, info = self.arm.ik(tf, current_arm_motor_q=self.last_dof)
        self.last_dof = self.dof
        
        dof5_target = self.controller.get_dof5_target()
        if dof5_target is not None:
            self.dof[5] = dof5_target
        
        self.data.qpos[:6] = self.dof[:6]

        mujoco.mj_step(self.model, self.data)
        time.sleep(0.01)
    
    def build_transform_simple(self, x, y, z, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
            [-sp,   cp*sr,            cp*cr,            z],
            [0,     0,                0,                1]
        ])

if __name__ == "__main__":
    controller = XboxController()
    if not controller.is_connected():
        print("控制器连接失败，程序将退出。")
        exit(1)
    
    try:
        robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH, controller)
        robot.run_loop()
    finally:
        controller.cleanup()
