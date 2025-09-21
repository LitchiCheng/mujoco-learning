import mujoco
import time
import mujoco_viewer
import numpy as np
import math
import zmq
import json

class ZMQCommunicator:
    """ZMQ通信封装类，负责处理与外部的通信"""
    
    def __init__(self, ipc_path="ipc:///tmp/robot_arm_comm.ipc"):
        """初始化ZMQ通信
        
        Args:
            ipc_path: IPC通信路径，默认为"ipc:///tmp/robot_arm_comm.ipc"
        """
        self.ipc_path = ipc_path
        self.context = None
        self.socket = None
        self._initialize()
    
    def _initialize(self):
        """初始化ZMQ上下文和socket"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(self.ipc_path)
            print(f"ZMQ发布者启动，绑定到：{self.ipc_path}")
        except zmq.ZMQError as e:
            print(f"ZMQ初始化失败: {e}")
            self.cleanup()
    
    def send_data(self, actions:dict):
        """发送关节位置数据
        
        Args:
            joint_pos: 关节位置数据（角度）
            control_mode: 控制模式，默认为"position"
        """
        if not self.socket:
            print("ZMQ socket未初始化，无法发送数据")
            return
            
        try:
            # 转换为JSON字符串并发布
            json_data = json.dumps(actions)
            self.socket.send_string(json_data)
        except zmq.ZMQError as e:
            print(f"发送数据失败: {e}")
    
    def cleanup(self):
        """清理ZMQ资源"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("ZMQ资源已释放")


class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path, communicator):
        super().__init__(path, 1.5, azimuth=135, elevation=-30)
        self.path = path
        self.communicator = communicator  # 注入通信器实例
    
    def runBefore(self):
        pass
    
    def runFunc(self):
        sim_joint_rad = self.data.qpos[:6].copy()
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        self.communicator.send_data(sim_joint_deg)  # 使用通信器发送数据
        time.sleep(0.01)  # 控制发送频率


if __name__ == "__main__":
    # 创建通信器实例
    zmq_communicator = ZMQCommunicator()
    
    try:
        # 将通信器实例传入Test类
        test = Test("./model/trs_so_arm100/scene.xml", zmq_communicator)
        test.run_loop()
    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()
