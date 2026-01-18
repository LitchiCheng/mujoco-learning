import mujoco
import time
import src.mujoco_viewer as mujoco_viewer as mujoco_viewer
import numpy as np

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=-45, elevation=-30)
        self.path = path
    
    def runBefore(self):
        for body_id in range(self.model.nbody):
            # 参数说明：model=模型，obj_type=对象类型（body），obj_id=body ID
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # 获取父 body ID
            parent_body_id = self.model.body_parentid[body_id]
            # print(f"{body_id:<10} {body_name:<20} {parent_body_id:<15}") 
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat
            print(f"id:{body_id}, name: {body_name}, Position: {pos}, Quaternion: {quat}")
            
    def runFunc(self):
        placeholder = 0

test = Test("./model/franka_emika_panda/scene_pos.xml")
test.run_loop()