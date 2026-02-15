import time
import mujoco
import mujoco.viewer
from xml.etree import ElementTree as ET
from io import StringIO
import numpy as np
import src.utils as utils
import glfw
import cv2

class CustomViewer:
    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.handle = None
        self.has_inited_glfw = False

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport
    
    def initGlfw(self, width, height):
        if not self.has_inited_glfw:
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.glfw_window = glfw.create_window(width, height, "Offscreen", None, None)
            glfw.make_context_current(self.glfw_window)

            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            # 创建帧缓冲对象
            self.framebuffer = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            self.camera_view = mujoco.MjvCamera()
            self.has_inited_glfw = True

    
    def setTimestep(self, timestep):
        self.model.opt.timestep = timestep
    
    def addVisuGeom(self, geoms_pos:np.ndarray, geoms_type:list, geoms_size:np.ndarray, geoms_rgba:np.ndarray):
        now_user_geom_num = self.handle.user_scn.ngeom
        self.handle.user_scn.ngeom = 0
        total_geoms = now_user_geom_num + len(geoms_pos)
        self.handle.user_scn.ngeom = total_geoms

        for i in range(len(geoms_pos)):
            pos = geoms_pos[i]
            rgba = geoms_rgba[i]
            size = geoms_size[i]
            if len(size) < 3:
                if len(size) < 2:
                    size = np.concatenate([size, [0.0, 0.0]])
                else:
                    size = np.concatenate([size, [0.0]])
            ob_type_str = geoms_type[i]
            ob_type = mujoco.mjtGeom.mjGEOM_SPHERE
            if ob_type_str == "sphere":
                ob_type = mujoco.mjtGeom.mjGEOM_SPHERE
            elif ob_type_str == "box":
                ob_type = mujoco.mjtGeom.mjGEOM_BOX
            elif ob_type_str == "capsule":
                ob_type = mujoco.mjtGeom.mjGEOM_CAPSULE
            elif ob_type_str == "cylinder":
                ob_type = mujoco.mjtGeom.mjGEOM_CYLINDER
            elif ob_type_str == "ellipsoid":
                ob_type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            elif ob_type_str == "mesh":
                ob_type = mujoco.mjtGeom.mjGEOM_MESH
            else:
                raise ValueError(f"Unsupported geom type: {ob_type_str}")
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i+now_user_geom_num],
                type = ob_type,
                size = size, 
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
    
    def addObstacles(self, obstacles_pos:np.ndarray, obstacles_type:list, obstacles_size:np.ndarray, obstacles_rgba:np.ndarray):
        """
        Add obstacles to the model.
        :param obstacles_pos: (n, 3) array of obstacle positions
        :param obstacles_size: (n, 3) array of obstacle size
        :param obstacles_rgba: (n,4) array of obstacle color
        """
        self.original_model = self.model  # 保存原始模型
        self.num_obstacles = len(obstacles_pos)

        # 解析 XML 树（root 是 XML 的根节点）
        tree = ET.parse(self.model_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("原始 XML 中未找到 <worldbody> 标签，请检查 MuJoCo 模型格式")

        for i in range(self.num_obstacles):
            pos = obstacles_pos[i]
            rgba = obstacles_rgba[i]
            size = obstacles_size[i]
            ob_type = obstacles_type[i]
            
            # 创建 <geom> 节点，添加到 <worldbody> 下
            obstacle_geom = ET.SubElement(worldbody, "geom")
            obstacle_geom.set("name", f"obstacle_{i}")
            obstacle_geom.set("type", ob_type)  # 几何类型
            obstacle_geom.set("size", " ".join(f"{x:.3f}" for x in size))
            obstacle_geom.set("pos", f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")  # 位置（保留3位小数）
            obstacle_geom.set("contype", "1")  # 碰撞类型（必须，与 conaffinity 匹配）
            obstacle_geom.set("conaffinity", "1")  # 碰撞亲和性（必须，1=可与同类碰撞）
            # obstacle_geom.set("active", "1")  # 启用碰撞/渲染（1=激活）
            obstacle_geom.set("mass", "0.0")  # 静态物体（质量=0，不会被推动）
            obstacle_geom.set("rgba", f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}")

        new_xml_path = self.model_path.replace(".xml", "_with_obstacles.xml")
        tree.write(new_xml_path, encoding="utf-8", xml_declaration=True)
    
        self.model = mujoco.MjModel.from_xml_path(new_xml_path)
        self.data = mujoco.MjData(self.model)

        print(f"原始 Geom 数：{self.original_model.ngeom}")
        print(f"新模型 Geom 数：{self.model.ngeom}")
        
    def getBodyIdsByName(self):
        map = {}
        for body_id in range(self.model.nbody):
            # 参数说明：model=模型，obj_type=对象类型（body），obj_id=body ID
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # 获取父 body ID
            parent_body_id = self.model.body_parentid[body_id]
            map[body_name] = body_id
        return map
    
    def getBodyNames(self):
        return list(self.getBodyIdsByName().keys())
    
    def getBodyIdByName(self, name):
        return self.getBodyIdsByName()[name]
    
    def getGeomIdByName(self, geom_name):
        """根据geom名称获取其索引"""
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == geom_name:
                return i
        return -1
    
    def setGeomPositionByName(self, geom_name, position):
        """根据geom名称设置其位置"""
        geom_id = self.getGeomIdByName(geom_name)
        if geom_id == -1:
            raise ValueError(f"未找到geom名称为{geom_name}的geom")
        self.model.geom_pos[geom_id] = position.copy()
        mujoco.mj_forward(self.model, self.data)
    
    def getGeomPositionByName(self, geom_name):
        """根据geom名称获取其位置"""
        geom_id = self.getGeomIdByName(geom_name)
        if geom_id == -1:
            raise ValueError(f"未找到geom名称为{geom_name}的geom")
        return self.data.geom_pos[geom_id].copy()

    def getBodyPositionByName(self, name):
        body_id = self.getBodyIdByName(name)
        return self.data.body(body_id).xpos.copy()
    
    def getBodyQuatByName(self, name):
        body_id = self.getBodyIdByName(name)
        return self.data.body(body_id).xquat.copy()
    
    def getBodyPoseByName(self, name):
        position = self.getBodyPositionByName(name)
        quat = self.getBodyQuatByName(name)
        return np.concatenate([position, quat])
    
    def getBodyPoseEulerByName(self, name):
        position = self.getBodyPositionByName(name)
        quat = self.getBodyQuatByName(name)
        euler = utils.quat2euler(quat)
        return np.concatenate([position, euler])
    
    def getContactInfo(self):
        info = {}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 获取几何体对应的body_id
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            # 通过mj_id2name转换body_id为名称
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            info["pair"+str(i)] = {}
            info["pair"+str(i)]["geom1"] = contact.geom1
            info["pair"+str(i)]["geom2"] = contact.geom2
            info["pair"+str(i)]["pos"] = contact.pos.copy()
            info["pair"+str(i)]["body1_name"] = body1_name
            info["pair"+str(i)]["body2_name"] = body2_name
        return info

    def getTrackingCameraImage(self, body_name="ee_center_body", width=640, height=480, distance=0, fix_azimuth=None, fix_elevation=None):
        self.initGlfw(width, height)
        tracking_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        yaw = self.getBodyPoseEulerByName(body_name)[3]
        pitch = self.getBodyPoseEulerByName(body_name)[4]
        self.camera_view.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera_view.trackbodyid = tracking_body_id
        self.camera_view.distance = distance  # 相机与目标的距离
        if fix_azimuth is not None:
            self.camera_view.azimuth = fix_azimuth
        else:
            self.camera_view.azimuth = yaw    # 水平方位角（度）
        if fix_elevation is not None:
            self.camera_view.elevation = fix_elevation
        else:
            self.camera_view.elevation = pitch # 俯仰角（度）
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), 
                            mujoco.MjvPerturb(), self.camera_view, 
                            mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow('MuJoCo Camera Output', bgr)
        if cv2.waitKey(1) == 27:
            return

    def run_loop(self):
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = self.distance
        self.handle.cam.azimuth = self.azimuth
        self.handle.cam.elevation = self.elevation
        self.runBefore()
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            self.runFunc()
            mujoco.mj_step(self.model, self.data)
            self.sync()
            time.sleep(self.model.opt.timestep)
    
    def runBefore(self):
        pass

    def runFunc(self):
        pass