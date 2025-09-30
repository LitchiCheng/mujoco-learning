import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import random

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PandaObstacleEnv(gym.Env):
    def __init__(self, num_obstacles=1):
        super(PandaObstacleEnv, self).__init__()
        # 1. 加载MuJoCo机械臂模型（需确保scene.xml路径正确）
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # 2. 初始化Viewer（被动模式，不阻塞训练）
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        # 调整相机视角（聚焦机械臂工作空间）
        self.handle.cam.distance = 3.0
        self.handle.cam.azimuth = 0.0
        self.handle.cam.elevation = -30.0
        self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        # 3. 机械臂关键参数
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()  # 固定起点
        
        # 4. 障碍物与目标配置（核心：路径约束）
        self.num_obstacles = num_obstacles
        self.obstacle_size = 0.04          # 障碍物半径
        self.goal_size = 0.03            # 目标点半径
        self.obstacles = []               # 障碍物实时位置列表
        self.path_offset_range = 0.4     # 障碍物偏离路径的最大距离（小范围）
        self.min_safety_dist = 0.3        # 障碍物与起点/终点的最小安全距离
        self.obstacle_disturb_prob = 0.1  # step中障碍物扰动概率
        self.obstacle_disturb_step = 0.02 # 扰动步长（小幅度避免突变）
        
        # 5. 工作空间（限制目标点生成，确保机械臂可达）
        # self.workspace = {
        #     'x': [-0.5, 0.8],
        #     'y': [-0.5, 0.5],
        #     'z': [0.05, 0.8]
        # }

        # 约束下工作空间
        self.workspace = {
            'x': [0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }
        
        # 6. 动作空间与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # 7关节控制
        self.obs_size = 7 + 3 + 3 + self.num_obstacles * 3  # 观测维度计算
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        # 7. 其他初始化
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(7, dtype=np.float32)  # 初始化上一步动作为零向量
        # 新增：Panda机械臂安全home位姿（7个关节角度，单位：弧度）
        # 对应末端位置：X≈0.5, Y≈0, Z≈0.4（悬空，远离地面）
        self.home_joint_pos = np.array([
            0.0, -np.pi/4, 0.0, -3*np.pi/4, 
            0.0, np.pi/2, np.pi/4
        ], dtype=np.float32)

    def _get_valid_goal(self):
        """生成有效目标点（与起点距离>0.6，确保路径有意义）"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if np.linalg.norm(goal - self.initial_ee_pos) > 0.6:
                return goal.astype(np.float32)

    def _sample_path_point(self, start, end):
        """在起点→终点的线段上采样中间点（避开两端）"""
        t = self.np_random.uniform(low=0.2, high=0.8)
        return start + t * (end - start)

    def _add_path_offset(self, point):
        """给路径点添加小范围偏移（确保障碍物在路径附近）"""
        offset = self.np_random.normal(loc=0.0, scale=self.path_offset_range/3, size=3)
        offset = np.clip(offset, -self.path_offset_range, self.path_offset_range)
        return point + offset.astype(np.float32)

    def _generate_path_obstacles(self):
        """核心：在目标路径上生成小范围随机障碍物"""
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                obs_pos = self._add_path_offset(path_base)
                # 安全检查：避开起点和目标
                dist_to_start = np.linalg.norm(obs_pos - self.initial_ee_pos)
                dist_to_goal = np.linalg.norm(obs_pos - self.goal)
                if dist_to_start > self.min_safety_dist and dist_to_goal > self.min_safety_dist:
                    self.obstacles.append(obs_pos)
                    break
        self._render_scene()

    def _disturb_obstacles(self):
        """step中随机扰动障碍物（增加训练泛化性）"""
        if self.np_random.random() < self.obstacle_disturb_prob:
            for i in range(self.num_obstacles):
                disturb = self.np_random.normal(loc=0.0, scale=self.obstacle_disturb_step/3, size=3)
                disturb = np.clip(disturb, -self.obstacle_disturb_step, self.obstacle_disturb_step)
                self.obstacles[i] += disturb
                # 重新约束到路径附近
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                self.obstacles[i] = self._add_path_offset(path_base)
            self._render_scene()

    def _render_scene(self):
        """渲染障碍物（随机颜色）和目标点（固定蓝色）"""
        self.handle.user_scn.ngeom = 0  # 清除现有几何
        total_geoms = self.num_obstacles + 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染障碍物
        for i in range(self.num_obstacles):
            pos = self.obstacles[i]
            rgba = self.np_random.uniform(low=0.3, high=1.0, size=4)
            rgba[3] = 0.8  # 半透明
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.obstacle_size, 0.0, 0.0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[self.num_obstacles],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed=None, options=None):
        """环境重置：新目标+新障碍物+重置模拟"""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
         # 1. 重置MuJoCo数据（关节回到URDF默认值）
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. 强制将关节设置为安全home位姿（关键修改）
        self.data.qpos[:7] = self.home_joint_pos  # 覆盖默认关节角度
        mujoco.mj_forward(self.model, self.data)  # 刷新MuJoCo状态，计算末端位置
        
        # 3. 更新初始末端位置（此时已为home位姿的末端位置，远离地面）
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        print(f"[重置] 初始末端位置: {np.round(self.initial_ee_pos, 3)}")  # 验证：Z应≈0.4
        
        self.goal = self._get_valid_goal()
        print(f"[重置] 新目标位置: {np.round(self.goal, 3)}")
        self._generate_path_obstacles()  # 生成新障碍物
        
        obs = self._get_observation()
        assert obs.shape == (self.obs_size,), f"观测形状错误：{obs.shape} → 预期{self.obs_size,}"
        return obs, {}

    def _get_observation(self):
        """构建观测向量：关节位置+末端位置+目标+障碍物（展平）"""
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        obstacles_flat = np.concatenate(self.obstacles).astype(np.float32)
        return np.concatenate([joint_pos, ee_pos, self.goal, obstacles_flat])

    def _calc_reward(self, ee_pos, action):
        """计算奖励：目标引导+避障+动作平滑"""
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        goal_reward = -dist_to_goal  # 距离越近奖励越高
        
        # 避障惩罚
        obstacle_penalty = 0.0
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(ee_pos - obs_pos)
            if dist_to_obs < self.obstacle_size + 0.05:
                obstacle_penalty += 10.0 * (1.0 - (dist_to_obs / (self.obstacle_size + 0.05)))
        
        # 动作平滑惩罚（改为惩罚动作变化率）
        action_diff = action - self.prev_action  # 当前动作与上一步动作的差值
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)  # 差值越大，惩罚越重
        
        # 更新上一步动作为当前动作（用于下一步计算）
        self.prev_action = action.copy()
        
        return goal_reward - obstacle_penalty - smooth_penalty, dist_to_goal

    def step(self, action):
        """执行动作+更新状态+返回step结果"""
        # 1. 动作缩放（映射到关节实际范围）
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 2. 执行动作（单步模拟）
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 每个epsonde随机扰动一次障碍物（可选）
        # # 3. 扰动障碍物（可选）
        # self._disturb_obstacles()
        
        # 4. 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos
        reward, dist_to_goal = self._calc_reward(ee_pos, action)
        terminated = False
        collision = False
        
        # 目标达成（距离<0.05）
        if dist_to_goal < 0.05:
            reward += 10.0  # 额外奖励
            terminated = True
        
        # 碰撞检测（距离<障碍物半径+0.02）
        for obs_pos in self.obstacles:
            if np.linalg.norm(ee_pos - obs_pos) < self.obstacle_size + 0.02:
                collision = True
                break
        # print(ee_pos)
        # 低于地面
        if ee_pos[2] < 0.05:
            collision = True
            print(f"[碰撞] 末端碰地面，Z={ee_pos[2]:.3f}")
        if collision:
            reward -= 15.0  # 严厉惩罚
            terminated = True
        
        # 5. 更新渲染与观测
        self.handle.sync()
        obs = self._get_observation()
        
        # 6. 额外信息
        info = {
            'is_success': terminated and (dist_to_goal < 0.05),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        """释放Viewer资源"""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


if __name__ == "__main__":
    # -------------------------- 训练配置 --------------------------
    NUM_OBSTACLES = 1  # 障碍物数量（建议从1开始）
    ENV_KWARGS = {'num_obstacles': NUM_OBSTACLES}
    
    # 1. 创建向量环境（n_envs=1，单环境训练）
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=1,
        seed=42  # 固定种子确保可复现
    )
    
    # 2. PPO策略网络配置
    POLICY_KWARGS = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[1024, 256, 128], vf=[1024, 256, 128])]  # 策略/价值网络结构
    )
    
    # 3. 初始化PPO模型（参数完整，无截断）
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/panda_obstacle/"  # TensorBoard日志路径（可选）
    )
    
    # 4. 开始训练
    TOTAL_TIMESTEPS = 2048 * 10000  # 总训练步数（102.4万步，可根据需求调整）
    print(f"开始训练，总步数: {TOTAL_TIMESTEPS}")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True  # 显示训练进度条（需安装tqdm：pip install tqdm）
    )
    
    # 5. 保存训练好的模型
    MODEL_SAVE_PATH = "panda_ppo_obstacle_avoidance"
    model.save(MODEL_SAVE_PATH)
    print(f"模型已保存至: {MODEL_SAVE_PATH}")
    
    # 6. 关闭环境与资源
    env.close()
    print("训练结束，所有资源已释放")