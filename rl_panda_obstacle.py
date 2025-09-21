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

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PandaObstacleEnv(gym.Env):
    def __init__(self, num_obstacles=5):
        super(PandaObstacleEnv, self).__init__()
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # 初始化Viewer
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30
        
        # 机械臂末端执行器ID
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
        
        # 障碍物和目标点配置
        self.num_obstacles = num_obstacles
        self.obstacle_size = 0.03  # 障碍物大小
        self.goal_size = 0.05  # 目标点大小
        self.obstacles = []  # 存储障碍物位置
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.8]
        }
        
        # 动作空间：7个关节的位置控制
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        
        # 计算观测空间大小：关节位置(7) + 末端执行器位置(3) + 目标位置(3) + 障碍物位置(num_obstacles*3)
        self.obs_size = 7 + 3 + 3 + num_obstacles * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,))
        
        # 目标位置
        self.goal = np.zeros(3)
        self.np_random = None
        
        # 初始化障碍物和目标点渲染
        self._generate_obstacles()
        self._render_goal()  # 渲染目标点

    def _get_random_position(self):
        """在工作空间内生成随机位置"""
        x = random.uniform(self.workspace['x'][0], self.workspace['x'][1])
        y = random.uniform(self.workspace['y'][0], self.workspace['y'][1])
        z = random.uniform(self.workspace['z'][0], self.workspace['z'][1])
        return np.array([x, y, z])
    
    def _generate_obstacles(self):
        """生成随机障碍物并添加到场景中"""
        # 清除现有障碍物
        self.obstacles = []
        
        # 生成新障碍物
        for _ in range(self.num_obstacles):
            # 确保障碍物不会太靠近目标或初始位置
            while True:
                pos = self._get_random_position()
                # 检查与初始位置的距离
                initial_pos = self.data.body(self.end_effector_id).xpos.copy()
                if np.linalg.norm(pos - initial_pos) > 0.3 and np.linalg.norm(pos - self.goal) > 0.3:
                    self.obstacles.append(pos)
                    break
        
        # 在场景中绘制障碍物和目标点
        self._render_scene_objects()
    
    def _render_scene_objects(self):
        """渲染所有场景对象：障碍物和目标点"""
        # 清除现有几何图形 (障碍物 + 目标点)
        self.handle.user_scn.ngeom = 0
        
        # 总共需要渲染的几何数量：障碍物 + 1个目标点
        total_geoms = self.num_obstacles + 1
        self.handle.user_scn.ngeom = total_geoms
        
        # 渲染障碍物
        for i in range(self.num_obstacles):
            pos = self.obstacles[i]
            # 随机颜色，确保可见
            rgba = np.random.rand(4)
            rgba[3] = 0.8  # 固定透明度
            if np.sum(rgba[:3]) < 0.5:  # 确保颜色不太暗
                rgba[:3] += 0.5
                rgba[:3] = np.clip(rgba[:3], 0, 1)
            
            # 初始化几何形状
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.obstacle_size, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
        
        # 渲染目标点（放在最后一个几何位置）
        self._render_goal()
    
    def _render_goal(self):
        """渲染目标点，使用特定颜色（蓝色）以便区分"""
        if self.handle.user_scn.ngeom > 0:
            # 目标点使用固定的蓝色，全透明度过半
            goal_rgba = np.array([0.1, 0.1, 0.9, 0.9])  # 蓝色
            
            # 目标点放在最后一个几何位置（障碍物数量之后）
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[self.num_obstacles],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.goal_size, 0, 0],
                pos=self.goal,
                mat=np.eye(3).flatten(),
                rgba=goal_rgba
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        
        # 重置模拟
        mujoco.mj_resetData(self.model, self.data)
        
        # 生成新目标
        self.goal = self.np_random.uniform(
            low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
            high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
        )
        print(f"新目标位置: {self.goal}")
        
        # 生成新障碍物和渲染场景
        self._generate_obstacles()
        
        # 构建观测
        obs = self._get_observation()
        # 验证观测形状是否正确
        assert obs.shape == (self.obs_size,), f"观测形状错误: {obs.shape} 应为 {self.obs_size,}"
        return obs, {}
    
    def _get_observation(self):
        """获取当前观测，包括关节位置、末端执行器位置、目标位置和障碍物位置"""
        joint_pos = self.data.qpos[:7].copy()
        end_effector_pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # 展平障碍物位置
        obstacles_flat = np.array(self.obstacles).flatten()
        
        # 组合观测：关节位置(7) + 末端执行器位置(3) + 目标位置(3) + 障碍物位置
        obs = np.concatenate([joint_pos, end_effector_pos, self.goal, obstacles_flat])
        return obs
    
    def _calculate_obstacle_penalty(self):
        """计算与障碍物的距离惩罚"""
        end_effector_pos = self.data.body(self.end_effector_id).xpos.copy()
        penalty = 0.0
        
        for obstacle_pos in self.obstacles:
            distance = np.linalg.norm(end_effector_pos - obstacle_pos)
            # 距离越近，惩罚越大
            if distance < self.obstacle_size + 0.05:  # 安全距离
                # 当进入危险区域时，施加较大惩罚
                penalty += 10.0 * (1.0 - (distance / (self.obstacle_size + 0.05)))
        
        return penalty
    
    def step(self, action):
        # 缩放动作到合适的关节范围
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7)
        for i in range(7):
            # 将[-1, 1]范围的动作映射到关节的实际范围
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 执行动作
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 获取末端执行器位置
        achieved_goal = self.data.body(self.end_effector_id).xpos.copy()
        
        # 计算奖励
        # 1. 到达目标的奖励
        distance_to_goal = np.linalg.norm(achieved_goal - self.goal)
        goal_reward = -distance_to_goal
        
        # 2. 避障惩罚
        obstacle_penalty = self._calculate_obstacle_penalty()
        
        # 3. 关节运动惩罚（鼓励平滑运动）
        action_penalty = 0.01 * np.linalg.norm(action)
        
        # 总奖励
        reward = goal_reward - obstacle_penalty - action_penalty
        
        # 检查是否到达目标
        terminated = distance_to_goal < 0.05  # 目标容忍度
        
        # 检查是否碰撞（与任何障碍物距离过近）
        collision = False
        for obstacle_pos in self.obstacles:
            if np.linalg.norm(achieved_goal - obstacle_pos) < self.obstacle_size + 0.02:
                collision = True
                break
        
        if collision:
            reward -= 5.0  # 碰撞惩罚
            terminated = True  # 碰撞后结束回合
        
        # 更新场景渲染（确保目标点位置正确）
        self._render_goal()
        
        # 构建观测
        obs = self._get_observation()
        # 验证观测形状是否正确
        assert obs.shape == (self.obs_size,), f"观测形状错误: {obs.shape} 应为 {self.obs_size,}"
        
        # 更新Viewer
        self.handle.sync()
        
        return obs, reward, terminated, False, {'is_success': terminated, 'distance': distance_to_goal}

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        if self.handle is not None:
            self.handle.close()


if __name__ == "__main__":
    # 创建环境，设置障碍物数量
    num_obstacles = 3
    env = make_vec_env(lambda: PandaObstacleEnv(num_obstacles=num_obstacles), n_envs=1)
    
    # 配置PPO算法参数
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])]
    )
    
    # 初始化模型
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/"
    )
    
    # 训练模型
    total_timesteps = 2048 * 200
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save("panda_ppo_obstacle_avoidance")
    print(f"模型已保存，训练步数: {total_timesteps}")
    