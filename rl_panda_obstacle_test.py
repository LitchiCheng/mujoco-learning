import numpy as np
import mujoco
import gym
from gym import spaces
import torch.nn as nn
import torch
import mujoco.viewer
import random
from stable_baselines3 import PPO

# 确保测试环境与训练环境一致
class PandaObstacleEnv(gym.Env):
    def __init__(self, num_obstacles=5, render_mode="human"):
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
        self.obstacle_size = 0.08  # 障碍物大小
        self.goal_size = 0.05  # 目标点大小
        self.obstacles = []  # 存储障碍物位置
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.8]
        }
        
        # 动作空间：7个关节的位置控制
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        
        # 计算观测空间大小
        self.obs_size = 7 + 3 + 3 + num_obstacles * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,))
        
        # 目标位置
        self.goal = np.zeros(3)
        self.np_random = None
        
        # 初始化障碍物和目标点渲染
        self._generate_obstacles()
        self._render_goal()

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
            while True:
                pos = self._get_random_position()
                initial_pos = self.data.body(self.end_effector_id).xpos.copy()
                if np.linalg.norm(pos - initial_pos) > 0.3 and np.linalg.norm(pos - self.goal) > 0.3:
                    self.obstacles.append(pos)
                    break
        
        # 在场景中绘制障碍物和目标点
        self._render_scene_objects()
    
    def _render_scene_objects(self):
        """渲染所有场景对象：障碍物和目标点"""
        # 清除现有几何图形
        self.handle.user_scn.ngeom = 0
        
        # 总共需要渲染的几何数量：障碍物 + 1个目标点
        total_geoms = self.num_obstacles + 1
        self.handle.user_scn.ngeom = total_geoms
        
        # 渲染障碍物
        for i in range(self.num_obstacles):
            pos = self.obstacles[i]
            rgba = np.random.rand(4)
            rgba[3] = 0.8  # 固定透明度
            if np.sum(rgba[:3]) < 0.5:
                rgba[:3] += 0.5
                rgba[:3] = np.clip(rgba[:3], 0, 1)
            
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.obstacle_size, 0, 0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )
        
        # 渲染目标点
        self._render_goal()
    
    def _render_goal(self):
        """渲染目标点，使用特定颜色（蓝色）以便区分"""
        if self.handle.user_scn.ngeom > 0:
            goal_rgba = np.array([0.1, 0.1, 0.9, 0.9])  # 蓝色
            
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
        return obs, {}
    
    def _get_observation(self):
        """获取当前观测"""
        joint_pos = self.data.qpos[:7].copy()
        end_effector_pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # 展平障碍物位置
        obstacles_flat = np.array(self.obstacles).flatten()
        
        # 组合观测
        obs = np.concatenate([joint_pos, end_effector_pos, self.goal, obstacles_flat])
        return obs
    
    def _calculate_obstacle_penalty(self):
        """计算与障碍物的距离惩罚"""
        end_effector_pos = self.data.body(self.end_effector_id).xpos.copy()
        penalty = 0.0
        
        for obstacle_pos in self.obstacles:
            distance = np.linalg.norm(end_effector_pos - obstacle_pos)
            if distance < self.obstacle_size + 0.05:
                penalty += 10.0 * (1.0 - (distance / (self.obstacle_size + 0.05)))
        
        return penalty
    
    def step(self, action):
        # 缩放动作到合适的关节范围
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 执行动作
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 获取末端执行器位置
        achieved_goal = self.data.body(self.end_effector_id).xpos.copy()
        
        # 计算奖励
        distance_to_goal = np.linalg.norm(achieved_goal - self.goal)
        goal_reward = -distance_to_goal
        obstacle_penalty = self._calculate_obstacle_penalty()
        action_penalty = 0.01 * np.linalg.norm(action)
        reward = goal_reward - obstacle_penalty - action_penalty
        
        # 检查是否到达目标
        terminated = distance_to_goal < 0.05
        
        # 检查是否碰撞
        collision = False
        for obstacle_pos in self.obstacles:
            if np.linalg.norm(achieved_goal - obstacle_pos) < self.obstacle_size + 0.02:
                collision = True
                break
        
        if collision:
            reward -= 5.0
            terminated = True
        
        # 更新场景渲染
        self._render_goal()
        
        # 构建观测
        obs = self._get_observation()
        
        # 更新Viewer
        self.handle.sync()
        
        return obs, reward, terminated, False, {'is_success': terminated, 'distance': distance_to_goal}

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        if self.handle is not None:
            self.handle.close()


def test_model(model_path, num_episodes=5, num_obstacles=5):
    """测试训练好的模型
    
    参数:
        model_path: 模型文件路径
        num_episodes: 测试回合数
        num_obstacles: 障碍物数量
    """
    # 创建测试环境
    env = PandaObstacleEnv(num_obstacles=num_obstacles)
    
    # 加载模型
    model = PPO.load(model_path, env=env)
    print(f"成功加载模型: {model_path}")
    
    # 存储测试结果
    success_count = 0
    collision_count = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n=== 测试回合 {episode + 1}/{num_episodes} ===")
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step = 0
        collision = False
        
        while not terminated and not truncated:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)  # deterministic=True表示使用确定性策略
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # 每10步打印一次信息
            if step % 10 == 0:
                print(f"步骤: {step}, 距离目标: {info['distance']:.4f}")
            
            # 控制仿真速度，方便观察
            time.sleep(0.05)
        
        # 记录结果
        episode_lengths.append(step)
        if info['is_success']:
            success_count += 1
            print(f"回合成功！使用了 {step} 步")
        if 'collision' in info and info['collision']:
            collision_count += 1
            print(f"回合失败：发生碰撞，使用了 {step} 步")
    
    # 打印测试总结
    print("\n=== 测试总结 ===")
    print(f"总回合数: {num_episodes}")
    print(f"成功回合数: {success_count}")
    print(f"成功率: {success_count/num_episodes:.2%}")
    print(f"平均步数: {np.mean(episode_lengths):.2f}")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    import time
    # 测试训练好的模型，默认测试5个回合
    test_model(
        model_path="panda_ppo_obstacle_avoidance",  # 模型文件路径
        num_episodes=5,                            # 测试回合数
        num_obstacles=5                            # 障碍物数量，与训练时保持一致
    )
    