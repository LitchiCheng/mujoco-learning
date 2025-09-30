# 测试脚本开头的导入区域，需包含以下内容
import numpy as np
import mujoco
import gym  # 必须导入gym
from gym import spaces  # 明确导入spaces，避免NameError
import torch
import mujoco.viewer
from stable_baselines3 import PPO
import time

# 导入自定义环境类（需与训练时的环境定义保持一致）
class PandaObstacleEnv(gym.Env):
    def __init__(self, num_obstacles=1, render_mode="human"):
        super(PandaObstacleEnv, self).__init__()
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # 初始化Viewer
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3.0
        self.handle.cam.azimuth = 0.0
        self.handle.cam.elevation = -30.0
        self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        # 机械臂参数
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # 障碍物与目标配置
        self.num_obstacles = num_obstacles
        self.obstacle_size = 0.1
        self.goal_size = 0.05
        self.obstacles = []
        self.path_offset_range = 0.15
        self.min_safety_dist = 0.3
        self.obstacle_disturb_prob = 0.1  # 测试时可减小扰动概率
        self.obstacle_disturb_step = 0.02
        
        # 工作空间
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.8]
        }
        
        # 动作与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.obs_size = 7 + 3 + 3 + self.num_obstacles * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        # 其他初始化
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.render_mode = render_mode

    # 以下方法与训练环境中的实现完全相同
    def _get_valid_goal(self):
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if np.linalg.norm(goal - self.initial_ee_pos) > 0.6:
                return goal.astype(np.float32)

    def _sample_path_point(self, start, end):
        t = self.np_random.uniform(low=0.2, high=0.8)
        return start + t * (end - start)

    def _add_path_offset(self, point):
        offset = self.np_random.normal(loc=0.0, scale=self.path_offset_range/3, size=3)
        offset = np.clip(offset, -self.path_offset_range, self.path_offset_range)
        return point + offset.astype(np.float32)

    def _generate_path_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                obs_pos = self._add_path_offset(path_base)
                dist_to_start = np.linalg.norm(obs_pos - self.initial_ee_pos)
                dist_to_goal = np.linalg.norm(obs_pos - self.goal)
                if dist_to_start > self.min_safety_dist and dist_to_goal > self.min_safety_dist:
                    self.obstacles.append(obs_pos)
                    break
        self._render_scene()

    def _disturb_obstacles(self):
        if self.np_random.random() < self.obstacle_disturb_prob:
            for i in range(self.num_obstacles):
                disturb = self.np_random.normal(loc=0.0, scale=self.obstacle_disturb_step/3, size=3)
                disturb = np.clip(disturb, -self.obstacle_disturb_step, self.obstacle_disturb_step)
                self.obstacles[i] += disturb
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                self.obstacles[i] = self._add_path_offset(path_base)
            self._render_scene()

    def _render_scene(self):
        self.handle.user_scn.ngeom = 0
        total_geoms = self.num_obstacles + 1
        self.handle.user_scn.ngeom = total_geoms

        for i in range(self.num_obstacles):
            pos = self.obstacles[i]
            rgba = np.array([0.8, 0.3, 0.3, 0.8])  # 测试时用固定的红色障碍物
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.obstacle_size, 0.0, 0.0],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=rgba
            )

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
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.goal = self._get_valid_goal()
        print(f"[测试] 新目标位置: {np.round(self.goal, 3)}")
        self._generate_path_obstacles()
        
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        obstacles_flat = np.concatenate(self.obstacles).astype(np.float32)
        return np.concatenate([joint_pos, ee_pos, self.goal, obstacles_flat])

    def _calc_reward(self, ee_pos, action):
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        goal_reward = -dist_to_goal
        
        obstacle_penalty = 0.0
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(ee_pos - obs_pos)
            if dist_to_obs < self.obstacle_size + 0.05:
                obstacle_penalty += 10.0 * (1.0 - (dist_to_obs / (self.obstacle_size + 0.05)))
        
        action_penalty = 0.01 * np.linalg.norm(action)
        return goal_reward - obstacle_penalty - action_penalty, dist_to_goal

    def step(self, action):
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 测试时可以减小障碍物扰动，更清晰观察模型表现
        self._disturb_obstacles()
        
        ee_pos = self.data.body(self.end_effector_id).xpos
        reward, dist_to_goal = self._calc_reward(ee_pos, action)
        terminated = False
        collision = False
        
        if dist_to_goal < 0.05:
            reward += 10.0
            terminated = True
        
        for obs_pos in self.obstacles:
            if np.linalg.norm(ee_pos - obs_pos) < self.obstacle_size + 0.02:
                collision = True
                break
        if collision:
            reward -= 15.0
            terminated = True
        
        if self.render_mode == "human":
            self.handle.sync()
            # 测试时增加延迟，便于观察动作
            time.sleep(0.01)
        
        obs = self._get_observation()
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
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        print("测试结束，环境已关闭")


def test_model(model_path, num_episodes=5, num_obstacles=1):
    """
    测试训练好的模型
    
    参数:
        model_path: 模型保存路径
        num_episodes: 测试回合数
        num_obstacles: 障碍物数量（需与训练时一致）
    """
    # 创建测试环境
    env = PandaObstacleEnv(num_obstacles=num_obstacles, render_mode="human")
    
    # 加载模型
    model = PPO.load(model_path, env=env)
    print(f"已加载模型: {model_path}")
    
    # 记录测试结果
    success_count = 0
    collision_count = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n=== 测试回合 {episode + 1}/{num_episodes} ===")
        obs, _ = env.reset(seed=episode)  # 固定种子，便于复现
        terminated = False
        step = 0
        
        while not terminated:
            # 使用模型预测动作（确定性策略，确保结果稳定）
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            step += 1
            
            # 每10步打印一次状态
            if step % 10 == 0:
                print(f"步骤 {step}: 距离目标 {info['distance_to_goal']:.3f}m, 碰撞: {info['collision']}")
        
        # 记录结果
        episode_lengths.append(step)
        if info['is_success']:
            success_count += 1
            print(f"回合完成，步数: {step}, 成功到达目标!")
        elif info['collision']:
            collision_count += 1
            print(f"回合完成，步数: {step}, 发生碰撞!")
        else:
            print(f"回合完成，步数: {step}, 未到达目标!")
    
    # 打印测试总结
    print("\n=== 测试总结 ===")
    print(f"总回合数: {num_episodes}")
    print(f"成功到达目标: {success_count} ({success_count/num_episodes*100:.1f}%)")
    print(f"发生碰撞: {collision_count} ({collision_count/num_episodes*100:.1f}%)")
    print(f"平均步数: {np.mean(episode_lengths):.1f}")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    # 模型路径（需与训练时保存的路径一致）
    MODEL_PATH = "panda_ppo_obstacle_avoidance"
    
    # 测试参数
    NUM_EPISODES = 10  # 测试10个回合
    NUM_OBSTACLES = 1  # 障碍物数量（需与训练时一致）
    
    # 运行测试
    test_model(
        model_path=MODEL_PATH,
        num_episodes=NUM_EPISODES,
        num_obstacles=NUM_OBSTACLES
    )
