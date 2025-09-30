import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # 新增DummyVecEnv（单进程测试用）
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import random
import time  # 测试时控制帧率
from typing import Optional  # 类型提示

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")


class PandaObstacleEnv(gym.Env):
    # 新增visualize参数：True=开启可视化，False=关闭可视化（默认关闭）
    def __init__(self, num_obstacles: int = 1, visualize: bool = False):
        super(PandaObstacleEnv, self).__init__()
        self.visualize = visualize  # 保存可视化开关状态
        self.handle = None  # 初始化Viewer句柄为None（避免未创建时调用）

        # 1. 加载MuJoCo机械臂模型
        self.model = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # 2. 仅当visualize=True时，创建Viewer并调整视角
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        # 3. 机械臂关键参数（不变）
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32)  # 初始位置将在reset中更新
        self.home_joint_pos = np.array([  # 安全home位姿
            0.0, -np.pi/4, 0.0, -3*np.pi/4, 
            0.0, np.pi/2, np.pi/4
        ], dtype=np.float32)
        
        # 4. 障碍物与目标配置（不变）
        self.num_obstacles = num_obstacles
        self.obstacle_size = 0.04
        self.goal_size = 0.03
        self.obstacles = []
        self.path_offset_range = 0.2
        self.min_safety_dist = 0.3
        self.obstacle_disturb_prob = 0.1
        self.obstacle_disturb_step = 0.02
        
        # 5. 约束工作空间（不变）
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }
        
        # 6. 动作空间与观测空间（不变）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.obs_size = 7 + 3 + 3 + self.num_obstacles * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        
        # 7. 其他初始化（不变）
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(7, dtype=np.float32)

    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点（优化：增加工作空间内距离约束，避免无解）"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            # 调整距离约束：适配缩小的工作空间（原>0.6可能无解）
            if 0.2 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5:
                return goal.astype(np.float32)

    def _sample_path_point(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        t = self.np_random.uniform(low=0.2, high=0.8)
        return start + t * (end - start)

    def _add_path_offset(self, point: np.ndarray) -> np.ndarray:
        offset = self.np_random.normal(loc=0.0, scale=self.path_offset_range/3, size=3)
        offset = np.clip(offset, -self.path_offset_range, self.path_offset_range)
        return point + offset.astype(np.float32)

    def _generate_path_obstacles(self) -> None:
        """仅当visualize=True时，才执行渲染（避免无Viewer时报错）"""
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
        # 关键：仅可视化开启时渲染障碍物/目标
        if self.visualize and self.handle is not None:
            self._render_scene()

    def _disturb_obstacles(self) -> None:
        if self.np_random.random() < self.obstacle_disturb_prob:
            for i in range(self.num_obstacles):
                disturb = self.np_random.normal(loc=0.0, scale=self.obstacle_disturb_step/3, size=3)
                disturb = np.clip(disturb, -self.obstacle_disturb_step, self.obstacle_disturb_step)
                self.obstacles[i] += disturb
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                self.obstacles[i] = self._add_path_offset(path_base)
            # 仅可视化开启时重新渲染
            if self.visualize and self.handle is not None:
                self._render_scene()

    def _render_scene(self) -> None:
        """渲染障碍物和目标点（仅在visualize=True时执行）"""
        if not self.visualize or self.handle is None:
            return
        # 清除现有几何
        self.handle.user_scn.ngeom = 0
        total_geoms = self.num_obstacles + 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染障碍物
        for i in range(self.num_obstacles):
            pos = self.obstacles[i]
            rgba = self.np_random.uniform(low=0.3, high=1.0, size=4)
            rgba[3] = 0.8
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # 生成目标和障碍物
        self.goal = self._get_valid_goal()
        if self.visualize:  # 仅可视化时打印重置信息
            print(f"[重置] 初始末端位置: {np.round(self.initial_ee_pos, 3)}")
            print(f"[重置] 新目标位置: {np.round(self.goal, 3)}")
        self._generate_path_obstacles()
        
        obs = self._get_observation()
        assert obs.shape == (self.obs_size,), f"观测形状错误：{obs.shape} → 预期{self.obs_size,}"
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        obstacles_flat = np.concatenate(self.obstacles).astype(np.float32)
        return np.concatenate([joint_pos, ee_pos, self.goal, obstacles_flat])

    def _calc_reward(self, ee_pos: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        goal_reward = -dist_to_goal  # 距离越近奖励越高
        
        # 避障惩罚
        obstacle_penalty = 0.0
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(ee_pos - obs_pos)
            if dist_to_obs < self.obstacle_size + 0.05:
                obstacle_penalty += 10.0 * (1.0 - (dist_to_obs / (self.obstacle_size + 0.05)))
        
        # 动作平滑惩罚
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        self.prev_action = action.copy()
        
        return goal_reward - obstacle_penalty - smooth_penalty, dist_to_goal

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        # 1. 动作缩放
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 2. 执行动作
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 3. 扰动障碍物（可选）
        # self._disturb_obstacles()
        
        # 4. 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos
        reward, dist_to_goal = self._calc_reward(ee_pos, action)
        terminated = False
        collision = False
        
        # 目标达成
        if dist_to_goal < 0.05:
            reward += 10.0
            terminated = True
        
        # 碰撞检测
        for obs_pos in self.obstacles:
            if np.linalg.norm(ee_pos - obs_pos) < self.obstacle_size + 0.02:
                collision = True
                break
        # 地面碰撞
        if ee_pos[2] < 0.05:
            collision = True
            if self.visualize:  # 仅可视化时打印碰撞信息
                print(f"[碰撞] 末端碰地面，Z={ee_pos[2]:.3f}")
        if collision:
            reward -= 15.0
            terminated = True
        
        # 5. 仅可视化开启时，同步Viewer更新画面
        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)  # 控制帧率（100Hz），避免画面闪太快
        
        # 6. 生成观测和信息
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < 0.05),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        """释放Viewer资源（仅在visualize=True时执行）"""
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    num_obstacles: int = 1,
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,
    model_save_path: str = "panda_ppo_obstacle_avoidance",
    visualize: bool = False  # 训练时可视化开关（默认关闭）
) -> None:
    """
    PPO训练函数（默认关闭可视化，多进程高效训练）
    
    Args:
        num_obstacles: 障碍物数量
        n_envs: 并行环境数（24G显存推荐24）
        total_timesteps: 总训练步数
        model_save_path: 模型保存路径
    """
    # 训练环境配置：visualize=False（关键！多进程不创建Viewer）
    ENV_KWARGS = {'num_obstacles': num_obstacles, 'visualize': visualize}
    
    # 创建多进程向量环境（训练专用）
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}  # Linux用fork；Windows改为"spawn"
    )
    
    # 策略网络配置（24G显存适配）
    POLICY_KWARGS = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[1024, 512, 256], vf=[1024, 512, 256])]
    )
    
    # 初始化PPO模型（参数优化）
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        n_steps=1024,          # 单环境步数：1024×24=24576（有效Batch）
        batch_size=2048,       # 单次更新样本数（24576/12=2048）
        n_epochs=4,            # 数据重复利用轮数（24G显存适配）
        gamma=0.99,
        learning_rate=1.8e-3,  # 24G显存适配的学习率（有效Batch增大3倍）
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/panda_obstacle/"
    )
    
    # 开始训练
    print(f"=== 开始训练（可视化关闭）===")
    print(f"并行环境数: {n_envs}, 总步数: {total_timesteps}, 障碍物数: {num_obstacles}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    # 保存模型
    model.save(model_save_path)
    env.close()
    print(f"\n=== 训练结束 ===")
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_obstacle_avoidance",
    num_obstacles: int = 1,
    total_episodes: int = 5,
    record_gif: bool = False  # 是否录制GIF
) -> None:
    """
    PPO测试函数（适配MuJoCo 3.0+，修复渲染上下文错误）
    """
    # 测试环境配置：visualize=True（开启Viewer）
    env = PandaObstacleEnv(num_obstacles=num_obstacles, visualize=True)
    # 加载模型（指定测试环境）
    model = PPO.load(model_path, env=env)
    
    # 初始化GIF录制相关组件（适配MuJoCo 3.0+）
    frames = [] if record_gif else None
    render_scene = None  # 场景缓存
    render_context = None  # 渲染上下文（修正核心）
    pixel_buffer = None  # 像素缓存
    viewport = None  # 视口参数
    
    if record_gif and env.visualize and env.handle is not None:
        # 1. 窗口尺寸（与Viewer一致）
        width, height = 640, 480
        # 2. 初始化场景缓存
        render_scene = mujoco.MjvScene(env.model, maxgeom=10000)
        # 3. 修正：创建渲染上下文（MuJoCo 3.0+使用mjv_createContext）
        render_context = mujoco.MjvContext()  # 先创建空上下文
        mujoco.mjv_createContext(
            env.model, 
            mujoco.mjtVisFlag.mjVIS_ALL,  # 可视化所有元素
            render_context
        )
        # 4. 初始化像素缓存（RGB格式）
        pixel_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        # 5. 初始化视口（匹配窗口尺寸）
        viewport = mujoco.MjrRect(0, 0, width, height)
    
    success_count = 0
    print(f"\n=== 开始测试（可视化开启）===")
    print(f"测试轮数: {total_episodes}, 障碍物数: {num_obstacles}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # 模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # 录制GIF（适配MuJoCo 3.0+渲染流程）
            if record_gif and env.visualize and env.handle is not None:
                # 1. 更新场景（包含当前模型状态）
                mujoco.mjv_updateScene(
                    env.model, env.data,
                    env.handle.vopt, env.handle.pert,
                    env.handle.cam, mujoco.mjtCatBit.mjCAT_ALL,
                    render_scene
                )
                # 2. 渲染到像素缓存（使用修正的上下文和视口）
                mujoco.mjv_render(
                    viewport,
                    render_scene,
                    render_context,
                    env.handle.framebuffer  # 使用Viewer的帧缓冲
                )
                # 3. 从帧缓冲复制像素（修正：直接读取Viewer的framebuffer）
                pixel_buffer = np.frombuffer(env.handle.framebuffer, dtype=np.uint8)
                pixel_buffer = pixel_buffer.reshape((height, width, 4))[..., :3]  # 取RGB通道
                pixel_buffer = pixel_buffer[::-1]  # 翻转上下
                frames.append(pixel_buffer.copy())
            
            done = terminated or truncated
        
        # 统计成功率
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    # 计算成功率
    success_rate = (success_count / total_episodes) * 100
    print(f"\n=== 测试结束 ===")
    print(f"总成功率: {success_rate:.1f}%")
    
    # 保存GIF
    if record_gif and frames:
        import imageio
        gif_path = "panda_obstacle_test.gif"
        imageio.mimsave(gif_path, frames, duration=0.01)
        print(f"GIF已保存至: {gif_path}")
    
    env.close()


if __name__ == "__main__":
    # -------------------------- 配置选项 --------------------------
    TRAIN_MODE = True    # True=训练，False=测试
    NUM_OBSTACLES = 2    # 障碍物数量（训练和测试需一致）
    MODEL_PATH = "panda_ppo_obstacle_1obs"  # 模型保存/加载路径
    
    # -------------------------- 执行逻辑 --------------------------
    if TRAIN_MODE:
        # 1. 执行训练（可视化关闭，多进程高效运行）
        train_ppo(
            num_obstacles=NUM_OBSTACLES,
            n_envs=512,                
            total_timesteps=800_000_000,
            model_save_path=MODEL_PATH,
            visualize = False
        )
    else:
        # 2. 执行测试（可视化开启，单进程实时观察）
        test_ppo(
            model_path=MODEL_PATH,
            num_obstacles=NUM_OBSTACLES,
            total_episodes=15,
            record_gif=False  # 可选：开启GIF录制
        )