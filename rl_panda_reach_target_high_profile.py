import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import random
import time  # 测试时控制帧率
from typing import Optional  # 类型提示
from scipy.spatial.transform import Rotation as R

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")


class PandaObstacleEnv(gym.Env):
    # 新增visualize参数：True=开启可视化，False=关闭可视化（默认关闭）
    def __init__(self, visualize: bool = False):
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
        
        self.goal_size = 0.03
        
        # 5. 约束工作空间（不变）
        self.workspace = {
            'x': [-0.5, 0.8],
            'y': [-0.5, 0.5],
            'z': [0.05, 0.3]
        }
        
        # 6. 动作空间与观测空间（不变）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # 7轴关节角度、目标位置
        self.obs_size = 7 + 3
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
            if 0.4 < np.linalg.norm(goal - self.initial_ee_pos) < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """渲染目标点（仅在visualize=True时执行）"""
        if not self.visualize or self.handle is None:
            return
        # 清除现有几何
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
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
            self._render_scene()        
        
        obs = self._get_observation()
        assert obs.shape == (self.obs_size,), f"观测形状错误：{obs.shape} → 预期{self.obs_size,}"
        self.start_t = time.time()  # 记录开始时间
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        # 1. 目标距离相关奖励
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        goal_threshold = 0.005  # 到达目标的阈值（单位：米）
        
        # 非线性距离奖励
        if dist_to_goal < goal_threshold:
            distance_reward = 100.0  # 目标达成奖励
        elif dist_to_goal < 2*goal_threshold:
            distance_reward = 50.0  # 目标接近奖励
        elif dist_to_goal < 3*goal_threshold:
            distance_reward = 10.0  # 目标接近但不足奖励
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)
        
        # 2. 姿态约束：保持末端朝下
        # 假设ee_orient是末端执行器的方向向量，我们希望它指向下方（例如[0, 0, -1]）
        target_orient = np.array([0, 0, -1])  # 目标朝下的方向向量
        # 计算当前方向与目标方向的夹角（使用点积）
        # 归一化方向向量
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        # 点积 = |a||b|cosθ，这里|a|=|b|=1，所以点积直接是cosθ
        dot_product = np.dot(ee_orient_norm, target_orient)
        # 角度偏差（0到π）
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        # 姿态惩罚：角度偏差越大，惩罚越大
        orientation_penalty = 0.3 * angle_error  # 系数可调整
        
        # 3. 动作相关惩罚
        action_diff = action - self.prev_action
        smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        contact_reward = 1.0*self.data.ncon
        
        # 4. 关节角度限制惩罚
        joint_penalty = 0.0
        for i in range(7):
            min_angle, max_angle = self.model.jnt_range[:7][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # 5. 时间惩罚
        time_penalty = 0.01
        
        # 总奖励计算
        # total_reward = distance_reward - contact_reward - smooth_penalty - action_magnitude_penalty - joint_penalty - time_penalty - orientation_penalty
        total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty # - joint_penalty # - orientation_penalty
        # print(f"[奖励] 距离目标: {distance_reward:.3f}, [碰撞]: {contact_reward:.3f}, 动作惩罚: {smooth_penalty:.3f}, 姿态: {orientation_penalty:.3f}  总奖励: {total_reward:.3f}")
        
        # 更新上一步动作
        self.prev_action = action.copy()
        
        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        # 1. 动作缩放
        joint_ranges = self.model.jnt_range[:7]
        scaled_action = np.zeros(7, dtype=np.float32)
        for i in range(7):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 2. 执行动作
        self.data.ctrl[:7] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 4. 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[:7], action)
        terminated = False
        collision = False
        
        # 目标达成
        if dist_to_goal < 0.005:
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        if not terminated:
            if time.time() - self.start_t > 20.0:  # 10秒超时
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < 0.005),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    # 创建多进程向量环境（训练专用）
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    # 策略网络配置
    POLICY_KWARGS = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    # 初始化PPO模型（参数优化）
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
        tensorboard_log="./tensorboard/panda_reach_target/"
    )
    
    # 开始训练
    print(f"=== 开始训练（可视化关闭）===")
    print(f"并行环境数: {n_envs}, 总步数: {total_timesteps}")
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
    model_path: str = "panda_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)
    
    # 初始化GIF录制相关组件（适配MuJoCo 3.0+）
    record_gif = False
    frames = [] if record_gif else None
    render_scene = None  
    render_context = None 
    pixel_buffer = None 
    viewport = None
    
    success_count = 0
    print(f"\n=== 开始测试（可视化开启）===")
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # 模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        # 统计成功率
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    # 计算成功率
    success_rate = (success_count / total_episodes) * 100
    print(f"\n=== 测试结束 ===")
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


if __name__ == "__main__":
    # -------------------------- 配置选项 --------------------------
    TRAIN_MODE = False    # True=训练，False=测试
    MODEL_PATH = "panda_ppo_reach_target"  # 模型保存/加载路径
    
    # -------------------------- 执行逻辑 --------------------------
    if TRAIN_MODE:
        # 1. 执行训练（可视化关闭，多进程高效运行）
        train_ppo(
            n_envs=256,                
            total_timesteps=400_000_000,
            model_save_path=MODEL_PATH,
            visualize = False
        )
    else:
        # 2. 执行测试（可视化开启，单进程实时观察）
        test_ppo(
            model_path=MODEL_PATH,
            total_episodes=15,
        )