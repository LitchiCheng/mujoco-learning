import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
import os

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("flag")
        return True
    except Exception:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    return os.path.exists(os.path.join("/tmp", flag_filename))

def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if os.path.exists(flag_path):
        os.remove(flag_path)
    return True

class PandaPickupEnv(gym.Env):
    def __init__(self, visualize: bool = False):
        super().__init__()
        
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.xml_path = './model/franka_emika_panda/scene_with_rl_pickup_cube.xml'
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.2
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -35.0
            self.handle.cam.lookat = np.array([0.3, 0.0, 0.3])

        self.np_random = np.random.default_rng(None)

        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'ee_center_body'
        )
        self.cube_body_name = "cube"
        self.cube_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name
        )
        self.cube_joint_name = "cube_joint"
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.cube_joint_name
        )
        
        self.freejoint_dim = 7
        self.cube_qpos_start = self.model.jnt_qposadr[self.cube_joint_id]
        self.cube_qpos_end = self.cube_qpos_start + self.freejoint_dim

        # 机械臂+夹爪维度
        self.arm_gripper_qpos_dim = 9
        self.home_joint_pos = self.model.key_qpos[0][:self.arm_gripper_qpos_dim].copy().astype(np.float32)

        mujoco.mj_forward(self.model, self.data)
        self.cube_default_pos = self.data.body(self.cube_id).xpos.copy().astype(np.float32)
        self.cube_init_pos = self.cube_default_pos.copy()

        # 动作空间：7臂增量 + 1夹爪
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 观测空间
        # 7(q) + 7(dq) + 3(ee_pos) + 3(cube_pos) + 3(target) + 3(ee_to_cube) + 2(gripper) = 28
        self.obs_size = 28
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # 任务参数
        self.target_place_pos = np.array([0.45, -0.15, 0.02], dtype=np.float32)
        self.grasp_threshold = 0.03
        self.lift_threshold = 0.05
        self.place_threshold = 0.03
        
        self.gripper_open = 0.04
        self.gripper_close = 0.00
        
        self.max_delta_q = 0.1
        self.control_dt = 0.04
        self.sim_dt = self.model.opt.timestep
        
        self.last_action = np.zeros(8, dtype=np.float32)
        self.episode_steps = 0
        self.max_episode_steps = 250
        
        # 阶段标记
        self.touched = False
        self.grasped = False
        self.lifted = False

    def _render_scene(self):
        """绘制目标点和方块实时位置"""
        if not self.visualize or self.handle is None:
            return
        
        self.handle.user_scn.ngeom = 0
        
        # 目标点（绿色）
        if self.handle.user_scn.ngeom < self.handle.user_scn.maxgeom:
            g = self.handle.user_scn.geoms[self.handle.user_scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.0, 0.0],
                pos=self.target_place_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.1, 0.8, 0.2, 0.7], dtype=np.float32)
            )
            self.handle.user_scn.ngeom += 1
        
        # 方块实时位置（红色）
        if self.handle.user_scn.ngeom < self.handle.user_scn.maxgeom:
            g = self.handle.user_scn.geoms[self.handle.user_scn.ngeom]
            current_cube_pos = self.data.body(self.cube_id).xpos.copy()
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.02, 0.0, 0.0],
                pos=current_cube_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.8, 0.2, 0.1, 0.7], dtype=np.float32)
            )
            self.handle.user_scn.ngeom += 1

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # 复位机械臂+夹爪
        self.data.qpos[:self.arm_gripper_qpos_dim] = self.home_joint_pos.copy()

        # 方块位置随机扰动
        cube_rand = self.np_random.uniform(-0.05, 0.05, size=3)
        cube_rand[2] = 0.0
        self.cube_init_pos = self.cube_default_pos + cube_rand
        self.data.qpos[self.cube_qpos_start:self.cube_qpos_start+3] = self.cube_init_pos
        
        # 保持默认四元数
        default_quat = self.model.qpos0[self.cube_qpos_start+3:self.cube_qpos_end]
        self.data.qpos[self.cube_qpos_start+3:self.cube_qpos_end] = default_quat

        mujoco.mj_forward(self.model, self.data)

        # 重置阶段标记
        self.touched = False
        self.grasped = False
        self.lifted = False
        
        self.episode_steps = 0
        self.last_action = np.zeros(8, dtype=np.float32)

        if self.visualize:
            self._render_scene()

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """构建观测向量"""
        q = self.data.qpos[:7].copy().astype(np.float32)
        dq = self.data.qvel[:7].copy().astype(np.float32)
        
        gripper_q = self.data.qpos[7:9].copy().astype(np.float32)
        
        ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        cube_pos = self.data.body(self.cube_id).xpos.copy().astype(np.float32)
        target_pos = self.target_place_pos.copy()
        
        ee_to_cube = cube_pos - ee_pos
        
        obs = np.concatenate([
            q,              # 7
            dq,             # 7
            ee_pos,         # 3
            cube_pos,       # 3
            target_pos,     # 3
            ee_to_cube,     # 3
            gripper_q,      # 2
        ]).astype(np.float32)
        
        # 观测噪声
        obs += np.random.normal(0, 0.001, obs.shape).astype(np.float32)
        return obs

    def _calc_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """分阶段奖励计算"""
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        cube_pos = self.data.body(self.cube_id).xpos.copy()
        gripper_width = self.data.qpos[7:9].mean()
        
        dist_ee2cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube2target = np.linalg.norm(cube_pos - self.target_place_pos)
        cube_z = cube_pos[2]
        
        reward = 0.0
        
        # 接近奖励（稠密）
        reach_reward = 1.0 - np.tanh(5.0 * dist_ee2cube)
        reward += reach_reward * 0.5
        
        # 接触奖励（一次性）
        if not self.touched and dist_ee2cube < self.grasp_threshold:
            reward += 5.0
            self.touched = True
        
        # 抓取奖励（一次性）
        is_grasping = (gripper_width < 0.015) and (dist_ee2cube < self.grasp_threshold + 0.02)
        if not self.grasped and is_grasping:
            reward += 10.0
            self.grasped = True
        
        # 抓取维持
        if self.grasped and is_grasping:
            reward += 0.5
        
        # 提升奖励
        if is_grasping and cube_z > self.lift_threshold:
            lift_reward = (cube_z - self.lift_threshold) * 10.0
            reward += lift_reward
            self.lifted = True
        
        # 放置奖励
        if self.lifted:
            place_reward = 1.0 - np.tanh(5.0 * dist_cube2target)
            reward += place_reward * 1.0
        
        # 成功奖励
        is_success = dist_cube2target < self.place_threshold and cube_z > self.lift_threshold
        if is_success:
            reward += 50.0
        
        # 惩罚项
        action_penalty = -0.01 * np.linalg.norm(action)
        reward += action_penalty
        
        time_penalty = -0.05
        reward += time_penalty
        
        is_drop = cube_z < -0.02
        if is_drop:
            reward -= 20.0
        
        # 碰撞惩罚（轻微）
        if self.data.ncon > 3:
            reward -= 1.0
        
        terminated = is_success or is_drop
        
        return reward, terminated, is_success, is_drop

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        # 增量位置控制
        current_q = self.data.qpos[:7].copy()
        for i in range(7):
            delta = action[i] * self.max_delta_q
            target = current_q[i] + delta
            jnt_range = self.model.jnt_range[i]
            self.data.ctrl[i] = float(np.clip(target, jnt_range[0], jnt_range[1]))
        
        # 夹爪控制
        gripper_target = (action[6] + 1.0) * 0.5 * self.gripper_open
        self.data.ctrl[7] = gripper_target
        if self.model.nu > 8:
            self.data.ctrl[8] = gripper_target

        # 仿真步进
        n_steps = int(self.control_dt / self.sim_dt)
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            if self.data.body(self.cube_id).xpos[2] < -0.05:
                break

        self.episode_steps += 1
        
        # 可视化
        if self.visualize and self.handle is not None:
            self._render_scene()
            self.handle.sync()
            time.sleep(self.control_dt)

        # 计算奖励
        reward, terminated, is_success, is_drop = self._calc_reward(action)
        
        truncated = self.episode_steps >= self.max_episode_steps
        if truncated:
            reward -= 5.0

        obs = self._get_observation()
        
        info = {
            "is_success": is_success,
            "cube_to_target_dist": np.linalg.norm(self.data.body(self.cube_id).xpos - self.target_place_pos),
            "ee_to_cube_dist": np.linalg.norm(self.data.body(self.end_effector_id).xpos - self.data.body(self.cube_id).xpos),
            "cube_z": self.data.body(self.cube_id).xpos[2],
            "collision": self.data.ncon > 3,
            "cube_drop": is_drop,
            "grasped": self.grasped,
            "lifted": self.lifted,
        }
        
        self.last_action = action.copy()
        
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None


def train_ppo(
    n_envs: int = 16,
    total_timesteps: int = 5_000_000,
    model_save_path: str = "panda_pickup_ppo",
    visualize: bool = False,
    resume_from: Optional[str] = None
):
    delete_flag_file()
    
    ENV_KWARGS = {'visualize': visualize}

    def make_env():
        env = PandaPickupEnv(**ENV_KWARGS)
        env = Monitor(env)
        return env

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
    )

    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"并行环境数: {n_envs}")

    if resume_from is not None and os.path.exists(resume_from + ".zip"):
        print(f"从 {resume_from} 恢复训练...")
        model = PPO.load(resume_from, env=env)
    else:
        policy_kwargs = dict(
            activation_fn=nn.LeakyReLU,
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_pickup/",
        )
    eval_env = make_vec_env(make_env, n_envs=1, seed=999)

    print(f"开始训练，总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )
    
    model.save(model_save_path)
    env.close()
    eval_env.close()
    print(f"模型已保存至: {model_save_path}")
    delete_flag_file()


def test_ppo(
    model_path: str = "panda_pickup_ppo",
    total_episodes: int = 10,
    render: bool = True
):
    delete_flag_file()
    
    env = PandaPickupEnv(visualize=render)
    model = PPO.load(model_path, env=env)

    success_count = 0
    grasp_count = 0
    print(f"开始测试，总轮数: {total_episodes}")

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        if info["is_success"]:
            success_count += 1
        if info["grasped"]:
            grasp_count += 1

        print(f"回合{ep+1:2d} | 步数:{steps:3d} | 奖励:{ep_reward:7.2f} | "
              f"成功:{info['is_success']} | 抓取:{info['grasped']} | "
              f"最终高度:{info['cube_z']:.3f}")

    print(f"\n测试完成:")
    print(f"  成功率: {success_count}/{total_episodes} ({success_count/total_episodes*100:.1f}%)")
    print(f"  抓取率: {grasp_count}/{total_episodes} ({grasp_count/total_episodes*100:.1f}%)")
    
    env.close()
    delete_flag_file()


if __name__ == "__main__":
    TRAIN_MODE = True
    
    MODEL_SAVE_PATH = "assets/model/rl_pickup_checkpoint/panda_pickup_v2"
    RESUME_PATH = None

    if TRAIN_MODE:
        import shutil
        if os.path.exists("./tensorboard/panda_pickup/"):
            shutil.rmtree("./tensorboard/panda_pickup/")

        train_ppo(
            n_envs=32,
            total_timesteps=10_000_000,
            model_save_path=MODEL_SAVE_PATH,
            visualize=True,
            resume_from=RESUME_PATH,
        )
    else:
        test_ppo(
            model_path=MODEL_SAVE_PATH,
            total_episodes=20,
            render=True,
        )
    os.system("date")