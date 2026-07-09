import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
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
        self.left_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger'
        )
        self.right_finger_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger'
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

        # 动作空间：7个臂关节增量（夹爪由脚本控制）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 观测空间
        # 7(q) + 7(dq) + 3(ee_pos) + 3(cube_pos) + 3(target) + 3(ee_to_cube) + 2(gripper) + 1(grasp_phase) = 29
        self.obs_size = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # 任务参数
        self.target_place_pos = np.array([0.55, -0.12, 0.02], dtype=np.float32)
        self.lift_threshold = 0.05
        # 成功阈值
        self.place_threshold = 0.03
        # 助力把方块抬到这个高度再交给策略，避免太起来就放回
        self.carry_height = 0.12
        # 离目标横向距离小于此值时撤掉助力，让策略自由下放
        self.place_release_dist = 0.08
        
        self.gripper_open = 255.0
        self.gripper_close = 0.0

        self.cube_rand_range = 0.01
        # 后面可以根据 episode 增加随机范围
        # self.cube_rand_range = min(0.05, 0.01 + episode * 0.00001)
        
        self.max_delta_q = 0.05
        self.control_dt = 0.04
        self.sim_dt = self.model.opt.timestep

        # 让策略倾向保持自然 home 姿态
        self.posture_weights = np.array([0.08, 0.08, 0.06, 0.10, 0.18, 0.18, 0.25], dtype=np.float32)
        self.joint_margin = 0.15

        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_gripper_width = 0.04
        self.last_ee_z = 0.0
        self.last_cube_z = 0.0
        self.cube_init_z = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 250

        # 阶段标记
        self.touched = False
        self.gripper_opened = False
        self.gripper_opened_step = 0
        self.grasped = False
        self.lifted = False
        self.lost_contact_steps = 0
        self._assist_on = False
        # 夹爪脚本化状态：0=张开靠近, 1=闭合抓取, 2=提升
        self.grasp_phase = 0
        self.close_steps = 0  # 已闭合的步数

    def _render_scene(self):
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
        self.data.qvel[:self.arm_gripper_qpos_dim] = 0.0
        self.data.ctrl[:7] = self.home_joint_pos[:7]
        self.data.ctrl[7] = self.gripper_open

        # 方块位置随机扰动
        cube_rand = self.np_random.uniform(-self.cube_rand_range, self.cube_rand_range, size=3)
        cube_rand[2] = 0.0
        self.cube_init_pos = self.cube_default_pos + cube_rand
        self.data.qpos[self.cube_qpos_start:self.cube_qpos_start+3] = self.cube_init_pos
        
        # 保持默认四元数
        default_quat = self.model.qpos0[self.cube_qpos_start+3:self.cube_qpos_end]
        self.data.qpos[self.cube_qpos_start+3:self.cube_qpos_end] = default_quat

        mujoco.mj_forward(self.model, self.data)

        # 重置阶段标记
        self.touched = False
        self.gripper_opened = False
        self.gripper_opened_step = 0
        self.grasped = False
        self.lifted = False
        self.lost_contact_steps = 0
        self._assist_on = False
        self.grasp_phase = 0
        self.close_steps = 0

        self.episode_steps = 0
        self.last_action = np.zeros(7, dtype=np.float32)
        self.last_gripper_width = float(self.data.qpos[7:9].mean())
        ee_init_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.last_ee_z = ee_init_pos[2]
        self.cube_init_z = float(self.data.body(self.cube_id).xpos[2])
        self.last_cube_z = self.cube_init_z

        if self.visualize:
            self._render_scene()

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
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
            np.array([float(self.grasp_phase)], dtype=np.float32),  # 1
        ]).astype(np.float32)
        
        # 观测噪声
        obs += np.random.normal(0, 0.001, obs.shape).astype(np.float32)
        return obs

    def _is_grasped_by_contact(self) -> bool:
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = contact.geom1
            body2 = contact.geom2
            body1_id = self.model.geom_bodyid[body1]
            body2_id = self.model.geom_bodyid[body2]
            bodies = {body1_id, body2_id}
            if self.cube_id in bodies and self.left_finger_id in bodies:
                left_contact = True
            if self.cube_id in bodies and self.right_finger_id in bodies:
                right_contact = True

        gripper_width = self.data.qpos[7:9].mean()
        return left_contact and right_contact and gripper_width < 0.035

    def _calc_reward(self, action: np.ndarray) -> tuple[float, bool, bool, bool]:
        """阶梯式奖励: stage_base + 阶段内稠密[0,1]"""

        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        cube_pos = self.data.body(self.cube_id).xpos.copy()
        gripper_width = self.data.qpos[7:9].mean()

        dist_ee2cube = np.linalg.norm(ee_pos - cube_pos)
        dist_cube2target = np.linalg.norm(cube_pos - self.target_place_pos)
        lateral_to_target = np.linalg.norm(cube_pos[:2] - self.target_place_pos[:2])
        cube_z = cube_pos[2]

        reward = 0.0

        lateral_dist = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        vertical_offset = ee_pos[2] - cube_pos[2] - 0.025
        vertical_gap = ee_pos[2] - cube_pos[2]

        # 末端姿态朝下
        ee_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        alignment = np.dot(ee_rot[:, 2], np.array([0.0, 0.0, -1.0]))
        # 计算Y轴水平方向投影
        finger_axis_xy = (-ee_rot[:, 1])[:2]
        finger_axis_xy_norm = np.linalg.norm(finger_axis_xy)
        if finger_axis_xy_norm > 1e-6:
            finger_axis_xy = finger_axis_xy / finger_axis_xy_norm
            # 只关心和方块的一个边平行即可，不是斜着就可以
            yaw_align = max(abs(finger_axis_xy[0]), abs(finger_axis_xy[1]))
        else:
            yaw_align = 0.0

        is_physically_grasped = self._is_grasped_by_contact()

        # 是否抓着标志位消抖
        if self.grasped:
            if is_physically_grasped and gripper_width < 0.035:
                self.lost_contact_steps = 0
            else:
                self.lost_contact_steps += 1
                if self.lost_contact_steps >= 4:
                    self.grasped = False

        # 每步只落在一个 stage
        if not self.grasped:
            if self.grasp_phase == 0:
                # stage0 接近：base 0 + reach∈[0,1]
                reach = 0.5 * (1.0 - np.tanh(6.0 * lateral_dist)) \
                        + 0.5 * (1.0 - np.tanh(6.0 * abs(vertical_offset)))
                reward += reach  # ≤1

                if (lateral_dist < 0.018 and 0.01 < vertical_gap < 0.038):
                    pose = 0.5 * alignment + 0.5 * yaw_align
                    reward += pose

                # 里程碑: 到达抓取位姿
                if (lateral_dist < 0.018 and 0.008 < vertical_gap < 0.038
                        and alignment > 0.965 and yaw_align > 0.93):
                    reward += 1.0
                    self.grasp_phase = 1
                    self.close_steps = 0
                
                # 记录到 tensorboard 中
                self.alignment = alignment
                self.ee_yaw = yaw_align
            else:
                # stage1 闭合：base 1.0 + 0.5
                reward += 1.0
                if is_physically_grasped:
                    reward += 0.5
                    cube_velocity = np.linalg.norm(
                        self.data.qvel[self.cube_qpos_start:self.cube_qpos_start+3]
                    )
                    # 两个夹抓贴着一边推着走的情况
                    if cube_velocity < 0.05:
                        reward += 1.0
                        self.grasped = True
                        self.lost_contact_steps = 0
        else:
            # 防止拿起来放下去
            if (self.lifted and cube_z < self.lift_threshold - 0.02 and lateral_to_target > 0.06):
                self.lifted = False

            if not self.lifted:
                # stage2 抬升中：base 1.0 + 2*lift∈[0,2]
                reward += 1.0
                lift_norm = np.clip((cube_z - self.cube_init_z) / self.lift_threshold, 0.0, 1.0)
                reward += 2.0 * lift_norm  # ≤2
                if cube_z > self.lift_threshold:
                    reward += 1.0 
                    self.lifted = True
            else:
                # stage3 搬运放置：base 3.0 + 4*place_xy + 0.5*place_z
                reward += 3.0
                place_xy = 1.0 - np.tanh(2.5 * lateral_to_target)
                reward += 4.0 * place_xy  # ≤4
                target_blend = 1.0 - np.tanh(8.0 * lateral_to_target)  # lat=0→1, lat=0.15→0
                # lat 大 → desired_z ≈ carry_height（托住，防脱手）
                # lat 小 → desired_z ≈ target_z（鼓励下放）
                desired_z = (self.carry_height * (1.0 - target_blend)
                             + self.target_place_pos[2] * target_blend)
                z_err = abs(cube_z - desired_z)
                reward += 0.5 * (1.0 - np.tanh(8.0 * z_err))  # ≤0.5

        is_success = self.grasped and self.lifted and dist_cube2target < self.place_threshold
        if is_success:
            reward += 1.0

        # 姿态保持
        if not self.lifted:
            reward += 0.3 * (alignment - 1.0)
            reward += 0.2 * (yaw_align - 1.0)

        # 惩罚项 ≤0.1 
        cube_velocity = np.linalg.norm(self.data.qvel[self.cube_qpos_start:self.cube_qpos_start+3])
        if not self.grasped and dist_ee2cube < 0.05 and cube_velocity > 0.05:
            reward -= 0.3

        q = self.data.qpos[:7].copy()
        dq = self.data.qvel[:7].copy()
        q_home = self.home_joint_pos[:7]
        joint_ranges = self.model.jnt_range[:7]
        range_width = joint_ranges[:, 1] - joint_ranges[:, 0]

        posture_error = (q - q_home) / range_width
        posture_penalty = np.sum(self.posture_weights * np.square(posture_error))
        reward -= 0.05 * posture_penalty

        lower_margin = q - joint_ranges[:, 0]
        upper_margin = joint_ranges[:, 1] - q
        limit_violation = np.maximum(0.0, self.joint_margin - np.minimum(lower_margin, upper_margin))
        reward -= 0.5 * np.sum(np.square(limit_violation / self.joint_margin))

        reward -= 0.003 * np.linalg.norm(dq)
        reward -= 0.01 * np.linalg.norm(action - self.last_action)
        reward += -0.02  # 时间惩罚

        is_drop = cube_z < -0.02
        if is_drop:
            reward -= 5.0

        terminated = is_drop
        return reward, terminated, is_success, is_drop

    def _scripted_gripper(self) -> float:
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        cube_pos = self.data.body(self.cube_id).xpos.copy()

        lateral_dist = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        vertical_dist = ee_pos[2] - cube_pos[2]

        ee_rot = self.data.body(self.end_effector_id).xmat.reshape(3, 3)
        alignment = np.dot(ee_rot[:, 2], np.array([0.0, 0.0, -1.0]))

        finger_axis_world = -ee_rot[:, 1]
        finger_axis_xy = finger_axis_world[:2]
        finger_axis_xy_norm = np.linalg.norm(finger_axis_xy)
        if finger_axis_xy_norm > 1e-6:
            finger_axis_xy = finger_axis_xy / finger_axis_xy_norm
            yaw_align = max(abs(finger_axis_xy[0]), abs(finger_axis_xy[1]))
        else:
            yaw_align = 0.0

        if self.grasp_phase == 0:
            # 张开靠近
            if lateral_dist < 0.018 and 0.008 < vertical_dist < 0.038 and alignment > 0.965 and yaw_align > 0.93:
                self.grasp_phase = 1
                self.close_steps = 0
        elif self.grasp_phase == 1:
            # 闭合抓取
            self.close_steps += 1
            if self.close_steps > 5 and self._is_grasped_by_contact():
                self.grasp_phase = 2
            elif self.close_steps > 15:
                self.grasp_phase = 0

        if self.grasp_phase == 0:
            return self.gripper_open
        else:
            return self.gripper_close

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        cube_pos_now = self.data.body(self.cube_id).xpos
        cube_z_now = cube_pos_now[2]
        if not self.grasped or self.lifted:
            self._assist_on = False
        elif cube_z_now < self.carry_height - 0.03:
            self._assist_on = True
        elif cube_z_now >= self.carry_height:
            self._assist_on = False

        commanded_action = action.copy()
        if self._assist_on:
            current_q = self.data.qpos[:7]
            home_action = np.clip(
                (self.home_joint_pos[:7] - current_q) / self.max_delta_q, -1.0, 1.0
            )
            # 保证从桌面可靠抬起；lifted 后此分支已不再进入
            blend = 0.5
            commanded_action = np.clip(blend * home_action + (1-blend) * action, -1.0, 1.0)

        # 增量位置控制
        current_q = self.data.qpos[:7].copy()
        for i in range(7):
            delta = commanded_action[i] * self.max_delta_q
            target = current_q[i] + delta
            jnt_range = self.model.jnt_range[i]
            self.data.ctrl[i] = float(np.clip(target, jnt_range[0], jnt_range[1]))

        # 脚本化夹爪控制
        gripper_target = self._scripted_gripper()
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

        reward, terminated, is_success, is_drop = self._calc_reward(action)
        
        truncated = self.episode_steps >= self.max_episode_steps
        if truncated:
            reward -= 1.0

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
            "alignment": self.alignment,
            "yaw_align": self.ee_yaw
        }

        self.last_action = action.copy()
        gripper_width = self.data.qpos[7:9].mean()
        self.last_gripper_width = gripper_width
        self.last_ee_z = self.data.body(self.end_effector_id).xpos[2]
        self.last_cube_z = self.data.body(self.cube_id).xpos[2]

        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None


class TaskMetricsCallback(BaseCallback):
    """记录 TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_buf = {"grasped": [], "lifted": [], 
                       "is_success": [], "cube_z_max": [],
                       "alignment": [], "yaw_align": [], 
                       "cube_to_target_dist": [], "ee_to_cube_dist": []}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            if dones[i] and "episode" in info:
                self.ep_buf["grasped"].append(float(info.get("grasped", False)))
                self.ep_buf["lifted"].append(float(info.get("lifted", False)))
                self.ep_buf["is_success"].append(float(info.get("is_success", False)))
                self.ep_buf["cube_z_max"].append(float(info.get("cube_z", 0.0)))
                self.ep_buf["alignment"].append(float(info.get("alignment", 0.0)))
                self.ep_buf["yaw_align"].append(float(info.get("yaw_align", 0.0)))
                self.ep_buf["cube_to_target_dist"].append(float(info.get("cube_to_target_dist", 0.0)))
                self.ep_buf["ee_to_cube_dist"].append(float(info.get("ee_to_cube_dist", 0.0)))                                   

        # 每 1000 步
        if self.num_timesteps % 2048 == 0 and len(self.ep_buf["grasped"]) > 0:
            for k, v in self.ep_buf.items():
                if v:
                    self.logger.record(f"task/{k}_rate", np.mean(v))
            self.ep_buf = {k: [] for k in self.ep_buf}
        return True


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
        env = Monitor(env, info_keywords=("is_success", "grasped", "lifted", "cube_z", "ee_to_cube_dist"))
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
            log_std_init=-0.5,
        )

        # 学习率线性衰减到 0：后期更新变小，收敛更稳，抑制末期震荡
        def lr_schedule(progress_remaining: float) -> float:
            return 3e-4 * progress_remaining

        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=256,
            batch_size=1024,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            max_grad_norm=0.5,
            learning_rate=lr_schedule,
            target_kl=0.03,
            device="cuda:1" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_pickup/",
        )
    eval_env = make_vec_env(make_env, n_envs=1, seed=999)

    print(f"开始训练，总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=TaskMetricsCallback(),
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
    # model = PPO.load(model_path, device="cpu", env=env)
    # model.to("cuda")

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
    TRAIN_MODE = False
    
    MODEL_SAVE_PATH = "assets/model/rl_pickup_checkpoint/panda_pickup_v1_5"
    RESUME_PATH = None

    if TRAIN_MODE:
        import shutil
        if os.path.exists("./tensorboard/panda_pickup/"):
            shutil.rmtree("./tensorboard/panda_pickup/")

        train_ppo(
            n_envs=64,
            total_timesteps=60_000_000,
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
