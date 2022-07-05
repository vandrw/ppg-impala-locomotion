from src.utils.env_loader import make_gym_env
from src.ppg.agent import Agent
from dataclasses import asdict
import pandas as pd
class Runner:
    def __init__(
        self,
        experiment_type,
        data_subject,
        initial_logstd,
        training_mode,
        save_pose,
        render,
        n_steps,
        tag,
        save_path,
    ):
        self.env = make_gym_env(
            experiment_type, data_subject=data_subject, visualize=render
        )

        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(
            self.state_dim,
            self.action_dim,
            initial_logstd,
            training_mode
        )

        self.tag = tag
        self.training_mode = training_mode
        self.save_pose = save_pose
        self.n_steps = n_steps

        self.save_path = save_path
        print("[Proc {}] Worker initialized.".format(tag))

        self.total_reward = 0
        self.max_reward = 0
        self.eps_time = 0
        self.num_episodes = 1.0e-8
        self.pose_info = pd.DataFrame(
            columns=[
                "time",
                "pelvis_tilt",
                "pelvis_list",
                "pelvis_rotation",
                "pelvis_tx",
                "pelvis_ty",
                "pelvis_tz",
                "hip_flexion_r",
                "hip_adduction_r",
                "hip_rotation_r",
                "knee_angle_r",
                "ankle_angle_r",
                "hip_flexion_l",
                "hip_adduction_l",
                "hip_rotation_l",
                "knee_angle_l",
                "ankle_angle_l",
                "lumbar_extension",
            ]
        )

    def run_episode(self):
        self.agent.memory.clear_memory()
        self.agent.load_weights(self.save_path)
    
        self.num_episodes = 1.0e-8
        ep_info = {"total_reward": 0, "episode_time": 0, "max_reward": 0, "pose": None}
        curr_pose = {}

        for _ in range(self.n_steps):
            action, action_mean, action_std = self.agent.act(self.states)

            try:
                next_state, reward, done, info = self.env.step(action)

                if self.save_pose:
                    pose = asdict(info["obs/dyn_pose"])
                    curr_pose["time"] = self.eps_time / 200.0
                    for body in pose["pose"]:
                        curr_pose[body] = pose["pose"][body]
                    curr_pose["lumbar_extension"] = -5.00000214

                    self.pose_info.loc[-1] = curr_pose.values()
                    self.pose_info.index += 1
                
                self.eps_time += 1
                self.total_reward += reward

                if self.training_mode:
                    self.agent.save_eps(
                        self.states.tolist(),
                        action,
                        action_mean,
                        reward,
                        float(done),
                        next_state.tolist(),
                    )

                self.states = next_state
            except AssertionError:
                # If the agent runs out of imitation data, we reset
                # the environment.
                done = True

            if done:
                self.states = self.env.reset()

                ep_info["total_reward"] += self.total_reward
                ep_info["episode_time"] += self.eps_time
                self.num_episodes += 1

                if self.save_pose:
                    if self.total_reward >= self.max_reward:
                        self.max_reward = self.total_reward
                        ep_info["max_reward"] = self.max_reward
                        ep_info["pose"] = self.pose_info.copy()
                
                    self.pose_info = self.pose_info[0:0]

                self.done = True
                self.total_reward = 0
                self.eps_time = 0


        self.agent.memory.update_std(action_std)
        
        ep_info["total_reward"] /= self.num_episodes
        ep_info["episode_time"] /= self.num_episodes

        return (
            self.agent.get_all(),
            ep_info,
        )