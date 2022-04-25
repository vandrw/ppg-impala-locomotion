import gym
from src.env_loader import load_ppg_env

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np

import time
import datetime
import os

import argparse
import yaml
import logging

import ray

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

_logger = None
output_path = None

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice=None):
        super(Policy_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
              ).float().to(self.device)

        self.actor_layer = nn.Sequential(
                nn.Linear(128, action_dim),
                nn.Tanh()
              ).float().to(self.device)

        self.critic_layer = nn.Sequential(
                nn.Linear(128, 1)
              ).float().to(self.device)

    def forward(self, states):
        x = self.nn_layer(states)
        return self.actor_layer(x), self.critic_layer(x)


class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim, myDevice=None):
        super(Value_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              ).float().to(self.device)

    def forward(self, states):
        return self.nn_layer(states)


class PolicyMemory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_means = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return (
            np.array(self.states[idx], dtype=np.float32),
            np.array(self.actions[idx], dtype=np.float32),
            np.array(self.action_means[idx], dtype=np.float32),
            np.array([self.rewards[idx]], dtype=np.float32),
            np.array([self.dones[idx]], dtype=np.float32),
            np.array(self.next_states[idx], dtype=np.float32),
        )

    def get_all(self):
        return (
            self.states,
            self.actions,
            self.action_means,
            self.rewards,
            self.dones,
            self.next_states,
        )

    def save_all(self, states, actions, action_means, rewards, dones, next_states):
        self.states = states
        self.actions = actions
        self.action_means = action_means
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states

    def save_list(self, states, actions, action_means, rewards, dones, next_states):
        self.states += states
        self.actions += actions
        self.action_means += action_means
        self.rewards += rewards
        self.dones += dones
        self.next_states += next_states

    def save_eps(self, state, action, action_mean, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.action_means.append(action_mean)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.action_means[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]


class AuxMemory(Dataset):
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32)

    def save_all(self, states):
        self.states = self.states + states

    def clear_memory(self):
        del self.states[:]


class Continous:
    def __init__(self, myDevice=None):
        self.device = myDevice if myDevice != None else device

    def sample(self, mean, std):
        distribution = Normal(mean, std)
        return distribution.sample().float().to(self.device)

    def entropy(self, mean, std):
        distribution = Normal(mean, std)
        return distribution.entropy().float().to(self.device)

    def logprob(self, mean, std, value_data):
        distribution = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(self.device)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(self.device)


class PolicyFunction:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []

        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return torch.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def vtrace_generalized_advantage_estimation(
        self, values, rewards, next_values, dones, learner_logprobs, worker_logprobs
    ):
        gae = 0
        adv = []

        limit = torch.FloatTensor([1.0]).to(device)
        ratio = torch.min(limit, (worker_logprobs - learner_logprobs).sum().exp())

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        delta = ratio * delta

        for step in reversed(range(len(rewards))):
            gae = (1.0 - dones[step]) * self.gamma * self.lam * gae
            gae = delta[step] + ratio * gae
            adv.insert(0, gae)

        return torch.stack(adv)


class TrulyPPO:
    def __init__(
        self,
        policy_kl_range,
        policy_params,
        value_clip,
        vf_loss_coef,
        entropy_coef,
        gamma,
        lam,
    ):
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef

        self.distributions = Continous()
        self.policy_function = PolicyFunction(gamma, lam)

    def compute_loss(
        self,
        action_mean,
        action_std,
        old_action_mean,
        old_action_std,
        values,
        old_values,
        next_values,
        actions,
        rewards,
        dones,
        worker_action_means,
        worker_std,
    ):
        # Don't use old value in backpropagation
        Old_values = old_values.detach()
        Old_action_mean = old_action_mean.detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(action_mean, action_std, actions)
        Old_logprobs = self.distributions.logprob(
            Old_action_mean, old_action_std, actions
        ).detach()
        Worker_logprobs = self.distributions.logprob(
            worker_action_means, worker_std, actions
        ).detach()

        # Getting general advantages estimator and returns
        Advantages = self.policy_function.vtrace_generalized_advantage_estimation(
            values, rewards, next_values, dones, logprobs, Worker_logprobs
        )
        Returns = (Advantages + values).detach()
        Advantages = (
            (Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)
        ).detach()

        # Finding Surrogate Loss
        ratios = (logprobs - Old_logprobs).exp()  # ratios = old_logprobs / logprobs
        Kl = self.distributions.kl_divergence(
            Old_action_mean, old_action_std, action_mean, action_std
        )

        pg_targets = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages,
        )
        pg_loss = pg_targets.mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_mean, action_std).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
            critic_loss = ((Returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped = old_values + torch.clamp(
                values - Old_values, -self.value_clip, self.value_clip
            )  # Minimize the difference between old value and new value
            vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
            vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
            critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (
            (critic_loss * self.vf_loss_coef)
            - (dist_entropy * self.entropy_coef)
            - pg_loss
        )
        return loss


class JointAux:
    def __init__(self):
        self.distributions = Continous()

    def compute_loss(
        self, action_mean, action_std, old_action_mean, old_action_std, values, Returns
    ):
        # Don't use old value in backpropagation
        Old_action_mean = old_action_mean.detach()

        # Finding KL Divergence
        Kl = self.distributions.kl_divergence(
            Old_action_mean, old_action_std, action_mean, action_std
        ).mean()
        aux_loss = ((Returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl


class Learner:
    def __init__(
        self,
        state_dim,
        action_dim,
        is_training_mode,
        policy_kl_range,
        policy_params,
        value_clip,
        entropy_coef,
        vf_loss_coef,
        batchsize,
        PPO_epochs,
        gamma,
        lam,
        learning_rate,
    ):
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.batchsize = batchsize
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.std = torch.ones([1, action_dim]).float().to(device)

        self.policy = Policy_Model(state_dim, action_dim)
        self.policy_old = Policy_Model(state_dim, action_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=learning_rate)

        self.value = Value_Model(state_dim, action_dim)
        self.value_old = Value_Model(state_dim, action_dim)
        self.value_optimizer = Adam(self.value.parameters(), lr=learning_rate)

        self.policy_memory = PolicyMemory()
        self.policy_loss = TrulyPPO(
            policy_kl_range,
            policy_params,
            value_clip,
            vf_loss_coef,
            entropy_coef,
            gamma,
            lam,
        )

        self.aux_memory = AuxMemory()
        self.aux_loss = JointAux()

        self.distributions = Continous()

        if is_training_mode:
            self.policy.train()
            self.value.train()
        else:
            self.policy.eval()
            self.value.eval()

    def save_all(self, states, actions, action_means, rewards, dones, next_states):
        self.policy_memory.save_all(
            states, actions, action_means, rewards, dones, next_states
        )

    def save_list(self, states, actions, action_means, rewards, dones, next_states):
        self.policy_memory.save_list(
            states, actions, action_means, rewards, dones, next_states
        )

    # Get loss and Do backpropagation
    def training_ppo(
        self, states, actions, worker_action_means, rewards, dones, next_states
    ):
        action_mean, _ = self.policy(states)
        values = self.value(states)
        old_action_mean, _ = self.policy_old(states)
        old_values = self.value_old(states)
        next_values = self.value(next_states)

        loss = self.policy_loss.compute_loss(
            action_mean,
            self.std,
            old_action_mean,
            self.std,
            values,
            old_values,
            next_values,
            actions,
            rewards,
            dones,
            worker_action_means,
            self.std,
        )

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def training_aux(self, states):
        Returns = self.value(states).detach()

        action_mean, values = self.policy(states)
        old_action_mean, _ = self.policy_old(states)

        joint_loss = self.aux_loss.compute_loss(
            action_mean, self.std, old_action_mean, self.std, values, Returns
        )

        self.policy_optimizer.zero_grad()
        joint_loss.backward()
        self.policy_optimizer.step()

    # Update the model
    def update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.batchsize, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for (
                states,
                actions,
                action_means,
                rewards,
                dones,
                next_states,
            ) in dataloader:
                self.training_ppo(
                    states.float().to(device),
                    actions.float().to(device),
                    action_means.float().to(device),
                    rewards.float().to(device),
                    dones.float().to(device),
                    next_states.float().to(device),
                )

        # Clear the memory
        states, _, _, _, _, _ = self.policy_memory.get_all()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self):
        dataloader = DataLoader(self.aux_memory, self.batchsize, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states in dataloader:
                self.training_aux(states.float().to(device))

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self):
        torch.save(self.policy.state_dict(), os.path.join(output_path, "agent.pth"))


class Agent:
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device("cpu")

        self.memory = PolicyMemory()
        self.distributions = Continous(self.device)
        self.policy = Policy_Model(state_dim, action_dim, self.device)
        self.std = torch.ones([1, action_dim]).float().to(self.device)

        if is_training_mode:
            self.policy.train()
        else:
            self.policy.eval()

    def save_eps(self, state, action, action_mean, reward, done, next_state):
        self.memory.save_eps(state, action, action_mean, reward, done, next_state)

    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_mean, _ = self.policy(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(action_mean, self.std)
        else:
            action = action_mean

        return action.squeeze(0).cpu().numpy(), action_mean.squeeze(0).detach().numpy()

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)

    def load_weights(self):
        self.policy.load_state_dict(torch.load(os.path.join(output_path, "agent.pth"), map_location=self.device))


@ray.remote
class Runner:
    def __init__(self, experiment_type, training_mode, render, n_update, tag):
        env_name = load_ppg_env(experiment_type, visualize=render)
        self.env = gym.make(env_name)
        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update
        self.max_action = 1.0

    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.load_weights()

        for _ in range(self.n_update):
            action, action_mean = self.agent.act(self.states)

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, _ = self.env.step(action_gym)

            eps_time += 1
            total_reward += reward

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

            if done:
                self.states = self.env.reset()
                i_episode += 1

                _logger.info(
                    "Episode {} \t t_reward: {} \t time: {} \t process no: {} \t".format(
                        i_episode, total_reward, eps_time, self.tag
                    )
                )

                total_reward = 0
                eps_time = 0

        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag

def init_output(run_name):
    global _logger
    global output_path

    output_path = os.path.join("output", run_name)

    _logger = logging.basicConfig(filename=os.path.join(output_path, "train.log"), encoding='utf-8')

    _logger.info("Saving configuration in {str}/{str}.".format(output_path, "train.log"))

def main(args):
    init_output(args.run_name)

    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        f.write(yaml.safe_dump(args.__dict__, default_flow_style=False))

    continue_run = (os.path.join(output_path, "agent.pth")).exists()
    if continue_run:
        _logger.info("Found previous model in {str}. Continuing training.".format(output_path))

    if args.log_wandb:
        if has_wandb:
                wandb.init(
                    project="rug-locomotion-ppg",
                    config=args,
                    name=args.run_name,
                    id=args.run_name,
                    resume="must" if continue_run else "never"
                )
        else:
            _logger.error("You've requested to log metrics to wandb but package was not found. "
                          "Metrics not being logged to wandb, try `pip install wandb`")
    
    env_name = load_ppg_env(args.env, visualize=args.visualize)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = Learner(
        state_dim,
        action_dim,
        args.test_mode,
        args.policy_kl_range,
        args.policy_params,
        args.value_clip,
        args.entropy_coef,
        args.vf_loss_coef,
        args.batch_size,
        args.PPO_epochs,
        args.gamma,
        args.lam,
        args.learning_rate,
    )
    #############################################
    t_aux_updates = 0
    start = time.time()
    ray.init()

    try:
        runners = [
            Runner.remote(args.env, args.test_mode, args.visualize, args.n_update, i)
            for i in range(args.n_agent)
        ]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(3)

        for _ in range(1, args.n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            episode_ids = not_ready
            episode_ids.append(
                runners[tag].run_episode.remote(i_episode, total_reward, eps_time)
            )

            states, actions, action_means, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, action_means, rewards, dones, next_states)

            wandb.log({"episode": i_episode,
                       "total_reward": total_reward,
                       "eps_time": eps_time})

            learner.update_ppo()
            if t_aux_updates == args.n_aux_update:
                learner.update_aux()
                t_aux_updates = 0

            learner.save_weights()
    except KeyboardInterrupt:
        _logger.warning("Training has been stopped.")
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        _logger.info("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))

def _parse_args(parser):
    # Do we have a config file to parse?
    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of Phasic Policy Gradient for training Physics based Musculoskeletal Models."
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        metavar='FILE',
        default='',
        help="Path to the config file specifying the default parameters.."
    )
    parser.add_argument(
        "--env",
        help="The environment type used. Currently supports the healthy and prosthesis models.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Display name to use in wandb. Also used for the path to save the model. Provide a unique name."
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Whether to save output on wandb."
    )
    parser.add_argument(
        "--test-mode",
        action="store_false",
        help="Whether to save new checkpoints during learning. If used, there will be no changes made to the model."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the environment. Disregard this argument if you run this on Peregrine/Google Collab."
    )
    parser.add_argument(
        "--n-update",
        type=int,
        default=1024,
        help="How many episodes before the Policy is updated."
    )
    parser.add_argument(
        "--n-aux-update",
        type=int,
        default=5,
        help="How many episodes before the Auxiliary is updated."
    )
    parser.add_argument(
        "--n-episode",
        type=int,
        default=1000000,
        help="How many episodes you want to run."
    )
    parser.add_argument(
        "--n-agent",
        type=int,
        default=4,
        help="How many agents you want to run asynchronously."
    )
    parser.add_argument(
        "--policy-kl-range",
        type=float,
        default=0.03,
        help="TODO."
    )
    parser.add_argument(
        "--policy-params",
        type=int,
        default=5,
        help="TODO."
    )
    parser.add_argument(
        "--value-clip",
        type=float,
        default=1.0,
        help="TODO."
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="How much action randomness is introduced."
    )
    parser.add_argument(
        "--vf-loss-coef",
        type=float,
        default=1.0,
        help="How much action randomness is introduced."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="How many batches per update. The number of batches is given by the number of updates divided by the batch size."
    )
    parser.add_argument(
        "--PPO-epochs",
        type=int,
        default=10,
        help="How many PPO epochs per update."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="TODO."
    )    
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="TODO."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="TODO."
    )

    args = _parse_args(parser)
    main(args)
