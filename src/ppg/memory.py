from torch.utils.data import Dataset
import numpy as np
import torch


class PolicyMemory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_means = []
        self.action_std = []
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
            np.array(self.action_std, dtype=np.float32),
            np.array([self.rewards[idx]], dtype=np.float32),
            np.array([self.dones[idx]], dtype=np.float32),
            np.array(self.next_states[idx], dtype=np.float32),
        )

    def get_all(self):
        return (
            self.states,
            self.actions,
            self.action_means,
            self.action_std,
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

    def save_eps(self, state, action, action_mean, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.action_means.append(action_mean)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def update_std(self, action_std):
        self.action_std = action_std

    def norm_states(self, mean, std, clip):
        self.states = np.clip((self.states - mean) / std, -clip, clip).tolist()

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.action_means[:]
        del self.action_std
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


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape=(), epsilon=1e-4, device=torch.device("cpu")):
        self.device = device
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = torch.Tensor([epsilon]).float().to(device)

    def update(self, batch):
        batch = torch.FloatTensor(batch).to(self.device).detach()
        batch_mean = torch.mean(batch, axis=0)
        batch_var = torch.var(batch, axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        self.mean, self.var, self.count = (new_mean, new_var, new_count)

    def norm_state(self, state: torch.FloatTensor, clip: float):
        return torch.clamp((state - self.mean) / self.var, -clip, clip)

