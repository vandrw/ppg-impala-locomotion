from torch.utils.data import Dataset
import numpy as np

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
