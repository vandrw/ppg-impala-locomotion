import torch
from pathlib import Path

from src.ppg.model import Policy_Model, PolicyMemory, Continous

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

    def load_weights(self, path):
        self.policy.load_state_dict(torch.load(Path(path) / "agent.pth", map_location=self.device))
