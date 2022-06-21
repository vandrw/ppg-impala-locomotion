import torch
from pathlib import Path
from src.ppg.distribution import Continuous
from src.ppg.memory import PolicyMemory
from src.ppg.model import PolicyModel


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        initial_logstd,
        train_mode,
    ):
        self.train_mode = train_mode
        self.device = torch.device("cpu")

        self.memory = PolicyMemory()
        self.distributions = Continuous(self.device)
        self.policy = PolicyModel(state_dim, action_dim, initial_logstd, self.device)

        if train_mode:
            self.policy.train()
        else:
            self.policy.eval()

    def save_eps(self, state, action, action_mean, reward, done, next_state):
        self.memory.save_eps(state, action, action_mean, reward, done, next_state)

    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_mean, action_std, _ = self.policy(state)

        action = self.distributions.sample(action_mean, action_std)

        return (
            action.squeeze(0).cpu().numpy(),
            action_mean.squeeze(0).detach().numpy(),
            action_std.squeeze(0).detach().numpy(),
        )

    def load_weights(self, path):
        self.policy.load_state_dict(
            torch.load(Path(path) / "agent_policy.pth", map_location=self.device)
        )
