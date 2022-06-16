import torch
from pathlib import Path
from src.ppg.distribution import Continous
from src.ppg.memory import PolicyMemory, RunningMeanStd
from src.ppg.model import PolicyModel


class Agent:
    def __init__(self, state_dim, action_dim, initial_logstd, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device("cpu")

        self.memory = PolicyMemory()
        self.distributions = Continous(self.device)
        self.policy = PolicyModel(state_dim, action_dim, initial_logstd, self.device)
        self.normalizer = RunningMeanStd(state_dim, device=self.device)

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
        state = self.normalizer.norm_state(state, clip=3)
        action_mean, action_std, _ = self.policy(state)

        # We don't need to sample the action in Test Mode. We only sample the action
        # in Training Mode to explore the actions.
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(action_mean, action_std)
        else:
            return action_mean.squeeze(0).detach().numpy(), None, None

        return (
            action.squeeze(0).cpu().numpy(),
            action_mean.squeeze(0).detach().numpy(),
            action_std.squeeze(0).detach().numpy(),
        )

    def load_weights(self, path):
        self.policy.load_state_dict(
            torch.load(Path(path) / "agent_policy.pth", map_location=self.device)
        )

    def load_normalizer(self, path):
        norm_dict = torch.load(Path(path) / "normalizer.pth", map_location=self.device)
        self.normalizer.mean = norm_dict["mean"]
        self.normalizer.var = norm_dict["var"]
        self.normalizer.var = norm_dict["count"]
