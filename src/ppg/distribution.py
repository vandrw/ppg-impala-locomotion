from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch

from src.ppg.model import device

class Continuous:
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

        # Clamp the logprob to avoid numerical error
        # This will keep the logprob in the range of ~(log(1.0e-8), -log(1.0e-8))
        logprob = torch.clamp(
            distribution.log_prob(value_data), 
            -18.42068, 18.42068
            )
        return logprob.float().to(self.device)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1 = Normal(mean1, std1)
        distribution2 = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(self.device)


class PolicyFunction:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.gamma = gamma
        self.lambd = lambd
        self.limit = torch.FloatTensor([1.0]).to(device)

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

        ratio = torch.min(self.limit, (worker_logprobs - learner_logprobs).sum().exp())

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        delta = ratio * delta

        for step in reversed(range(len(rewards))):
            gae = (1.0 - dones[step]) * self.gamma * self.lambd * gae
            gae = delta[step] + ratio * gae
            adv.append(gae)

        adv.reverse()

        return torch.stack(adv)