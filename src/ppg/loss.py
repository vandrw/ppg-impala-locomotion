from src.ppg.distribution import Continous, PolicyFunction
import torch


class TrulyPPO:
    def __init__(
        self,
        ppo_kl_range,
        slope_rollback,
        slope_likelihood,
        clip_range,
        vf_loss_coef,
        entropy_coef,
        gamma,
        lambd,
    ):
        self.ppo_kl_range = ppo_kl_range
        self.slope_rollback = slope_rollback
        self.slope_likelihood = slope_likelihood
        self.clip_range = clip_range
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef

        self.distributions = Continous()
        self.policy_function = PolicyFunction(gamma, lambd)

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
        # We are using positive log probs
        ratio = (logprobs - Old_logprobs).exp()
        Kl = self.distributions.kl_divergence(
            Old_action_mean, old_action_std, action_mean, action_std
        )

        pg_targets = torch.where(
            torch.logical_and(Kl >= self.ppo_kl_range, ratio * Advantages > 1 * Advantages),
            self.slope_likelihood * ratio * Advantages - self.slope_rollback * Kl,
            ratio * Advantages,
        )
        pg_loss = pg_targets.mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_mean, action_std).mean()

        # Getting Critic loss by using Clipped critic value
        if self.clip_range is None:
            critic_loss = (Returns - values).pow(2).mean() * 0.5
        else:
            # Minimize the difference between old value and new value
            vpredclipped = old_values + torch.clamp(
                values - Old_values, -self.clip_range, self.clip_range
            )  
            vf_losses1 = (values - Returns).pow(2)
            vf_losses2 = (vpredclipped - Returns).pow(2)
            
            # Mean Squared Error
            critic_loss = torch.max(vf_losses1, vf_losses2).mean() * 0.5

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (
            (critic_loss * self.vf_loss_coef)
            - (dist_entropy * self.entropy_coef)
            - pg_loss
        )
        return loss


class JointAux:
    def __init__(self, beta_clone):
        self.distributions = Continous()
        self.beta_clone = beta_clone

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

        return aux_loss + self.beta_clone * Kl
