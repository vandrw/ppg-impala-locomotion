from src.ppg.distribution import Continous, PolicyFunction
import torch


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

        if logprobs.isnan().any():
            print("logprobs are nan!")
            print(logprobs)
        if Old_logprobs.isnan().any():
            print("Old_logprobs are nan!")
            print(Old_logprobs)
        if Worker_logprobs.isnan().any():
            print("logprobs are nan!")
            print(Worker_logprobs)

        # Getting general advantages estimator and returns
        Advantages = self.policy_function.vtrace_generalized_advantage_estimation(
            values, rewards, next_values, dones, logprobs, Worker_logprobs
        )
        Returns = (Advantages + values).detach()
        Advantages = (
            (Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)
        ).detach()

        if Advantages.isnan().any():
            print("Advantages are nan!")
            print(Advantages)

        # Finding Surrogate Loss
        ratios = (logprobs - Old_logprobs).exp()  # ratios = old_logprobs / logprobs
        Kl = self.distributions.kl_divergence(
            Old_action_mean, old_action_std, action_mean, action_std
        )

        if Kl.isnan().any():
            print("Kl is nan!")
            print(Kl)

        pg_targets = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages - self.policy_kl_range,
        )
        pg_loss = pg_targets.mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_mean, action_std).mean()
        if dist_entropy.isnan().any():
            print("dist_entropy are nan!")
            print(dist_entropy)

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
        
        if critic_loss.isnan().any():
            print("critics_loss are nan!")
            print(critic_loss)

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
