from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

from src.ppg.distribution import Continuous
from src.ppg.loss import JointAux, TrulyPPO
from src.ppg.memory import AuxMemory, PolicyMemory
from src.ppg.model import PolicyModel, ValueModel
from src.ppg.model import device

from pathlib import Path


class Learner:
    def __init__(
        self,
        state_dim,
        action_dim,
        train_mode,
        ppo_kl_range,
        slope_rollback,
        slope_likelihood,
        val_clip_range,
        entropy_coef,
        vf_loss_coef,
        ppo_batchsize,
        ppo_epochs,
        aux_batchsize,
        aux_epochs,
        beta_clone,
        gamma,
        lambd,
        learning_rate,
        initial_logstd,
    ):
        self.ppo_batchsize = ppo_batchsize
        self.ppo_epochs = ppo_epochs
        self.aux_batchsize = aux_batchsize
        self.aux_epochs = aux_epochs

        self.policy = PolicyModel(state_dim, action_dim, initial_logstd)
        self.policy_old = PolicyModel(state_dim, action_dim, initial_logstd)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)

        self.value = ValueModel(state_dim)
        self.value_old = ValueModel(state_dim)
        self.value_optimizer = Adam(self.value.parameters(), lr=learning_rate, eps=1e-5)

        self.policy_memory = PolicyMemory()
        self.policy_loss = TrulyPPO(
            ppo_kl_range,
            slope_rollback,
            slope_likelihood,
            val_clip_range,
            vf_loss_coef,
            entropy_coef,
            gamma,
            lambd,
        )

        self.aux_memory = AuxMemory()
        self.aux_loss = JointAux(beta_clone)

        self.distributions = Continuous()
        
        if train_mode:
            self.policy.train()
            self.value.train()
        else:
            self.policy.eval()
            self.value.eval()

    def save_all(
        self, states, actions, action_means, action_std, rewards, dones, next_states
    ):
        self.policy_memory.save_all(
            states, actions, action_means, rewards, dones, next_states
        )
        self.policy_memory.update_std(action_std)

    # Get loss and Do backpropagation
    def training_ppo(
        self,
        states,
        actions,
        worker_action_means,
        worker_std,
        rewards,
        dones,
        next_states,
    ):
        action_mean, action_std, _ = self.policy(states)
        values = self.value(states)
        old_action_mean, old_action_std, _ = self.policy_old(states)
        old_values = self.value_old(states)
        next_values = self.value(next_states)

        loss = self.policy_loss.compute_loss(
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
        )

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def training_aux(self, states):
        Returns = self.value(states).detach()

        action_mean, action_std, values = self.policy(states)
        old_action_mean, old_action_std, _ = self.policy_old(states)

        joint_loss = self.aux_loss.compute_loss(
            action_mean, action_std, old_action_mean, old_action_std, values, Returns
        )

        self.policy_optimizer.zero_grad()
        joint_loss.backward()
        self.policy_optimizer.step()

    # Update the model
    def update_ppo(self):
        dataloader = DataLoader(self.policy_memory, self.ppo_batchsize, shuffle=False)
        # Optimize policy for K epochs:
        for _ in range(self.ppo_epochs):
            for (
                states,
                actions,
                action_means,
                action_std,
                rewards,
                dones,
                next_states,
            ) in dataloader:
                self.training_ppo(
                    states.float().to(device),
                    actions.float().to(device),
                    action_means.float().to(device),
                    action_std.float().to(device),
                    rewards.float().to(device),
                    dones.float().to(device),
                    next_states.float().to(device),
                )

        # Clear the memory
        states, _, _, _, _, _, _ = self.policy_memory.get_all()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self):
        dataloader = DataLoader(self.aux_memory, self.aux_batchsize, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.aux_epochs):
            for states in dataloader:
                self.training_aux(states.float().to(device))

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self, path):
        torch.save(self.policy.state_dict(), Path(path) / "agent_policy.pth")
        torch.save(self.value.state_dict(), Path(path) / "agent_value.pth")

    def load_weights(self, path):
        self.policy.load_state_dict(
            torch.load(Path(path) / "agent_policy.pth", map_location=device)
        )
        self.value.load_state_dict(
            torch.load(Path(path) / "agent_value.pth", map_location=device)
        )
