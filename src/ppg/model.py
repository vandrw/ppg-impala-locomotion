import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, initial_logstd, myDevice=None):
        super(Policy_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.LeakyReLU(0.01),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.01)
              ).float().to(self.device)

        # The muscle activations in OpenSim are constrained to [0, 1]
        # Therefore, we use a Sigmoid function as the output.
        self.actor_mean = nn.Sequential(
                nn.Linear(256, action_dim),
                nn.Sigmoid()
              ).float().to(self.device)

        self.actor_logstd = nn.parameter.Parameter(
            torch.ones([1, action_dim]) * initial_logstd
            ).float().to(self.device)
            
        self.critic_layer = nn.Sequential(
                nn.Linear(256, 1)
              ).float().to(self.device)

    def forward(self, states):
        # Get action and value
        x = self.nn_layer(states)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        return action_mean, action_std, self.critic_layer(x)


class Value_Model(nn.Module):
    def __init__(self, state_dim, myDevice=None):
        super(Value_Model, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LeakyReLU(0.01),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.01),
                nn.Linear(128, 1)
              ).float().to(self.device)

    def forward(self, states):
        # Get value only
        return self.nn_layer(states)