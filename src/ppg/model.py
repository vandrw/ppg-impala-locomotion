import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, initial_logstd, myDevice=None):
        super(PolicyModel, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                layer_init(nn.Linear(state_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh()
              ).float().to(self.device)

        # The muscle activations in OpenSim are constrained to [0, 1]
        # Therefore, we use a Sigmoid function as the output.
        self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(256, action_dim), std=0.01),
                nn.Sigmoid()
              ).float().to(self.device)

        self.actor_logstd = nn.parameter.Parameter(
            torch.ones([1, action_dim]) * initial_logstd
            ).float().to(self.device)
            
        self.critic_layer = nn.Sequential(
                layer_init(nn.Linear(256, 1), std=1.0)
              ).float().to(self.device)

    def forward(self, states):
        # Get action and value
        x = self.nn_layer(states)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        return action_mean, action_std, self.critic_layer(x)


class ValueModel(nn.Module):
    def __init__(self, state_dim, myDevice=None):
        super(ValueModel, self).__init__()

        self.device = myDevice if myDevice != None else device
        self.nn_layer = nn.Sequential(
                layer_init(nn.Linear(state_dim, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 128)),
                nn.Tanh(),
                layer_init(nn.Linear(128, 1), std=1.0)
              ).float().to(self.device)

    def forward(self, states):
        # Get value only
        return self.nn_layer(states)