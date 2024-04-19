import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class PrimalNet(nn.Module):
    """ Neural Network Module for Matching """
    def __init__(self, cfg):
        super(PrimalNet, self).__init__()
        self.cfg = cfg
        num_agents = self.cfg.num_agents
        num_hidden_nodes = self.cfg.num_hidden_nodes

        self.input_block = nn.Sequential(
            # Input Layer
            nn.Linear(2 * num_agents*num_agents, num_hidden_nodes),
            nn.LeakyReLU(),

            # Layer 1
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),

            # Layer 2
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),

            # Layer 3
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU(),

            # Layer 4
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.LeakyReLU())


        # Output Layer
        self.layer_out = nn.Linear(num_hidden_nodes, 2 * num_agents * num_agents)


    def forward(self, p, q):
        x = torch.stack([p, q], axis = -1)
        x = x.view(-1, self.cfg.num_agents * self.cfg.num_agents * 2)
        x = self.input_block(x)

        x = self.layer_out(x)
        r,t = x[:,:self.cfg.num_agents * self.cfg.num_agents],x[:,self.cfg.num_agents * self.cfg.num_agents:]
        r = r.view(-1, self.cfg.num_agents, self.cfg.num_agents)
        r = F.softplus(r)
        r = F.normalize(r, p = 1, dim = 1, eps=1e-8)

        t = t.view(-1, self.cfg.num_agents, self.cfg.num_agents)
        t = F.softplus(t)
        t = F.normalize(t, p = 1, dim = 1, eps=1e-8)

        return r,t
