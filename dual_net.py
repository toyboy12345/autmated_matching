import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class DualNet(nn.Module):
    """ Neural Network Module for Matching """
    def __init__(self, cfg):
        super(DualNet, self).__init__()
        self.cfg = cfg
        self.num_agents = self.cfg.num_agents
        num_hidden_nodes = self.cfg.num_hidden_nodes

        self.input_block = nn.Sequential(
            # Input Layer
            nn.Linear(2 * self.num_agents*self.num_agents, num_hidden_nodes),
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
        self.layer_out = nn.Linear(num_hidden_nodes, 2*self.num_agents + self.num_agents*self.num_agents + 2*(math.factorial(self.num_agents)*self.num_agents*self.num_agents))


    def forward(self, p, q):
        a = torch.stack([p, q], axis = -1)
        a = a.view(-1, self.cfg.num_agents * self.cfg.num_agents * 2)
        a = self.input_block(a)

        a = self.layer_out(a)
        a = a.view(-1, 2*self.num_agents + self.num_agents*self.num_agents + 2*(math.factorial(self.num_agents)*self.num_agents*self.num_agents))
        x,a = a[:,:self.num_agents],a[:,self.num_agents:]
        y,a = a[:,:self.num_agents],a[:,self.num_agents:]
        x = x.view(-1, self.cfg.num_agents)
        x = F.softplus(x)
        y = y.view(-1, self.cfg.num_agents)
        y = F.softplus(y)

        z,a = (a[:,:self.num_agents*self.num_agents]).view(-1,self.num_agents*self.num_agents),a[:,self.num_agents*self.num_agents:]
        z = z.view(-1, self.cfg.num_agents, self.cfg.num_agents)
        z = F.sigmoid(z)

        facto = math.factorial(self.num_agents)
        u,a = a[:,:facto*self.num_agents*self.num_agents], a[:,facto*self.num_agents*self.num_agents:]
        u = u.view(-1,self.num_agents,facto,self.num_agents)
        u = F.softplus(u)

        v = a
        v = v.view(-1,self.num_agents,facto,self.num_agents)
        v = F.softplus(v)

        return x,y,z,u,v
