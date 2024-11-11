import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """ Neural Network Module for Matching """
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        num_agents = self.cfg.num_agents
        num_hidden_nodes = self.cfg.num_hidden_nodes
        
        self.input_block = nn.Sequential( 
            # Input Layer
            nn.Linear(2 * (num_agents**2), num_hidden_nodes),
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
        self.layer_out_r = nn.Linear(num_hidden_nodes, (num_agents + 1) * num_agents)
        self.layer_out_c = nn.Linear(num_hidden_nodes, num_agents * (num_agents + 1))
        
           
    def forward(self, p, q):
        
        p = F.relu(p)
        q = F.relu(q)
        
        x = torch.stack([p, q], axis = -1)
        x = x.view(-1, self.cfg.num_agents * self.cfg.num_agents * 2)
        x = self.input_block(x)
        
        mask_p = (p > 0).to(p.dtype)
        mask_p = F.pad(mask_p, (0, 0, 0, 1, 0, 0), mode='constant', value=1)

        mask_q = (q > 0).to(q.dtype)
        mask_q = F.pad(mask_q, (0, 1, 0, 0, 0, 0), mode='constant', value=1)
                
        x_r = self.layer_out_r(x)
        x_r = x_r.view(-1, self.cfg.num_agents + 1, self.cfg.num_agents)
        x_c = self.layer_out_c(x)
        x_c = x_c.view(-1, self.cfg.num_agents, self.cfg.num_agents + 1)
                
        x_r = F.softplus(x_r) * mask_p
        x_c = F.softplus(x_c) * mask_q
        
        x_r = F.normalize(x_r, p = 1, dim = 1, eps=1e-8)
        x_c = F.normalize(x_c, p = 1, dim = 2, eps=1e-8)
        
        return torch.min(x_r[:, :-1, :], x_c[:, :, :-1])