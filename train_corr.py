import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

from data import Data
from net import Net


parser = argparse.ArgumentParser()

parser.add_argument('-n', '--num_agents', action='store',
                    dest='num_agents', required=True, type=int,
                    help='Num Agents')

parser.add_argument('-l', '--lambd', action='store',
                    dest='lambd', required=True, type=float,
                    help='Lambda')

parser.add_argument('-p', '--prob', action='store',
                    dest='prob', default=0.2 , type=float,
                    help='Truncation Probability')

parser.add_argument('-s', '--seed', action='store',
                    dest='seed', default=0, type=int,
                    help='Random Seed')

parser.add_argument('-c', '--corr', action='store',
                    dest='corr', required=True, type=float,
                    help='Correlation Probability')

cmd_args = parser.parse_args()

# Hyperparameters
class HParams:
    def __init__(self):
        self.num_agents = cmd_args.num_agents
        self.batch_size = 1024
        self.num_hidden_layers = 4
        self.num_hidden_nodes = 256
        self.lr = 2e-3
        self.epochs = 50000
        self.print_iter = 100
        self.val_iter = 1000
        self.save_iter = self.epochs - 1
        self.num_val_batches = 200
        
        # Higher probability => More truncation
        self.prob = cmd_args.prob
        # Higher lambd => More stability
        self.lambd = cmd_args.lambd
        # Correlation of rankings
        self.corr = cmd_args.corr
        # Run seed
        self.seed = cmd_args.seed
        
        
# Initialize config
cfg = HParams()
device = "cuda:0"
np.random.seed(cfg.seed)

# File names
root_dir = os.path.join("experiments", "agents_%d"%(cfg.num_agents), "corr_%.2f"%(cfg.corr))
log_fname = os.path.join(root_dir, "LOG_%d_lambd_%f_prob_%.2f_corr_%.2f.txt"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
model_path = os.path.join(root_dir, "MODEL_%d_lambd_%f_prob_%.2f_corr_%.2f"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
os.makedirs(root_dir, exist_ok=True)

# Logger 
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

handler = logging.FileHandler(log_fname, 'w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Data
G = Data(cfg)

# Neural Network
model = Net(cfg)
model.to(device)

def torch_var(x): return torch.Tensor(x).to(device)

# Loss functions
def compute_st(r, p, q):        
    wp = F.relu(p[:, :, None, :] - p[:, :, :, None])
    wq = F.relu(q[:, :, None, :] - q[:, None, :, :], 0)  
    t = (1 - torch.sum(r, dim = 1, keepdim = True))
    s = (1 - torch.sum(r, dim = 2, keepdim = True))
    rgt_1 = torch.einsum('bjc,bijc->bic', r, wq) + t * F.relu(q)
    rgt_2 = torch.einsum('bia,biac->bic', r, wp) + s * F.relu(p)
    regret =  rgt_1 * rgt_2 
    return regret.sum(-1).sum(-1).mean()/cfg.num_agents

def compute_ir(r, p, q):
    ir_1 = r * F.relu(-q)
    ir_2 = r * F.relu(-p)
    ir = ir_1 + ir_2
    return ir.sum(-1).sum(-1).mean()/cfg.num_agents

def compute_ic_FOSD(r, p, q, P, Q, r_mult = 1):
            
    IC_viol_P = torch.zeros(cfg.num_agents).to(device)
    IC_viol_Q = torch.zeros(cfg.num_agents).to(device)
    
    discount = torch_var((r_mult) ** np.arange(cfg.num_agents))

    for agent_idx in range(cfg.num_agents):

        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = True)
        p_mis, q_mis = torch_var(P_mis), torch_var(Q_mis)
        r_mis = model(p_mis.view(-1, cfg.num_agents, cfg.num_agents), q_mis.view(-1, cfg.num_agents, cfg.num_agents))
        r_mis = r_mis.view(*P_mis.shape)

        r_diff = (r_mis[:, :, agent_idx, :] - r[:, None, agent_idx, :])*(p[:, None, agent_idx, :] > 0).float()
        _, idx = torch.sort(-p[:, agent_idx, :])
        idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)
        
        fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx) * discount, -1)
        IC_viol_P[agent_idx] = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)

        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = True)
        p_mis, q_mis = torch_var(P_mis), torch_var(Q_mis)
        r_mis = model(p_mis.view(-1, cfg.num_agents, cfg.num_agents), q_mis.view(-1, cfg.num_agents, cfg.num_agents))
        r_mis = r_mis.view(*Q_mis.shape)
        
        r_diff = (r_mis[:, :, :, agent_idx] - r[:, None, :, agent_idx])*(q[:, None, :, agent_idx] > 0).float()
        _, idx = torch.sort(-q[:, :, agent_idx])
        idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)
        
        fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx) * discount, -1)
        IC_viol_Q[agent_idx] = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)

    IC_viol = (IC_viol_P.mean() + IC_viol_Q.mean())*0.5
    return IC_viol


# Optimizer
opt = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10000,25000], gamma=0.5)

# Trainer
tic = time.time()
i = 0
while i < cfg.epochs:
  
    # Reset opt
    opt.zero_grad()
    model.train()
    
    # Inference
    P, Q = G.generate_batch_with_corr(cfg.batch_size)    
    p, q = torch_var(P), torch_var(Q)
    r = model(p, q)
    
    # Compute loss
    st_loss = compute_st(r, p, q)
    
    ic_loss = compute_ic_FOSD(r, p, q, P, Q)
        
    total_loss = st_loss * (cfg.lambd) + ic_loss * (1 - cfg.lambd)
    total_loss.backward() 
    
    opt.step()
    scheduler.step()
    t_elapsed = time.time() - tic

    
    # Validation  
    if i% cfg.print_iter == 0 or i == cfg.epochs - 1:
        logger.info("[TRAIN-ITER]: %d, [Time-Elapsed]: %f, [Total-Loss]: %f"%(i, t_elapsed, total_loss.item()))
        logger.info("[Stability-Viol]: %f, [IC-Viol]: %f"%(st_loss.item(), ic_loss.item()))
        
    if (i>0) and (i % cfg.save_iter == 0) or i == cfg.epochs - 1:
        torch.save(model.state_dict(), model_path)
        
    if ((i>0) and (i% cfg.val_iter == 0)) or i == cfg.epochs - 1:
        model.eval()
        with torch.no_grad():
            val_st_loss = 0.0
            val_ic_loss = 0.0
            for j in range(cfg.num_val_batches):
                P, Q = G.generate_batch_with_corr(cfg.batch_size)
                p, q = torch_var(P), torch_var(Q)
                r = model(p, q)
                st_loss = compute_st(r, p, q)
                ic_loss = compute_ic_FOSD(r, p, q, P, Q)
                val_st_loss += st_loss.item()
                val_ic_loss += ic_loss.item()
            logger.info("\t[VAL-ITER]: %d, [ST-Loss]: %f, [IC-Loss]: %f"%(i, val_st_loss/cfg.num_val_batches, val_ic_loss/cfg.num_val_batches))
            
    i += 1