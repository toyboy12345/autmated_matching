import os
import sys
import time
import logging
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

from data import Data
from net import Net

# Hyperparameters
# LR - [1e-3, 5e-3]

class HParams:
    def __init__(self, num_agents = 3,
                 batch_size = 1024, num_hidden_layers = 4, num_hidden_nodes = 256, lr = 5e-3, epochs = 50000,
                 print_iter = 100, val_iter = 1000, num_val_batches = 200,
                 prob = 0.5, lambd = 0.1, corr = 0, seed = 0, device = "cuda:0",
                 use_lagr = None, lagr_mult = None, lagr_iter = None, rho = None, st_zero_one = False):
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.num_hidden_layers = 4
        self.num_hidden_nodes = 256
        self.lr = 5e-3
        self.epochs = 50000
        self.print_iter = 100
        self.val_iter = 1000
        self.save_iter = self.epochs - 1
        self.num_val_batches = 200

        # Higher probability => More truncation
        self.prob = prob
        # Higher lambd => More stability
        self.lambd = lambd
        # Correlation of rankings
        self.corr = corr
        # Run seed
        self.seed = seed

        self.device = device

        self.use_lagr = use_lagr # s for stability, r for regret
        self.lagr_mult = lagr_mult
        self.lagr_iter = lagr_iter
        self.rho = rho
        self.st_zero_one = st_zero_one

def algo(p_,q_):
    p,q = 4-p_*3,4-q_*3
    r = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if p[i,j] == 1 and q[i,j] == 1:
                r[i,j] = 1
                i1,j1 = i,j
                break

    p1,q1 = p[np.delete(np.arange(3),i1),:][:,np.delete(np.arange(3),j1)],q[np.delete(np.arange(3),i1),:][:,np.delete(np.arange(3),j1)]
    p1,q1 = np.argsort(p1)+1,np.argsort(q1,0)+1
    for i in range(2):
        for j in range(2):
            if p1[i,j] == 1 and q1[i,j] == 1:
                i2,j2 = i+int(bool(i1+i1*(1-i) <= 1)), j+int(bool(j1+j1*(1-j) <= 1))
                i3,j3 = 1-i+int(bool(i1+i1*i <= 1)), 1-j+int(bool(j1+j1*j <= 1))

                r[i2,j2] = 1
                r[i3,j3] = 1
                return r
    i2,j2 = np.delete(np.arange(3),i1)[0],np.delete(np.arange(3),j1)[1]
    i3,j3 = np.delete(np.arange(3),i1)[1],np.delete(np.arange(3),j1)[0]

    r[i2,j2] = 1
    r[i3,j3] = 1
    return r


def algo_batch(p,q,model):
    P,Q = p.to('cpu').detach().numpy().copy(),q.to('cpu').detach().numpy().copy()
    bacth_size = p.shape[0]
    r = torch.zeros(p.shape,device="cuda:0")
    idx = torch.where(torch.max(torch.max(p*q,1).values,1).values == 1, 1, 0)
    idx_ = idx*(-1)+1

    if torch.sum(idx) >= 1:
        r1 = torch.Tensor(np.array(list((map(algo,P[idx.to(bool).to('cpu').detach().numpy()],Q[idx.to(bool).to('cpu').detach().numpy()]))))).to("cuda:0")
        r[idx.to(bool)] = r1

    if torch.sum(idx_) >= 1:
        r2 = model(p[idx_.to(bool)],q[idx_.to(bool)]).to("cuda:0")
        r[(idx*(-1)+1).to(bool)] = r2

    return torch.reshape(r,(-1,3,3))

# Stability Violation
def compute_st(r, p, q, use_lagr = False, zero_one = False):
    if not zero_one: 
        wp = F.relu(p[:, :, None, :] - p[:, :, :, None])
        wq = F.relu(q[:, :, None, :] - q[:, None, :, :], 0)
    else:
        wp = torch.where(F.relu(p[:, :, None, :] - p[:, :, :, None])>0,1,0).to(torch.float)
        wq = torch.where(F.relu(q[:, :, None, :] - q[:, None, :, :], 0)>0,1,0).to(torch.float)

    t = (1 - torch.sum(r, dim = 1, keepdim = True))
    s = (1 - torch.sum(r, dim = 2, keepdim = True))
    rgt_1 = torch.einsum('bjc,bijc->bic', r, wq) + t * F.relu(q)
    rgt_2 = torch.einsum('bia,biac->bic', r, wp) + s * F.relu(p)
    regret =  rgt_1 * rgt_2
    if use_lagr:
        regret2 = torch.square(regret)
        return regret.sum(-1).sum(-1).mean(),regret2.sum(-1).sum(-1).mean()
    else:
        return regret.sum(-1).sum(-1).mean()

# IR Violation
def compute_ir(r, p, q):
    ir_1 = r * F.relu(-q)
    ir_2 = r * F.relu(-p)
    ir = ir_1 + ir_2
    return ir.sum(-1).sum(-1).mean()/p.shape[1]

# FOSD Violation
def compute_ic_FOSD(model, G, r, p, q, r_mult = 1, lagr_mult = None, use_lagr = False, include_truncation = False):
    cfg = G.cfg
    num_agents = cfg.num_agents
    device = cfg.device

    P,Q = p.to('cpu').detach().numpy().copy(),q.to('cpu').detach().numpy().copy()

    IC_viol_P = torch.zeros(num_agents).to(device)
    IC_viol_Q = torch.zeros(num_agents).to(device)

    discount = torch.Tensor(((r_mult) ** np.arange(num_agents))).to(device)

    for agent_idx in range(num_agents):

        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = include_truncation)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        r_mis = r_mis.view(*P_mis.shape)

        r_diff = (r_mis[:, :, agent_idx, :] - r[:, None, agent_idx, :])*(p[:, None, agent_idx, :] > 0).float()
        _, idx = torch.sort(-p[:, agent_idx, :])
        idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)

        fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx) * discount, -1)
        IC_viol_P[agent_idx] = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)

        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = include_truncation)
        p_mis, q_mis = torch.Tensor(P_mis).to(device), torch.Tensor(Q_mis).to(device)
        r_mis = model(p_mis.view(-1, num_agents, num_agents), q_mis.view(-1, num_agents, num_agents))
        r_mis = r_mis.view(*Q_mis.shape)

        r_diff = (r_mis[:, :, :, agent_idx] - r[:, None, :, agent_idx])*(q[:, None, :, agent_idx] > 0).float()
        _, idx = torch.sort(-q[:, :, agent_idx])
        idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)

        fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx) * discount, -1)
        IC_viol_Q[agent_idx] = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)
    
    ic_viol = torch.cat((IC_viol_P,IC_viol_Q))

    if use_lagr:
        lagr_mult = torch.Tensor(lagr_mult).to(device)
        IC_viol = torch.dot(lagr_mult,ic_viol)
        IC_viol2 = torch.sum(torch.square(ic_viol))*0.5*cfg.rho
        return IC_viol, IC_viol2, ic_viol

    else:
        IC_viol = ic_viol.mean()
        return IC_viol
        
def compute_anonimity_violation(model,G,r,p,q):
    av_P = torch.zeros(3)
    av_Q = torch.zeros(3)
    for i in range(2):
        for j in range(i+1,3):
            idx = [0,1,2]
            idx[i],idx[j] = idx[j],idx[i]
            p_P,q_P = p[:,idx,:],q[:,idx,:]
            p_Q,q_Q = p[:,:,idx],q[:,:,idx]
            r_P = model(p_P,q_P)
            r_Q = model(p_Q,q_Q)
            av_P[i+j-1] = torch.sum(torch.abs(r-r_P[:,idx,:]))
            av_Q[i+j-1] = torch.sum(torch.abs(r-r_Q[:,:,idx]))
    av = torch.cat((av_P,av_Q))
    return av

def eval_model(model, G, P, Q, rtn = False, include_truncation = False):
    num_agents = G.cfg.num_agents
    device = G.cfg.device
    st_zero_one = G.cfg.st_zero_one
    p,q = torch.Tensor(P).to(device),torch.Tensor(Q).to(device)
    r = model(p,q)
    ic_loss = compute_ic_FOSD(model,G,r,p,q,use_lagr=False,include_truncation=include_truncation)
    st_loss = compute_st(r,p,q,zero_one=st_zero_one)
    ir_loss = compute_ir(r,p,q)

    print(f"ic_loss: {ic_loss:.10f}, st_loss: {st_loss:.10f}, ir_loss: {ir_loss:.10f}")
    if rtn:
        return ic_loss, st_loss, ir_loss

def train_net(cfg, G, model, include_truncation = None):
    # File names
    root_dir = os.path.join("experiments", "agents_%d"%(cfg.num_agents), "corr_%.2f"%(cfg.corr))
    log_fname = os.path.join(root_dir, "LOG_%d_lambd_%f_prob_%.2f_corr_%.2f.txt"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
    model_path = os.path.join(root_dir, "MODEL_%d_lambd_%f_prob_%.2f_corr_%.2f"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
    os.makedirs(root_dir, exist_ok=True)

    # # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    num_agents = cfg.num_agents

    if not logger.hasHandlers():
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

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr = cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10000,25000], gamma=0.5)

    # Trainer
    tic = time.time()
    i = 0

    lagr_mult = cfg.lagr_mult

    while i < cfg.epochs:

        # Reset opt
        opt.zero_grad()
        model.train()

        # Inference
        P, Q = G.generate_batch(cfg.batch_size)
        p, q = torch.Tensor(P).to(cfg.device), torch.Tensor(Q).to(cfg.device)
        r = model(p, q)

        # Compute loss
        if cfg.use_lagr == "s":
            st_loss,st_loss2 = compute_st(r, p, q, use_lagr=True, zero_one=cfg.st_zero_one)
            ic_loss = compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation)
            total_loss = st_loss + st_loss2 + ic_loss

            if (i>0) and (i % cfg.lagr_iter == 0):
                lagr_mult += cfg.rho * st_loss.item()
                logger.info("[lambda]: %f"%(lagr_mult))

        elif cfg.use_lagr == "r":
            st_loss = compute_st(r,p,q,zero_one=cfg.st_zero_one)
            ic_loss, ic_loss2, ic_losses = compute_ic_FOSD(model, G, r, p, q, lagr_mult=lagr_mult, use_lagr=True, include_truncation = include_truncation)
            total_loss = st_loss + ic_loss + ic_loss2
            total_loss.backward(retain_graph = True)

            if (i>0) and (i % cfg.lagr_iter == 0):
                lagr_mult = lagr_mult + ic_losses.to('cpu').detach().numpy().copy() * cfg.rho
                logger.info(f"[lambda]: {lagr_mult.tolist()}")

        else:
            st_loss = compute_st(r,p,q,zero_one=st_zero_one)
            ic_loss = compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation)

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
            torch.save(model, "deep-matching/models/model_tmp.pth")

        if ((i>0) and (i% cfg.val_iter == 0)) or i == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_st_loss = 0.0
                val_ic_loss = 0.0
                for j in range(cfg.num_val_batches):
                    P, Q = G.generate_batch(cfg.batch_size)
                    p, q = torch.Tensor(P).to(cfg.device), torch.Tensor(Q).to(cfg.device)
                    r = model(p, q)
                    st_loss = compute_st(r, p, q, zero_one=cfg.st_zero_one)
                    ic_loss = compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation)

                    val_st_loss += st_loss.item()
                    val_ic_loss += ic_loss.item()
                logger.info("\t[VAL-ITER]: %d, [ST-Loss]: %f, [IC-Loss]: %f"%(i, val_st_loss/cfg.num_val_batches, val_ic_loss/cfg.num_val_batches))

        i += 1
