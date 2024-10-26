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
                 use_lagr = None, lagr_mult = None, lagr_iter = None, rho = None, st_zero_one = False, anonymity_vio = False):
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

        self.anonymity_vio = anonymity_vio

def algo(p,q):
    device = p.device
    # 自分のこと好きな人リスト
    faved_p,faved_q = [torch.where(q[i,:]==1)[0].tolist() for i in range(3)],[torch.where(p[:,i]==1)[0].tolist() for i in range(3)]

    # 自分のこと一番好きな人の中で一番好きな人よりも好きな人リスト
    fav_p = torch.Tensor([[1,1,1] if len(p[i,faved_p[i]])==0 else torch.where(p[i,:] >= torch.max(p[i,faved_p[i]]),1,0).tolist() for i in range(3)]).to(device)
    fav_q = torch.Tensor([[1,1,1] if len(q[faved_q[i],i])==0 else torch.where(q[:,i] >= torch.max(q[faved_q[i],i]),1,0).tolist() for i in range(3)]).to(device).transpose(1,0)

    avl = torch.where(fav_p+fav_q == 2,1,0)

    nummin = torch.Tensor([[torch.min(torch.sum(avl[:,i]),torch.sum(avl[j,:])) if avl[j,i] else 4 for i in range(3)] for j in range(3)]).to(device)

    r = torch.zeros((3,3),device=device)
    idx,idy = np.arange(3),np.arange(3)
    if torch.min(nummin) == 1:
        for x,y in zip(torch.where(nummin==1)[0],torch.where(nummin==1)[1]):
            r[x,y] = 1
        idx,idy = np.delete(idx,torch.where(nummin==1)[0]),np.delete(idy,torch.where(nummin==1)[1])
        if len(idx) == 1:
            r[idx[0],idy[0]] = 1
        elif len(idx) == 2:
            p_,q_ = torch.argsort(p[idx,:][:,idy],1),torch.argsort(q[idx,:][:,idy],0)
            # 自分のこと好きな人リスト
            faved_p_,faved_q_ = [torch.where(q_[i,:]==1)[0].tolist() for i in range(2)],[torch.where(p_[:,i]==1)[0].tolist() for i in range(2)]

            # 自分のこと一番好きな人の中で一番好きな人よりも好きな人リスト
            fav_p_ = torch.Tensor([[1,1] if len(p_[i,faved_p_[i]])==0 else torch.where(p_[i,:] >= torch.max(p_[i,faved_p_[i]]),1,0).tolist() for i in range(2)]).to(device)
            fav_q_ = torch.Tensor([[1,1] if len(q_[faved_q_[i],i])==0 else torch.where(q_[:,i] >= torch.max(q_[faved_q_[i],i]),1,0).tolist() for i in range(2)]).to(device).transpose(1,0)

            avl_ = torch.where(fav_p_+fav_q_ == 2,1,0)

            nummin_ = torch.Tensor([[torch.min(torch.sum(avl_[:,i]),torch.sum(avl_[j,:])) if avl_[j,i] else 3 for i in range(2)] for j in range(2)]).to(device)
            if torch.min(nummin_) == 1:
                for x,y in zip(torch.where(nummin_==1)[0],torch.where(nummin_==1)[1]):
                    r[idx[x],idy[y]] = 1
            else:
                for x in idx:
                    for y in idy:
                        r[x,y] = 1/2


    elif torch.min(nummin) == 2:
        idx_cnt,idy_cnt = np.array([0]*3),np.array([0]*3)
        for x,y in zip(torch.where(nummin==2)[0],torch.where(nummin==2)[1]):
            r[x,y] = 1/2
            idx_cnt[x] += 1
            idy_cnt[y] += 1
        idx,idy = np.delete(idx,np.where(idx_cnt==2)),np.delete(idy,np.where(idy_cnt==2))
        if len(idx) == 2:
            p_,q_ = torch.argsort(p[idx,:][:,idy],1),torch.argsort(q[idx,:][:,idy],0)
            # 自分のこと好きな人リスト
            faved_p_,faved_q_ = [torch.where(q_[i,:]==1)[0].tolist() for i in range(2)],[torch.where(p_[:,i]==1)[0].tolist() for i in range(2)]

            # 自分のこと一番好きな人の中で一番好きな人よりも好きな人リスト
            fav_p_ = torch.Tensor([[1,1] if len(p_[i,faved_p_[i]])==0 else torch.where(p_[i,:] >= torch.max(p_[i,faved_p_[i]]),1,0).tolist() for i in range(2)]).to(device)
            fav_q_ = torch.Tensor([[1,1] if len(q_[faved_q_[i],i])==0 else torch.where(q_[:,i] >= torch.max(q_[faved_q_[i],i]),1,0).tolist() for i in range(2)]).to(device).transpose(1,0)

            avl_ = torch.where(fav_p_+fav_q_ == 2,1,0)

            nummin_ = torch.Tensor([[torch.min(torch.sum(avl_[:,i]),torch.sum(avl_[j,:])) if avl_[j,i] else 3 for i in range(2)] for j in range(2)]).to(device)
            if torch.min(nummin_) == 1:
                for x,y in zip(torch.where(nummin_==1)[0],torch.where(nummin_==1)[1]):
                    r[idx[x],idy[y]] = 1/2
            else:
                for x in idx:
                    for y in idy:
                        r[x,y] = 1/4

    else:
        r += 1/3

    return r

def algo_batch(p,q):
    return torch.stack(list((map(algo,p,q))),dim=0).to(p.device)

def algo2_batch(p,q):
    return torch.stack(list((map(algo2,p,q))),dim=0).to(p.device)

def algo2(p,q):
    r = torch.zeros((3,3),device=p.device)
    for x1 in range(2):
        for x2 in range(x1+1,3):
            for y1 in range(2):
                for y2 in range(y1+1,3):
                    p_,q_ = torch.argsort(p[:,[x1,x2]][[y1,y2],:],dim=1),torch.argsort(q[:,[x1,x2]][[y1,y2],:],dim=0)
                    r_ = algo_mini(p_,q_)
                    for i,x in enumerate([x1,x2]):
                        for j,y in enumerate([y1,y2]):
                            r[y,x] += r_[j,i]
    r = r/6 
    return r

def algo_mini(p,q):
    r = torch.zeros((2,2),device=p.device)
    if torch.max(p*q) == 1:
        i1,j1 = torch.where(p*q == 1)[0][0].item(),torch.where(p*q == 1)[1][0].item()
        i2,j2 = 1-i1,1-j1
        r[i1,j1] += 1
        r[i2,j2] += 1
    else:
        r += 1/2
    return r

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
def compute_ic_FOSD(model, G, r, p, q, r_mult = 1, include_truncation = False):
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

    return ic_viol
        
def compute_anonimity_violation(model,G,r,p,q):
    av_P = torch.zeros(3,device=G.cfg.device)
    av_Q = torch.zeros(3,device=G.cfg.device)
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
            ic_loss = compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation)
            anon_vio = compute_anonimity_violation(model,G,r,p,q)
            if cfg.anonymity_vio:
                loss = torch.cat((ic_loss,anon_vio))
                total_loss = st_loss + torch.dot(torch.Tensor(lagr_mult).to(cfg.device),loss) + torch.sum(torch.square(loss))*cfg.rho
            else:
                total_loss = st_loss + torch.dot(torch.Tensor(lagr_mult).to(cfg.device),ic_loss) + torch.sum(torch.square(ic_loss))*cfg.rho
            total_loss.backward(retain_graph = True)

            if (i>0) and (i % cfg.lagr_iter == 0):
                if cfg.anonymity_vio:
                    lagr_mult = lagr_mult + loss.to('cpu').detach().numpy().copy() * cfg.rho
                else:
                    lagr_mult = lagr_mult + ic_loss.to('cpu').detach().numpy().copy() * cfg.rho
                logger.info(f"[lambda]: {lagr_mult.tolist()}")

        else:
            st_loss = compute_st(r,p,q,zero_one=st_zero_one)
            ic_loss = torch.sum(compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation))

            total_loss = st_loss * (cfg.lambd) + ic_loss * (1 - cfg.lambd)
            total_loss.backward()

        opt.step()
        scheduler.step()
        t_elapsed = time.time() - tic


        # Validation
        if i% cfg.print_iter == 0 or i == cfg.epochs - 1:
            logger.info("[TRAIN-ITER]: %d, [Time-Elapsed]: %f, [Total-Loss]: %f"%(i, t_elapsed, total_loss.item()))
            logger.info("[Stability-Viol]: %f, [IC-Viol]: %f, [ANON-Viol]: %f"%(st_loss.item(), torch.sum(ic_loss).item(), torch.sum(anon_vio).item()))

        if (i>0) and (i % cfg.save_iter == 0) or i == cfg.epochs - 1:
            torch.save(model, "deep-matching/models/model_tmp.pth")

        if ((i>0) and (i% cfg.val_iter == 0)) or i == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_st_loss = 0.0
                val_ic_loss = 0.0
                val_anon_loss = 0.0
                for j in range(cfg.num_val_batches):
                    P, Q = G.generate_batch(cfg.batch_size)
                    p, q = torch.Tensor(P).to(cfg.device), torch.Tensor(Q).to(cfg.device)
                    r = model(p, q)
                    st_loss = compute_st(r, p, q, zero_one=cfg.st_zero_one)
                    ic_loss = torch.sum(compute_ic_FOSD(model, G, r, p, q, include_truncation = include_truncation))
                    anon_vio = torch.sum(compute_anonimity_violation(model,G,r,p,q))
                    val_st_loss += st_loss.item()
                    val_ic_loss += ic_loss.item()
                    val_anon_loss += anon_vio.item()
                logger.info("\t[VAL-ITER]: %d, [ST-Loss]: %f, [IC-Loss]: %f, [ANON-Loss]: %f"%(i, val_st_loss/cfg.num_val_batches, val_ic_loss/cfg.num_val_batches, val_anon_loss/cfg.num_val_batches))

        i += 1
