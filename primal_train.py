import os
import sys
import time
import logging
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

from data import Data
from primal_net import PrimalNet
from primal_loss import *

class HParams:
    def __init__(self, num_agents = 3,
                 batch_size = 1024, num_hidden_layers = 4, num_hidden_nodes = 256, lr = 5e-3, epochs = 50000,
                 print_iter = 100, val_iter = 1000, num_val_batches = 200,
                 prob = 0, corr = 0, seed = 0, device = "cuda:0",
                 lambd = torch.ones((3,3)), rho = 1, lagr_iter = 100):
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
        # Correlation of rankings
        self.corr = corr
        # Run seed
        self.seed = seed

        self.device = device

        self.lambd = lambd
        self.rho = rho
        self.lagr_iter = lagr_iter

def train_primal(cfg, G, model, include_truncation = False):
    # # File names
    # root_dir = os.path.join("experiments", "agents_%d"%(cfg.num_agents), "corr_%.2f"%(cfg.corr))
    # log_fname = os.path.join(root_dir, "LOG_%d_lambd_%f_prob_%.2f_corr_%.2f.txt"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
    # model_path = os.path.join(root_dir, "MODEL_%d_lambd_%f_prob_%.2f_corr_%.2f"%(cfg.seed, cfg.lambd, cfg.prob, cfg.corr))
    # os.makedirs(root_dir, exist_ok=True)

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

    lambd = cfg.lambd

    while i < cfg.epochs:

        # Reset opt
        opt.zero_grad()
        model.train()

        # Inference
        P, Q = G.generate_batch(cfg.batch_size)
        p, q = torch.Tensor(P).to(cfg.device), torch.Tensor(Q).to(cfg.device)
        r = model(p, q)

        loss,constr_vio,obj = compute_loss(cfg,model,r,p,q,torch.Tensor(lambd).to(cfg.device),cfg.rho)
        if (i>0) and (i%cfg.lagr_iter == 0):
            lambd += cfg.rho*constr_vio.item()
            print(lambd)
        
        loss.backward(retain_graph=True)

        opt.step()
        scheduler.step()
        t_elapsed = time.time() - tic


        # Validation
        if i% cfg.print_iter == 0 or i == cfg.epochs - 1:
            logger.info("[TRAIN-ITER]: %d, [Time-Elapsed]: %f, [Total-Loss]: %f"%(i, t_elapsed, loss.item()))
            logger.info("[CONSTR-Vio]: %f, [OBJECTIVE]: %f"%(constr_vio.item(),obj.item()))

        if (i>0) and (i % cfg.save_iter == 0) or i == cfg.epochs - 1:
            torch.save(model, "deep-matching/models/primal/model_tmp.pth")

        if ((i>0) and (i% cfg.val_iter == 0)) or i == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_constr_vio = 0
                val_obj = 0
                for j in range(cfg.num_val_batches):
                    P, Q = G.generate_batch(cfg.batch_size)
                    p, q = torch.Tensor(P).to(cfg.device), torch.Tensor(Q).to(cfg.device)
                    r = model(p, q)
                    loss,constr_vio,obj = compute_loss(cfg,model,r,p,q,torch.Tensor(lambd).to(cfg.device),cfg.rho)
                    val_loss += loss.item()
                    val_constr_vio += constr_vio.item()
                    val_obj += obj.item()
                logger.info("\t[VAL-ITER]: %d, [LOSS]: %f, [Constr-vio]: %f, [Objective]: %f"%(i, val_loss/cfg.num_val_batches, val_constr_vio/cfg.num_val_batches, val_obj/num_val_batches))

        i += 1
