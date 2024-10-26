import json
import torch
import numpy as np
from itertools import permutations
import os

dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(dir, "pref_anonymity.json")
dic = json.load(open(json_path, "r"))

def pref_to_num(pref):
    prefs = np.array(list(permutations(range(1,4))))
    return np.where((prefs==pref).all(axis=1))[0][0]+1

def num_to_pref(num):
    prefs = np.array(list(permutations(range(1,4))))
    pref = np.array([[prefs[int(num[i])-1] for i in range(3)],[prefs[int(num[i])-1] for i in range(3,6)]])
    return pref

def prefs_to_num(prefs):
    num = ""
    for i in range(prefs.shape[0]):
        for j in range(prefs.shape[1]):
            num += str(pref_to_num(prefs[i,j]))
    return num

def pref_anonymous(p,q):
    device = p.device
    prefs = list(map(lambda x,y: prefs_to_num((torch.cat([x,torch.transpose(y,0,1)]).reshape((-1,3,3))*3).to(int).to('cpu').detach().numpy().copy()), p,q))
    parents = np.array(list(map(lambda x: num_to_pref(dic[x]['parent']), prefs)))
    p_, q_ = torch.Tensor(parents[:,0,:]/3).to(device), torch.Tensor(parents[:,1,:].transpose(0,2,1)/3).to(device)
    return p_, q_

def recover_match(r, steps):
    r_ = r.to('cpu').detach().numpy().copy()
    for step in steps:
        if step[2] == 0:
            r_[step[0],:], r_[step[1],:] = r_[step[1],:], r_[step[0],:].copy()
        else:
            r_[:,step[0]], r_[:,step[1]] = r_[:,step[1]], r_[:,step[0]].copy()
    return r_

def match_anonymity(model, p, q):
    device = p.device
    prefs = list(map(lambda x,y: prefs_to_num((torch.cat([x,torch.transpose(y,0,1)]).reshape((-1,3,3))*3).to(int).to('cpu').detach().numpy().copy()), p,q))
    parents = np.array(list(map(lambda x: num_to_pref(dic[x]['parent']), prefs)))
    p_, q_ = torch.Tensor(parents[:,0,:]/3).to(device), torch.Tensor(parents[:,1,:].transpose(0,2,1)/3).to(device)
    r = model(p_,q_)

    return torch.Tensor(list(map(lambda r,step: recover_match(r,step).tolist(),r,list(map(lambda x: dic[x]['step'], prefs))))).to(device)
