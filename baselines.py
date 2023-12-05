import itertools
import numpy as np
from numba import jit
import torch


@jit(nopython=True)
def numba_DA(P, Q, menPreferences, womenPreferences):
    
    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):
        
        # Start with no married men
        unmarriedMen = list(range(num_agents))
        
        # No Spouse Yet
        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
        
        # Ptr to index of top choice
        nextManchoice = [0] * num_agents
        
        while unmarriedMen:
            
            he = unmarriedMen[0] 
            
            # He is out of choices, single    
            if nextManchoice[he] == num_agents: 
                manSpouse[he] = num_agents
                unmarriedMen.pop(0)
                continue
                
            she = menPreferences[inst, he, nextManchoice[he]]
            
            # He prefers being single than his top choice: Stay single
            if P[inst, he, she] < 0:
                manSpouse[he] = num_agents
                unmarriedMen.pop(0)
                continue
                            
            # Top Choice is not Married
            if womanSpouse[she] == -1:  
                # She prefers being married rather than being single, so she accepts
                if Q[inst, he, she] > 0: 
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    unmarriedMen.pop(0)
            else:
                # She prefers this man over her current husband, break-up, accept proposal  
                currentHusband = womanSpouse[she]
                if Q[inst, he, she] > Q[inst, currentHusband, she]:
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    R[inst, currentHusband, she] = 0                    
                    unmarriedMen[0] = currentHusband

            nextManchoice[he] = nextManchoice[he] + 1
            
            
    return R

@jit(nopython=True)
def numba_SD(P, Q, menPreferences, womenPreferences, order):    
    num_instances, num_agents = P.shape[0], P.shape[1]  
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):
    
        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
                
        for agent in order:
            
            # MAN
            if agent < num_agents:
                he = agent
                
                # Already taken: skip
                if not manSpouse[he] == -1: continue
                
                # Iterate over his top choices
                for she in menPreferences[inst, he]:
                    
                    # Current Top Choice less preferred than being single
                    if P[inst, he, she] < 0: break
                        
                    # His top-choice is not already taken, then marry
                    if womanSpouse[she] == -1:
                        manSpouse[he], womanSpouse[she] = she, he
                        R[inst, he, she] = 1
                        break
                        
                # If no assignments worked out, he is single
                if manSpouse[he] == -1: manSpouse[he] = num_agents
                    
            # WOMAN
            else:
                she = agent - num_agents
                
                # Already taken: skip
                if not womanSpouse[she] == -1: continue
                    
                # Iterate over her top choices
                for he in womenPreferences[inst, :, she]:
                   
                    # Current Top Choice less preferred than being single
                    if Q[inst, he, she] < 0: break
                        
                    # Her top-choice is not already taken, then marry
                    if manSpouse[he] == -1:
                        manSpouse[he], womanSpouse[she] = she, he
                        R[inst, he, she] = 1
                        break
                    
                # If no assignments worked out, she is single
                if womanSpouse[she] == -1: womanSpouse[she] = num_agents
                
    return R


@jit(nopython=True)
def numba_RSD(P, Q, menPreferences, womenPreferences, orders):    
    num_instances, num_agents = P.shape[0], P.shape[1]  
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):
           
        for order in orders:
            
            manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
        
            for agent in order:

                # MAN
                if agent < num_agents:
                    he = agent

                    # Already taken: skip
                    if not manSpouse[he] == -1: continue

                    # Iterate over his top choices
                    for she in menPreferences[inst, he]:

                        # Current Top Choice less preferred than being single
                        if P[inst, he, she] < 0: break

                        # His top-choice is not already taken, then marry
                        if womanSpouse[she] == -1:
                            manSpouse[he], womanSpouse[she] = she, he
                            R[inst, he, she] += 1
                            break

                    # If no assignments worked out, he is single
                    if manSpouse[he] == -1: manSpouse[he] = num_agents

                # WOMAN
                else:
                    she = agent - num_agents

                    # Already taken: skip
                    if not womanSpouse[she] == -1: continue

                    # Iterate over her top choices
                    for he in womenPreferences[inst, :, she]:

                        # Current Top Choice less preferred than being single
                        if Q[inst, he, she] < 0: break

                        # Her top-choice is not already taken, then marry
                        if manSpouse[he] == -1:
                            manSpouse[he], womanSpouse[she] = she, he
                            R[inst, he, she] += 1
                            break

                    # If no assignments worked out, she is single
                    if womanSpouse[she] == -1: womanSpouse[she] = num_agents
                
    return R/orders.shape[0]


@jit(nopython=True)
def numba_one_RSD(P, Q, menPreferences, womenPreferences, orders):    
    num_instances, num_agents = P.shape[0], P.shape[1]  
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):           
        for order in orders:
            womanSpouse = [-1] * num_agents
            for he in order:
                
                # Iterate over his top choices
                for she in menPreferences[inst, he]:
                    
                    # Current Top Choice less preferred than being single
                    if P[inst, he, she] < 0: break
                    
                    # His top-choice is not already taken, then marry
                    if womanSpouse[she] == -1:
                        womanSpouse[she] = he
                        R[inst, he, she] += 1
                        break
                        
    return R/orders.shape[0]

@jit(nopython=True)
def numba_TTC(P, Q, menPreferences, womenPreferences): 
         
    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):        
        matched = [0] * (2 * num_agents)
        for rnd in range(2 * num_agents): 
            # Create graph              
            G = np.arange(2 * num_agents)    
            for agent in range(2*num_agents):
                if matched[agent] == 1: 
                    G[agent] = -1
                    continue
                                
                # Men Processing
                if agent < num_agents:                   
                    he = agent
                    # Iterate through his preferences
                    for she in menPreferences[inst, he]:
                        
                        # If top available choice is unacceptable, self-point
                        if P[inst, he, she] < 0: break

                        # If top choice is is acceptable and unmatched, point
                        if matched[num_agents + she] == 0:
                            G[he] = num_agents + she
                            break
                            
                else:                   
                    she = agent - num_agents
                    # Iterate through her preferences
                    for he in womenPreferences[inst, :, she]:
                        # If top available choice is unacceptable, self-point
                        if Q[inst, he, she] < 0:  break

                        # If top choice is is acceptable and unmatched, point
                        if matched[he] == 0:
                            G[num_agents + she] = he
                            break
                        
            #print(G)
            # Pick the first unmatched man.   
            curr = -1
            for agent in range(num_agents):
                if matched[agent] == 0:
                    curr = agent
                    break
            
            # If every man is matched, exit    
            if curr == -1: break

            # Find head of a cycle
            visited = [0] * (2 * num_agents)  
            while not visited[curr] == 1:
                visited[curr] = 1
                curr = G[curr]

            # If it's a self point, match and continue with next round
            if curr == G[curr]: 
                matched[curr] = 1
                continue
            
            # Make sure to start with a manc//man-proposing
            if curr >= num_agents: 
                curr = G[curr]

            visited = [0] * (2 * num_agents)  
            # Do matching and exit
            while not visited[curr] == 1:
                R[inst, curr, G[curr] - num_agents] = 1
                matched[curr], matched[G[curr]] = 1, 1
                visited[curr], visited[G[curr]] = 1, 1
                curr = G[G[curr]]
                
    return R


# Worker Proposing
# Serial Dictatorship
def compute_SD_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    order = np.arange(2 * P.shape[1])
    return numba_SD(P, Q, menPreferences, womenPreferences, order)

# RSD
def compute_RSD_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(2 * P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders)

# Worker-Proposing RSD
def compute_one_RSD_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders)

# Worker-Proposing TTC
def compute_TTC_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_TTC(P, Q, menPreferences, womenPreferences)

# Worker-Proposing DA
def compute_DA_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_DA(P, Q, menPreferences, womenPreferences)


# Firm Proposing
# Serial Dictatorship
def compute_SD_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    order = np.arange(2 * P.shape[1])
    return numba_SD(P, Q, menPreferences, womenPreferences, order).transpose(0, 2, 1)

# RSD
def compute_RSD_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(2 * P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders).transpose(0, 2, 1)

# Worker-Proposing RSD
def compute_one_RSD_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders).transpose(0, 2, 1)

# Worker-Proposing TTC
def compute_TTC_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_TTC(P, Q, menPreferences, womenPreferences).transpose(0, 2, 1)

# Worker-Proposing DA
def compute_DA_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_DA(P, Q, menPreferences, womenPreferences).transpose(0, 2, 1)

def compute_algo1(p,q):
    n = len(p)
    r = torch.zeros(n,n)
    for w in range(n):
        for f in range(n):
            if (p[w,f]>0) and (q[w,f]>0):
                nf = len([i for i in range(n) if p[i,f]>0])
                mw = len([i for i in range(n) if q[w,i]>0])

                r[w,f] = 1/max(nf,mw)
    return r
    
def compute_algo1_batch(P,Q):
    r = None
    n = len(P[0])
    for (p,q) in zip(P,Q):
        if r == None:
            r = compute_algo1(p,q)
        else:
            r = torch.vstack((r,compute_algo1(p,q)))
    return torch.reshape(r,(-1,n,n))
    
