import itertools
import numpy as np
from numba import jit


@jit(nopython=True)
def numba_gs(P, Q, menPreferences, womenPreferences):
    
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
def numba_rsd_ex_ante(P, Q, menPreferences, womenPreferences, orders):    
    num_instances, num_agents = P.shape[0], P.shape[1]  
    R = np.zeros(P.shape)
    
    for inst in range(num_instances):
           
        # Generate a random order
        for order in orders:
            
            manSpouse = [-1] * num_agents   
            womanSpouse = [-1] * num_agents
        
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


def compute_rsd_ex_ante_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(2 * P.shape[1])))))
    return numba_rsd_ex_ante(P, Q, menPreferences, womenPreferences, orders)

def compute_gsm_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_gs(P, Q, menPreferences, womenPreferences)
    
def compute_gsw_batch(P, Q):
    P_ = Q.transpose((0, 2, 1))
    Q_ = P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P_, axis = -1)
    womenPreferences = np.argsort(-Q_, axis = -2)
    return numba_gs(P_, Q_, menPreferences, womenPreferences).transpose((0, 2, 1))


