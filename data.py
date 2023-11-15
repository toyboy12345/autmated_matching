import itertools
import numpy as np
from numba import jit


#@jit(nopython=True)
def generate_permutation_array(N, num_agents):
    P = np.zeros((N, num_agents))
    for i in range(N): P[i] = np.random.permutation(num_agents)
    return P
    

class Data(object):
    """
    A class for generating data for the matching problem
    """
    
    def __init__(self, cfg): 
        self.num_agents = cfg.num_agents
        self.prob = cfg.prob
        self.corr = cfg.corr
        
        
    def sample_ranking(self, N, prob):
        """ 
        Samples ranked lists
        Arguments
            N: Number of samples
            prob: Probability of truncation       
        Returns:
            Ranked List of shape [N, Num_agents]
        """
                              
        N_trunc = int(N * prob)
        P = generate_permutation_array(N, self.num_agents) + 1
               
        if N_trunc > 0:
            
            # Choose indices to truncate
            idx = np.random.choice(N, N_trunc, replace = False)
            
            # Choose a position to truncate
            trunc = np.random.randint(self.num_agents, size = N_trunc)
            
            # Normalize so preference to remain single has 0 payoff
            swap_vals = P[idx, trunc]
            P[idx, trunc] = 0
            P[idx] = P[idx] - swap_vals[:, np.newaxis]
        
        return P/self.num_agents
    
    def generate_all_ranking(self, include_truncation = True):
        """ 
        Generates all possible rankings 
        Arguments
            include_truncation: Whether to include truncations or only generate complete rankings
        Returns:
            Ranked of list of shape: [m, num_agents]
                where m = N! if complete, (N+1)! if truncations are included
        """
                  
        if include_truncation is False:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents)))) + 1.0
        else:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents + 1))))
            M = (M - M[:, -1:])[:, :-1]
            
        return M/self.num_agents
    
        
    def generate_batch(self, batch_size, prob = None, corr = None):
        """
        Samples a batch of data from training
        Arguments
            batch_size: number of samples
            prob: probability of truncation
        Returns
            P: Men's preferences, 
                P_{ij}: How much Man-i prefers to be Women-j
            Q: Women's preferences,
                Q_{ij}: How much Woman-j prefers to be with Man-i
        """

        if corr is None: corr = self.corr
        if prob is None: prob = self.prob
        
        N = batch_size * self.num_agents
        
        P = self.sample_ranking(N, prob)
        Q = self.sample_ranking(N, prob) 
        
        P = P.reshape(-1, self.num_agents, self.num_agents)                           
        Q = Q.reshape(-1, self.num_agents, self.num_agents)
                
        if corr > 0.00:
            P_common = self.sample_ranking(batch_size, prob).reshape(batch_size, 1, self.num_agents)
            Q_common = self.sample_ranking(batch_size, prob).reshape(batch_size, 1, self.num_agents)
        
            P_idx = np.random.binomial(1, corr, [batch_size, self.num_agents, 1])
            Q_idx = np.random.binomial(1, corr, [batch_size, self.num_agents, 1])
        
            P = P * (1 - P_idx) + P_common * P_idx
            Q = Q * (1 - Q_idx) + Q_common * Q_idx
            
        Q = Q.transpose(0, 2, 1)
                
        return P, Q

    def compose_misreport(self, P, Q, M, agent_idx, is_P = True):
        """ Composes mis-report
        Arguments:
            P: Men's preference, [Batch_size, num_agents, num_agents]
            Q: Women's preference [Batch_size, num_agents, num_agents]
            M: Ranked List of mis_reports
                    either [num_misreports, num_agents]
                    or [batch_size, num_misreports, num_agents]                    
            agent_idx: Agent-idx that is mis-reporting
            is_P: if True, Men[agent-idx] misreporting 
                    else, Women[agent-idx] misreporting
                    
        Returns:
            P_mis, Q_mis: [batch-size, num_misreports, num_agents, num_agents]
            
        """
        
        num_misreports = M.shape[-2]
        P_mis = np.tile(P[:, None, :, :], [1, num_misreports, 1, 1])
        Q_mis = np.tile(Q[:, None, :, :], [1, num_misreports, 1, 1])
        
        if is_P: P_mis[:, :, agent_idx, :] = M
        else: Q_mis[:, :, :, agent_idx] = M
        
        return P_mis, Q_mis
    
    
    def generate_all_misreports(self, P, Q, agent_idx, is_P, include_truncation = True):
        """ Generates all mis-reports
        Arguments:
            P: Men's preference, [Batch_size, num_agents, num_agents]
            Q: Women's preference [Batch_size, num_agents, num_agents]
            agent_idx: Agent-idx that is mis-reporting
            is_P: if True, Men[agent-idx] misreporting 
                    else, Women[agent-idx] misreporting
            include_truncation: Whether to truncate preference or submit complete preferences
                    
        Returns:
            P_mis, Q_mis: [batch-size, M, num_agents, num_agents]
                where M = (num_agents + 1)! if truncations are includes
                      M = (num_agents)! if preferences are complete 
        """
        
        M = self.generate_all_ranking(include_truncation = include_truncation)
        P_mis, Q_mis = self.compose_misreport(P, Q, M, agent_idx, is_P)
        
        return P_mis, Q_mis
    
    def sample_misreports(self, P, Q, num_misreports_per_sample, agent_idx, is_P, prob = None):
        """ Samples misreports
        Arguments:
            P: Men's preference, [Batch_size, num_agents, num_agents]
            Q: Women's preference [Batch_size, num_agents, num_agents]
            num_misreports_per_sample: Number of misreports per sample
            agent_idx: Agent-idx that is mis-reporting            
            is_P: if True, Men[agent-idx] misreporting 
                    else, Women[agent-idx] misreporting
            prob: probability of truncation
                    
        Returns:
            P_mis, Q_mis: [batch-size, num_misreports_per_sample, num_agents, num_agents]
        """
        
        if prob is None: prob = self.cfg.prob
                
        N = P.shape[0]
        M = self.sample_ranking(N * num_misreports_per_sample, prob).reshape(N, num_misreports_per_sample, -1)
        P_mis, Q_mis = self.compose_misreport(P, Q, M, agent_idx, is_P)
        
        return P_mis, Q_mis