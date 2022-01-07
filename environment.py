'''Random generated-MDP

    References:
    -----------
    * `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.` ---
    Zhang, et al. 2018

    * `Policy Evaluation with Temporal Differences: A survey and comparison` --- 
    Dann, et al. 2014
'''
from operator import itemgetter
from functools import lru_cache, cached_property
from collections import defaultdict

import numpy as np
from numpy.random import uniform
from scipy.sparse import csr_matrix
from tqdm import tqdm

# from consensus import laplacian_weights_matrix
from consensus import metropolis_weights_matrix
from consensus import adjacency_matrix

# x is a list of zeros and ones.
def bin2dec(x): return sum([2**j for j, xx in enumerate(x) if bool(xx)])
# x numeric represention of action.
# y number of nodes.
def dec2str(x, y):
    z = bin(x)[2:]; return z[::-1]+ '0'*(y - len(z))
def sumones(x):
    y = [int(xx == '1') for xx in x];
    return sum(y) if any(y) else 0
# simple majority
def smaj(x): return x / 2 if x % 2 == 0 else (x + 1) / 2

class Environment(object):
    '''Randomly Generated MDP'''
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_nodes=20,
                 n_phi=10,
                 n_varphi=5,
                 seed=0):

        # n_nodes = n_nodes
        # default: 4 / n_nodes <--> n_edges = 2 * (n_nodes - 1)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.n_phi = n_phi
        self.n_varphi = n_varphi
        self.n_edges = 2 * (n_nodes - 1)
        self.n_action_space = n_actions ** n_nodes 
        self.seed = seed

        # Transitions & Shared set of features phi(s,a).
        n_dims = n_states * self.n_action_space
        np.random.seed(seed)

        # MDP dynamics [|n_states||n_actions ** n_nodes|, n_state]
        P = uniform(size=(n_dims, n_states))
        self.P = P / P.sum(axis=-1, keepdims=True)

        # PHI actor features [|n_states||n_actions ** n_nodes|, n_phi]
        self.PHI = uniform(size=(n_dims, n_phi))

        # Average Rewards: [|n_states||n_actions ** n_nodes|, n_nodes]
        self.R = uniform(low=0, high=4, size=(n_dims, n_nodes))

        # Private set of features var_phi(s, a_i)
        # [n_states, n_actions, n_nodes, n_varphi]
        self.VARPHI = uniform(size=(n_states, n_actions, n_nodes, n_varphi))

        # 5. Builds a list with every possible edge. 
        self.edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]

        self.log = defaultdict(list)
        self.reset()

    def reset(self):
        # for the same seed starts from the same.
        np.random.seed(self.seed) 
        self.state = np.random.choice(self.n_states)
        self.n_step = 0
        for key in ('actions', 'steps', 'state', 'reward'):
            if key in self.log: del self.log[key]

    def get_features(self, actions=None):
        if actions is None: return self.get_varphi()
        return self.get_phi(actions), self.get_varphi()

    def get_phi(self, actions, state=None):
        # [|n_states||n_actions ** n_nodes|, n_phi]
        if state is None: state = self.state 
        return self.PHI[self.get_dim(state, actions), :]

    def get_varphi(self, state=None):
        # [n_states, n_actions, n_nodes, n_varphi]
        if state is None: state = self.state
        return self.VARPHI[state, ...]

    def get_rewards(self, actions):
        # [|n_states||n_actions ** n_nodes|, n_nodes]
        r = self.R[self.get_dim(self.state, actions), :] 
        u = uniform(low=-0.5, high=0.5, size=self.n_nodes)
        return r + u 

    def get_dim(self, state, actions):
        return state * self.n_action_space + bin2dec(actions)

    # use this for debugging purpose
    @cached_property
    def best_actions(self):
        r = self.R.reshape((self.n_states, -1, self.n_nodes))
        return np.argmax(np.average(r, axis=2), axis=1)

    # use this for debugging purpose
    @cached_property
    def max_team_reward(self):
        r = self.R.reshape((self.n_states, -1, self.n_nodes))
        r = np.average(r, axis=2)

        ba = self.best_actions
        max_team_reward = []
        for state in range(self.n_states):
            max_team_reward.append(float(np.round(r[state, ba[state]], 2)))
        return np.array(max_team_reward)

    def next_step(self, actions):
        probs = self.P[self.get_dim(self.state, actions), :]
        self.state = np.random.choice(self.n_states, p=probs)
        self.n_step += 1

    @property
    def adjacency(self):
        return self._adjacency(self.n_step)
        return np.ones((self.n_nodes, self.n_nodes))

    @lru_cache(maxsize=1)
    def _adjacency(self, n_step):
        return adjacency_matrix(self.n_nodes, self.n_edges)

    def get_consensus(self):
        adj = self.adjacency
        lwe = metropolis_weights_matrix(adj)
        return lwe
        

    def loop(self, n_steps):
        self.reset()        
        first = True
        for step in tqdm(range(n_steps)):
            done = (step == (n_steps -1))
            if first:
                r = 0
                actions = yield self.get_features()
            else:
                r = self.get_rewards(actions)
                actions = yield self.get_features(actions), r, done

            if first:
                # do this one
                self.log['best_actions'] = self.best_actions.tolist()
                self.log['best_actions_rewards'] = self.max_team_reward.tolist()
                first = False
            self.log['state'].append(self.state)
            self.log['reward'].append(float(np.mean(r)))
            self.log['actions'].append(bin2dec(actions))
            self.log['steps'].append(self.n_step)
            self.next_step(actions)
        return 0 

class SemiDeterministicEnvironment(Environment):
    '''Randomly Generated MDP but easier'''
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_nodes=20,
                 n_phi=10,
                 n_varphi=5,
                 seed=0):

        super(SemiDeterministicEnvironment, self).__init__(
            n_states=n_states, 
            n_actions=n_actions,
            n_nodes=n_nodes,
            n_phi=n_phi,
            n_varphi=n_varphi,
            seed=seed
        )
        # Re-defines the reward.
        n_dims = self.n_states * self.n_action_space
        for n_dim in range(n_dims):
            n_state = n_dim // self.n_action_space
            n_action = n_dim - n_state * self.n_action_space
            if n_state == self.n_states - 1:
                self.R[n_dim, :] = 4
            else:
                # Gives higher rewards for selecting 1 on 'lower' states
                digits = dec2str(n_action, n_nodes) # zeros or ones.
                for i, digit in enumerate(digits): 
                    self.R[n_dim, i] = 4 * (n_state - (1 - int(digit == '1'))) / self.n_states

        # Re-defines the transitions.
        # simple majority threshold
        maj = smaj(self.n_nodes)
        for n_dim in range(n_dims):

            n_state = n_dim // self.n_action_space
            n_action = n_dim - n_state * self.n_action_space
            n_ones = sumones(dec2str(n_action, self.n_nodes))

            for next_state in range(self.n_states):
                if n_state == 0:
                    if next_state == 0:
                        self.P[n_dim, next_state] = max(float(n_ones < maj), 1e-5)
                    elif next_state == 1:
                        self.P[n_dim, next_state] = max(float(n_ones >= maj), 1e-5)
                    else:
                        self.P[n_dim, next_state] = 1e-5
                else:
                    if next_state == n_state - 1:
                        self.P[n_dim, next_state] = max(float(n_ones < maj), 1e-5)
                    elif next_state == n_state:
                        if n_state == self.n_states - 1:
                            self.P[n_dim, next_state] = max(float(n_ones >= maj), 1e-5)
                        else:
                            self.P[n_dim, next_state] = max(float(n_ones == maj), 1e-5)
                    elif next_state == n_state + 1:
                        self.P[n_dim, next_state] = max(float(n_ones > maj), 1e-5)
                    else:
                        self.P[n_dim, next_state] = 1e-5
        self.P = self.P / self.P.sum(axis=-1, keepdims=True)
        self.log = defaultdict(list)

    @property
    def adjacency(self):
        return np.ones((self.n_nodes, self.n_nodes))

    def get_rewards(self, actions):
        # [|n_states||n_actions ** n_nodes|, n_nodes]
        r = self.R[self.get_dim(self.state, actions), :] 
        # u = uniform(low=-0.5, high=0.5, size=self.n_nodes)
        return r

if __name__ == '__main__':

    np.random.seed(42)
    n_states = 3
    n_actions = 2
    n_nodes = 2
    n_phi = 10
    n_varphi = 5
    env = SemiDeterministicEnvironment(
            n_states=n_states,
            n_actions=n_actions,
            n_nodes=n_nodes,
            n_phi=n_phi,
            n_varphi=n_varphi
    )

    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # for actions [0, 0] and state 2 --> a probability array
    agents = tuple([i for i in range(n_nodes)])
    actions = np.zeros(n_nodes, dtype=np.int32)
    state = env.state
    n_action_schema = 2 ** n_nodes

    print(f'current state: {state}')
    print(env.P.shape) 
    print(env.P)
    assert env.P.shape == (n_states * n_action_schema, n_states)

    print(f'{actions} -> {bin2dec(list(actions))}')


    dim = env.get_dim(env.state, actions)
    print('average reward')
    print(env.R[dim, :])
    print('get_rewards(actions)')
    print(env.get_rewards(actions))
    env.next_step(actions)
    print(f'next_state {env.state}')
    phi, varphi = env.get_features(actions)
    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # [n_states, n_actions, n_phi]
    np.testing.assert_almost_equal(phi, env.PHI[env.state * env.n_action_space + bin2dec(actions), :])
    print(f'features:{varphi.shape}')
    env.state = n_states - 1
    actions = np.ones(n_nodes, dtype=np.int32)
    print(env.get_dim(env.state, actions), env.PHI.shape[0])
    assert env.get_dim(env.state, actions) + 1, env.PHI.shape[0]

