'''Random generated-MDP

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.` ---
    Zhang, et al. 2018
    `Policy Evaluation with Temporal Differences: A survey and comparison` --- 
    Dann, et al. 2014
'''
from operator import itemgetter
from functools import lru_cache, cached_property
from collections import defaultdict

import numpy as np
from numpy.random import uniform
from scipy.sparse import csr_matrix
from tqdm import tqdm

# from consensus import metropolis_weights_matrix
from consensus import laplacian_weights_matrix
from consensus import adjacency_matrix

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, keepdims=True, axis=-1)

# x is a list of zeros and ones.
def b2d(x): return sum([2**j for j, xx in enumerate(x) if bool(xx)])


class Environment(object):
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_nodes=20,
                 n_phi=10,
                 n_varphi=5,
                 seed=0):

        # connectivity_ratio = 2 * n_edges / (n_nodes)*(n_nodes - 1) 
        # n_nodes = n_nodes
        # default: 4 / n_nodes <--> n_edges = 2 * (n_nodes - 1)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.n_phi = n_phi
        self.n_varphi = n_varphi
        self.n_edges = 2 * (n_nodes - 1)
        self.seed = seed

        # Transitions & Shared set of features phi(s,a).
        n_action_space = np.power(n_actions, n_nodes)
        probabilities = []
        phis = [] 
        average_rewards = []
        
        np.random.seed(seed)
        for _ in range(n_action_space):
            u = uniform(size=(n_states, n_states)) + 1e-5 # ensure ergodicity
            p = softmax(u)
            probabilities.append(p)

            phi = uniform(size=(n_states, n_phi))
            phis.append(phi)

            # 2. Each agent has an individual average reward.
            average_rewards.append(uniform(low=0, high=4, size=(n_states, n_nodes)).astype(np.float))

        # MDP dynamics [n_states, n_actions, n_state]
        self.P = np.stack(probabilities, axis=1)
        # PHI actor features [n_states, n_actions, n_phi]
        self.PHI = np.stack(phis, axis=1)
        # Average rewards: [n_states, n_actions, n_nodes]
        self.R = np.stack(average_rewards, axis=1)
        # 4. Private set of features q(s, a_i)
        #[n_states, n_actions, n_nodes, n_varphi]
        self.VARPHI = uniform(size=(n_states, n_actions, n_nodes, n_varphi))

        action_schema = np.arange(n_actions)

        # 5. Builds a list with every possible edge. 
        self.edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]

        self.log = defaultdict(list)
        self.log['best_actions'] = self.best_actions.tolist()
        self.log['best_actions_rewards'] = self.max_team_reward.tolist()
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
        # [n_states, n_actions, n_phi]
        if state is not None: return self.PHI[state, b2d(actions), :]
        return self.PHI[self.state, b2d(actions), :]

    def get_varphi(self, state=None):
        #[n_states, n_actions, n_nodes, n_varphi]
        if state is not None: return self.VARPHI[state, ...]
        return self.VARPHI[self.state, ...]

    def get_rewards(self, actions):
        #[n_states, n_actions, n_nodes]
        r = self.R[self.state, b2d(actions), :]
        u = uniform(low=-0.5, high=0.5, size=self.n_nodes)
        return r + u

    # use this for debugging purpose
    @cached_property
    def best_actions(self):
        return np.argmax(np.average(self.R, axis=2), axis=1)

    # use this for debugging purpose
    @cached_property
    def max_team_reward(self):
        avg = np.average(self.R, axis=2)
        ba = self.best_actions
        max_team_reward = []
        for i in range(self.n_states):
            max_team_reward.append(float(np.round(avg[i, ba[i]], 2)))
        return np.array(max_team_reward)

    def next_step(self, actions):
        # [n_states, n_actions, n_state]
        kk, jj = self.state, b2d(actions)
        probabilities = self.P[kk, jj, :]
        self.state = \
            np.random.choice(self.n_states, p=probabilities)
        self.n_step += 1

    @property
    def adjacency(self):
        return self._adjacency(self.n_step)

    @lru_cache(maxsize=1)
    def _adjacency(self, n_step):
        return adjacency_matrix(self.n_nodes, self.n_edges)

    def get_consensus(self):
        adj = self.adjacency
        lwe = laplacian_weights_matrix(adj)
        return lwe
        

    def loop(self, n_steps):
        self.reset()        
        first = True
        for step in tqdm(range(n_steps)):
            done = (step == (n_steps -1))
            if first:
                r = 0
                actions = yield self.get_features()
                first = False
            else:
                r = self.get_rewards(actions)
                actions = yield self.get_features(actions), r, done


            self.log['state'].append(self.state)
            self.log['reward'].append(float(np.mean(r)))
            self.log['actions'].append(b2d(actions))
            self.log['steps'].append(self.n_step)
            self.next_step(actions)
        return 0 
if __name__ == '__main__':

    np.random.seed(42)
    n_states=10
    n_actions=2
    n_nodes=5
    n_phi=5
    n_varphi=3

    env = Environment(
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
    assert env.P.shape == (n_states, n_action_schema, n_states)

    print(f'{actions} -> {b2d(list(actions))}')


    print('average reward')
    print(env.R[agents, state, actions])
    print('get_rewards(actions)')
    print(env.get_rewards(actions))
    env.next_step(actions)
    print(f'next_state {env.state}')
    phi, varphi = env.get_features(actions)
    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # [n_states, n_actions, n_phi]
    np.testing.assert_almost_equal(phi, env.PHI[env.state, 0, :])
    print(f'features:{varphi.shape}')


