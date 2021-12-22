'''Random generated-MDP

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.` ---
    Zhang, et al. 2018
    `Policy Evaluation with Temporal Differences: A survey and comparison` --- 
    Dann, et al. 2014
'''
from operator import itemgetter
from functools import lru_cache

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
def b2d(x):
    return sum([2**j for j, xx in enumerate(x) if bool(xx)])



class Environment(object):
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_nodes=20,
                 n_shared=10,
                 n_private=5,
                 seed=0):

        # connectivity_ratio = 2 * n_edges / (n_nodes)*(n_nodes - 1) 
        # n_nodes = n_nodes
        # default: 4 / n_nodes <--> n_edges = 2 * (n_nodes - 1)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.n_shared = n_shared
        self.n_private = n_private
        self.n_edges = 2 * (n_nodes - 1)
        self.seed = seed

        # Transitions & Shared set of features phi(s,a).
        self.transitions = {}
        self.shared = {}

        n_action_space = np.power(n_actions, n_nodes)
        probabilities = []
        phis = [] 
        avg_rewards = []
        
        np.random.seed(seed)
        for _ in range(n_action_space):
            u = np.random.rand(n_states, n_states) + 1e-5 # ensure ergodicity
            p = softmax(u)
            probabilities.append(p)

            shared = uniform(size=(n_states, n_shared))
            phis.append(shared)

            # 2. Each agent has an individual average reward.
            avg_rewards.append(uniform(low=0, high=4, size=(n_states, n_nodes)))

        # [n_states, n_actions, n_state]
        self.transitions = np.stack(probabilities, axis=1)
        # [n_states, n_actions, n_shared]
        self.shared = np.stack(phis, axis=1)
        #[n_states, n_actions, n_nodes]
        self.average_rewards = np.stack(avg_rewards, axis=1)
        # 4. Private set of features q(s, a_i)
        #[n_states, n_actions, n_nodes, n_private]
        self.private = uniform(size=(n_states, n_actions, n_nodes, n_private))

        action_schema = np.arange(n_actions)

        # 5. Builds a list with every possible edge. 
        self.edge_list = [(i, j) for i in range(n_nodes - 1) for j in range(i + 1, n_nodes)]
        self.reset()

    def reset(self):
        # for the same seed starts from the same.
        np.random.seed(self.seed) 
        self.state = np.random.choice(self.n_states)
        self.n_step = 0

    def get_features(self, actions=None):
        if actions is not None:
            return self.get_shared(actions), self.get_private()
        return self.get_private()

    def get_shared(self, actions, state=None):
        # [n_states, n_actions, n_shared]
        return self.shared[self.state, b2d(list(actions)), :]

    def get_private(self):
        #[n_states, n_actions, n_nodes, n_private]
        ii  = np.arange(self.n_nodes)
        return self.private[self.state, :, ii, :]

    def get_rewards(self, actions):
        #[n_states, n_actions, n_nodes]
        r = self.average_rewards[self.state, b2d(list(actions)), :]
        u = uniform(low=-0.5, high=0.5, size=self.n_nodes)
        return r + u

    def next_step(self, actions):
        # [n_states, n_actions, n_state]
        kk, jj = self.state, b2d(actions.tolist())
        probabilities = self.transitions[kk, jj, :]
        self.state = \
            np.random.choice(np.arange(self.n_states), p=probabilities)
        self.n_step += 1

    @property
    def adjacency(self):
        return self._adjacency(self.n_step)
        # return np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

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
                actions = yield self.get_features()
                first = False
            else:
                actions = yield self.get_features(actions), self.get_rewards(actions), done
            self.next_step(actions)
        return 0 

if __name__ == '__main__':

    np.random.seed(42)
    n_states=10
    n_actions=2
    n_nodes=5
    n_shared=5
    n_private=3

    env = Environment(
            n_states=n_states,
            n_actions=n_actions,
            n_nodes=n_nodes,
            n_shared=n_shared,
            n_private=n_private
    )

    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # for actions [0, 0] and state 2 --> a probability array
    agents = tuple([i for i in range(n_nodes)])
    actions = np.zeros(n_nodes, dtype=np.int32)
    state = env.state
    n_action_schema = 2 ** n_nodes
    print(f'current state: {state}')
    print(env.transitions.shape) 
    assert env.transitions.shape == (n_states, n_action_schema, n_states)

    print(f'{actions} -> {b2d(list(actions))}')


    print('average reward')
    print(env.average_rewards[agents, state, actions])
    print('get_rewards(actions)')
    print(env.get_rewards(actions))
    env.next_step(actions)
    print(f'next_state {env.state}')
    shared, private = env.get_features(actions)
    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # [n_states, n_actions, n_shared]
    np.testing.assert_almost_equal(shared, env.shared[env.state, 0, :])
    print(f'features:{private.shape}')


