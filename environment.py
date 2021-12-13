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

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, keepdims=True, axis=-1)

# x is a list of zeros and ones.
def b2d(x): return sum([2**j for j, xx in enumerate(x) if bool(xx)])

def metropolis_weights_matrix(adjacency):
    adj = np.array(adjacency)
    degree = np.sum(adj, axis=1) - 1
    consensus = np.zeros_like(adjacency, dtype=float) 
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[0]):
            if adjacency[i, j] > 0:
                consensus[i, j] = 1 / (1 + max(degree[i], degree[j]))
                consensus[j, i] = consensus[i, j] # symmetrical
        consensus[i, i] = 1 - (consensus[i, :].sum())
                
    return consensus

class Environment(object):
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_agents=20,
                 n_shared=10,
                 n_private=5):

        # connectivity_ratio = 2 * n_edges / (n_nodes)*(n_nodes - 1) 
        # n_nodes = n_agents
        # default: 4 / n_agents <--> n_edges = 2 * (n_nodes - 1)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.n_shared = n_shared
        self.n_private = n_private
        self.n_edges = n_agents - 1 

        # Transitions & Shared set of features phi(s,a).
        self.transitions = {}
        self.shared = {}

        n_action_space = np.power(n_actions, n_agents)
        probabilities = []
        phis = [] 
        avg_rewards = []
        for _ in range(n_action_space):
            u = np.random.rand(n_states, n_states) + 1e-5 # ensure ergodicity
            p = softmax(u)
            probabilities.append(p)

            shared = uniform(size=(n_states, n_shared))
            phis.append(shared)

            # 2. Each agent has an individual average reward.
            avg_rewards.append(uniform(low=0, high=4, size=(n_states, n_agents)))

        # [n_states, n_actions, n_state]
        self.transitions = np.stack(probabilities, axis=1)
        # [n_states, n_actions, n_shared]
        self.shared = np.stack(phis, axis=1)
        #[n_states, n_actions, n_agents]
        self.average_rewards = np.stack(avg_rewards, axis=1)
        # 4. Private set of features q(s, a_i)
        #[n_states, n_actions, n_agents, n_private]
        self.private = uniform(size=(n_states, n_actions, n_agents, n_private))

        action_schema = np.arange(n_actions)

        # 5. Builds a list with every possible edge. 
        self.edge_list = [(i, j) for i in range(n_agents - 1) for j in range(i + 1, n_agents)]
        self.reset()

    def reset(self):
        self.state = np.random.choice(self.n_states)
        self.n_step = 0

    def get_features(self, actions=None):
        if actions is not None:
            return self.get_shared(actions), self.get_private()
        return self.get_private()

    def get_shared(self, actions):
        # [n_states, n_actions, n_shared]
        return self.shared[self.state, b2d(list(actions)), :]

    def get_private(self):
        ii  = np.arange(self.n_agents)
        return self.private[self.state, :, ii, :]

    def get_rewards(self, actions):
        #[n_states, n_actions, n_agents]
        r = self.average_rewards[self.state, b2d(list(actions)), :]
        u = uniform(low=-0.5, high=0.5, size=self.n_agents)
        return r + u

    def next_step(self, actions):
        # [n_states, n_actions, n_state]
        kk, jj = self.state, b2d(list(actions))
        probabilities = self.transitions[kk, jj, :]
        self.state = \
            np.random.choice(np.arange(self.n_states), p=probabilities)
        self.n_step += 1

    @property
    def adjacency(self):
        return self._adjacency(self.n_step)

    @lru_cache(maxsize=1)
    def _adjacency(self, n_step):
        # random edge selection
        edge_ids = np.random.choice(len(self.edge_list), size=self.n_edges, replace=False)
        edges = [edge for k, edge in enumerate(self.edge_list) if k in edge_ids]

        # make simetrical
        edges += [(edge[-1], edge[0]) for edge in edges]

        # include main diagonal
        edges += [(k, k) for k in range(self.n_agents)]

        # unique
        edges = sorted(sorted(set(edges), key=itemgetter(1)), key=itemgetter(0))
        data = np.ones(len(edges))
        adj = csr_matrix((data, zip(*edges)), dtype=int).todense()
        return adj
        
    def get_consensus(self):
        adj = self.adjacency
        mwe = metropolis_weights_matrix(adj)

        return mwe
        

    def loop(self, n_steps):
        self.reset()        
        first = True
        for step in tqdm(range(n_steps)):
            done = (step == (n_steps -1))
            if first:
                actions = yield self.get_features()
                first = False
            else:
                actions = yield self.get_features(actions), self.get_rewards(actions), done, self.get_consensus()
            self.next_step(actions)
        return 0 

if __name__ == '__main__':

    np.random.seed(42)
    n_states=10
    n_actions=2
    n_agents=5
    n_shared=5
    n_private=3

    env = Environment(
            n_states=n_states,
            n_actions=n_actions,
            n_agents=n_agents,
            n_shared=n_shared,
            n_private=n_private
    )

    print(f'Graph-{env.n_step}:')
    print(f'{env.adjacency}')
    # for actions [0, 0] and state 2 --> a probability array
    agents = tuple([i for i in range(n_agents)])
    actions = np.zeros(n_agents, dtype=np.int32)
    state = env.state
    n_action_schema = 2 ** n_agents
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


