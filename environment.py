'''Random generated-MDP

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.` ---
    Zhang, et al. 2018
    `Policy Evaluation with Temporal Differences: A survey and comparison` --- 
    Dann, et al. 2014
'''
import numpy as np
from itertools import product as prod
from tqdm import tqdm

def norm(x): return x / sum(x)
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, keepdims=True, axis=-1)

class Environment(object):
    def __init__(self,
                 n_states=20,
                 n_actions=2,
                 n_agents=20,
                 n_shared=10,
                 n_private=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.n_shared = n_shared
        self.n_private = n_private

        # Transitions & Shared set of features phi(s,a).
        self.transitions = {}
        self.shared = {}

        n_action_space = np.power(n_actions, n_agents)
        probabilities = []
        phis = [] 
        avg_rewards = []
        for _ in range(n_action_space):
            u = np.random.rand(n_states, n_states)
            p = softmax(u) + 1e-8 # ensure ergodicity
            probabilities.append(p)

            shared = np.random.uniform(size=(n_states, n_shared))
            phis.append(shared)

            # 2. Each agent has an individual average reward.
            _size = (n_states, n_agents)
            avg_rewards.append(np.random.uniform(low=0, high=4, size= _size))

        # [n_states, n_actions, n_state]
        self.transitions = np.stack(probabilities, axis=1)
        # [n_states, n_actions, n_shared]
        self.shared = np.stack(phis, axis=1)
        #[n_states, n_actions, n_agents]
        self.average_rewards = np.stack(avg_rewards, axis=1)

        # 4. Private set of features q(s, a_i)
        #[n_states, n_actions, n_agents, n_private]
        _size = (n_states, n_actions, n_agents, n_private)
        self.private = np.random.uniform(size=_size)

        # 5. Builds a randomly generated graph.
        self.adjacency = np.eye(n_agents)
        action_schema = np.arange(n_actions)
        for i in range(n_agents - 1):
            for j in range(i + 1, n_agents):
                edge = np.random.choice(action_schema, p=(0.3, 0.7))
                self.adjacency[i, j] = edge
                self.adjacency[j, i] = edge
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
        act = self.to_index(actions.tolist())
        ret = self.shared[self.state, act, :]
        return ret

    def get_private(self):
        ii  = np.arange(self.n_agents)
        return self.private[self.state, :, ii, :]

    def get_rewards(self, actions):
        #[n_states, n_actions, n_agents]
        jj = self.to_index(actions.tolist())
        kk, ii  = self.state, np.arange(self.n_agents)
        r = self.average_rewards[kk, jj, ii]
        u = np.random.uniform(low=-0.5, high=0.5, size=self.n_agents)
        return r + u

    def next_step(self, actions):
        # [n_states, n_actions, n_state]
        kk, jj = self.state, self.to_index(actions.tolist())
        probabilities = norm(self.transitions[kk, jj, :])
        self.state = \
            np.random.choice(np.arange(self.n_states), p=probabilities)
        self.n_step += 1

    def to_index(self, actions):
        return sum([2**j for j, act in enumerate(actions) if act])

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
    n_agents=3
    n_shared=5
    n_private=3

    env = Environment(
            n_states=n_states,
            n_actions=n_actions,
            n_agents=n_agents,
            n_shared=n_shared,
            n_private=n_private
    )

    print(f'Graph:')
    print(f'{env.adjacency}')
    # for actions [0, 0] and state 2 --> a probability array
    agents = tuple([i for i in range(n_agents)])
    actions = np.zeros(n_agents, dtype=np.int32)
    state = env.state
    n_action_schema = 2 ** n_agents
    print(f'current state: {state}')
    print(env.transitions.shape) 
    assert env.transitions.shape == (n_states, n_action_schema, n_states)

    print(f'{actions} -> {env.to_index(actions)}')


    print('average reward')
    print(env.average_rewards[agents, state, actions])
    print('get_rewards(actions)')
    print(env.get_rewards(actions))
    env.next_step(actions)
    print(f'next_state {env.state}')
    shared, private = env.get_features(actions)
    print(f'shared:{shared.shape}')
    # [n_states, n_actions, n_shared]
    np.testing.assert_almost_equal(shared, env.shared[env.state, 0, :])
    print(f'features:{private.shape}')

