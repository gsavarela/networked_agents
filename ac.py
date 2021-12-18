''' Centralized (version 1) of the actor-critic algorithm

    References:
    -----------
    `Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents.`

    Zhang, et al. 2018
'''
from operator import itemgetter
from functools import lru_cache
from copy import deepcopy
from operator import itemgetter
from collections import defaultdict
from pathlib import Path
import dill


from environment import Environment
import numpy as np
np.seterr(all='raise')

# Uncoment to run stand alone script.
import sys
sys.path.append(Path.cwd().as_posix())

# Helpful one-liners
def softmax(x):
    e_x = np.exp(np.clip(x - np.max(x), a_min=-20, a_max=0))
    return e_x / e_x.sum()
def replace(x, pos, elem): x[pos] = elem; return x


class ActorCritic(object):

    def __init__(self, env):

        # Network
        # todo eliminate dependency.
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_nodes = env.n_nodes
        self.n_shared = env.n_shared
        self.n_private = env.n_private
        self.seed = env.seed
        assert env.n_actions == 2

        # Parameters
        self.mu = np.zeros(1)
        self.next_mu = np.zeros(1)

        self.w = np.ones(self.n_shared) * (1 / self.n_shared)
        self.theta = np.ones((self.n_nodes, self.n_private)) * (1 / self.n_private)
        self.reset()

    @property 
    def alpha(self): return self._alpha(self.n_steps) 

    @lru_cache(maxsize=1)
    def _alpha(self, n_steps): return np.power((n_steps + 1), -0.65)

    @property
    def beta(self): return self._beta(self.n_steps)

    @lru_cache(maxsize=1)
    def _beta(self, n_steps): return np.power((n_steps + 1), -0.85)

    def reset(self):
        np.random.seed(self.seed)
        self.n_steps = 0

    def act(self, private):
        '''Pick actions'''
        choices = np.zeros(self.n_nodes, dtype=np.int32)
        action_set = np.arange(self.n_actions)
        for i in range(self.n_nodes):
            probs = self.policy(private, i)
            choices[i] = np.random.choice(action_set, p=probs)
        return choices

    def update_mu(self, rewards):
        '''Tracks long-term average reward'''
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(rewards)

    def update(self, state, actions, reward, next_state, next_actions):
        # Common knowledge at timestep-t
        shared, private = state
        next_shared, next_private = next_state

        dq = self.grad_q(shared)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        q_values = [] # output
        # 3. Loop through agents' decisions.
        weights = []
        # 3.1 Compute time-difference delta
        delta = np.mean(reward) - mu + \
                self.q(next_shared) - self.q(shared)

        # 3.2 Critic step
        # [n_shared,]
        self.w += alpha * (delta * dq)

        # 3.3 Actor step
        for i in range(self.n_nodes):
            ksi = self.grad_policy(private, actions, i)     # [n_shared]
            self.theta[i, :] += (beta * delta * ksi) # [n_shared]

        q_values.append(self.q(shared))

        self.n_steps += 1
        self.mu = self.next_mu

        return q_values

    def q(self, shared):
        '''Q-function'''
        return self.w @ shared

    def grad_q(self, shared):
        '''Gradient of the Q-function '''
        return shared

    def policy(self, private, i):
        '''pi(s, a_i)'''
        # gibbs distribution / Boltzman policies.
        # [n_private, n_actions]
        # [n_private] @ [n_private, n_actions] --> [n_actions]
        x = self.theta[i, :] @ private[i, :].T
        # [n_actions]
        x = softmax(x) 
        return x

    def grad_policy(self, private, actions, i):
        # [n_actions]
        probabilities = self.policy(private, i)
        ai = actions[i]
        return private[i, ai, :] - np.sum(probabilities @ private[i, :])


if __name__ == '__main__':
    n_states=3
    n_actions=2
    n_nodes=3
    n_shared=4
    n_private=2

    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_nodes=n_nodes,
        n_shared=n_shared,
        n_private=n_private
    )
    ac = ActorCritic(env)

    first_private = env.get_features()
    actions = ac.act(first_private)
    state = env.get_features(actions)
    shared, private = state
    np.testing.assert_almost_equal(private, first_private)
    n_steps = 5
    
    for n_step in range(n_steps):
        print(f'action_value-function {ac.q(shared)}')
        grad_q = ac.grad_q(shared)
        print(f'grad_q {grad_q}')
        np.testing.assert_almost_equal(grad_q, shared)
        pis = [ac.policy(private, i) for i in range(n_nodes)]
        print(f'policies {pis}')
        next_actions = ac.act(private)
        print(f'next_actions {next_actions}')
        env.next_step(actions)
        env.get_features(actions)

        actions = next_actions
        next_state = state

    env.next_step(actions)

    ksis = [ac.grad_policy(private, actions, i) for i in range(n_nodes)]
