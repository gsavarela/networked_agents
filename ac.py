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
    e_x = np.exp(x)
    return e_x / e_x.sum()

class ActorCritic(object):

    def __init__(self, env):

        # Network
        # todo eliminate dependency.
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_nodes = env.n_nodes
        self.n_phi = env.n_phi
        self.n_varphi = env.n_varphi
        self.seed = env.seed
        assert env.n_actions == 2

        # Parameters
        self.mu = np.zeros(1)
        self.next_mu = np.zeros(1)

        self.w = np.ones(self.n_phi) * (1 / self.n_phi)
        self.theta = np.ones((self.n_nodes, self.n_varphi)) * (1 / self.n_varphi)
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

    def act(self, varphi):
        '''Pick actions

        Parameters:
        -----------
        * varphi: [n_actions, n_nodes, n_varphi]
            Critic features

        Returns:
        --------
        * actions: [n_nodes,]
            Boolean array with agents actions for time t.
        '''
        choices = []
        for i in range(self.n_nodes):
            probs = self.policy(varphi, i)
            choices.append(int(np.random.choice(self.n_actions, p=probs)))
        return choices

    def update_mu(self, rewards):
        '''Tracks long-term mean reward

        Parameters:
        -----------
        * rewards: float<n_nodes> 
            instantaneous rewards.
        '''
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * np.mean(rewards)

    def update(self, state, actions, reward, next_state, next_actions):
        '''Updates actor and critic parameters

        Parameters:
        -----------
        * state: tuple<np.array<n_phi>, np.array<n_actions, n_nodes, n_varphi>>
            features representing the state where 
            state[0]: phi represents the state at time t as seen by the critic.
            state[1]: varphi represents the state at time t as seen by the actor.
        * actions: np.array<n_nodes>
            Actions for each agent at time t.
        * rewards: np.array<n_nodes> 
            Instantaneous rewards for each of the agents.
        * next_state: tuple<np.array<n_phi>, np.array<n_actions, n_nodes, n_varphi>>
            features representing the state where 
            next_state[0]: phi represents the state at time t+1 as seen by the critic.
            next_state[1]: varphi represents the state at time t+1 as seen by the actor.
        * next_actions: tuple(float<>, float<>)
            Actions for each agent at time t+1.
        '''
        # Common knowledge at timestep-t
        phi, varphi = state
        next_phi, next_private = next_state

        dq = self.grad_q(phi)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu

        # Log variables.
        advantages = []
        deltas = []
        grad_ws = []
        grad_thetas = []
        scores = []

        # Capture before updates.
        ws = [self.w.tolist()]
        thetas = [self.theta.tolist()]

        delta = np.mean(reward) - mu + \
                self.q(next_phi) - self.q(phi)

        # 3.2 Critic step
        # [n_phi,]
        grad_w = alpha * delta * dq 
        self.w += grad_w

        # 3.3 Actor step
        for i in range(self.n_nodes):
            ksi = self.grad_log_policy(varphi, actions, i)     # [n_phi]
            grad_theta = (beta * delta * ksi)
            self.theta[i, :] += grad_theta   # [n_phi]

            grad_thetas.append(grad_theta.tolist())
            scores.append(ksi.tolist())

        # Log.
        ws.append(self.w.tolist())
        grad_ws.append(grad_w.tolist())
        thetas.append(self.theta.tolist())

        self.n_steps += 1
        self.mu = self.next_mu

        return advantages, float(delta), ws, grad_ws, thetas, grad_thetas, scores

    def q(self, phi):
        '''Q-function 

        Parameters:
        -----------
        * phi: np.array<n_phi>
            critic features

        Returns:
        --------
        * q: float
           Q-value 
        '''
        return self.w @ phi

    def grad_q(self, phi):
        '''Gradient of the Q-function 

        Parameters:
        ----------
        * phi: np.array<n_phi>
            Critic features

        Returns:
        -------
        * gradient_q: np.array<n_phi>
        '''
        return phi

    def policy(self, varphi, i):
        '''Computes gibbs distribution / Boltzman policies

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_nodes, n_varphi>
            actor features

        * i: integer
            index of the node on the interval {0,N-1}

        Returns:
        -------
        * probabilities: np.array<n_actions>
            Stochastic policy
        '''
        # [n_varphi, n_actions]
        # [n_varphi] @ [n_varphi, n_actions] --> [n_actions]
        x = self.theta[i, :] @ varphi[:, i, :].T
        # [n_actions]
        x = softmax(x) 
        return x

    def grad_log_policy(self, varphi, actions, i):
        '''Computes gibbs distribution / Boltzman policies

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_nodes, n_varphi>
            actor features

        * actions: np.array<n_nodes>
            actions for agents

        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        -------
        * grad_log_policy: np.array<n_varphi>
            Score for policy of agent i.
        '''
        # [n_actions]
        probabilities = self.policy(varphi, i)
        return varphi[actions[i], i, :] - probabilities @ varphi[:, i, :]

    def get_q(self, phi):
        '''Advantage agent i and time t

        Parameters:
        -----------
        * phi: np.array<n_phi>
            Critic features

        Returns:
        -------
        * q: np.array<n_nodes>
            Q value
        '''
        return self.q(phi)

    def get_pi(self, varphi):
        '''Computes the global policy

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_nodes, n_varphi>
            Actor features

        Returns:
        --------
        * global policy: list<list<n_actions>>
            List of policies for each agent.
        '''
        return [self.policy(varphi, i).tolist() for i in range(self.n_nodes)]

if __name__ == '__main__':
    n_states=3
    n_actions=2
    n_nodes=3
    n_phi=4
    n_varphi=2

    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_nodes=n_nodes,
        n_phi=n_phi,
        n_varphi=n_varphi
    )
    ac = ActorCritic(env)

    first_private = env.get_features()
    actions = ac.act(first_private)
    state = env.get_features(actions)
    phi, varphi = state
    np.testing.assert_almost_equal(varphi, first_private)
    n_steps = 5
    
    for n_step in range(n_steps):
        print(f'action_value-function {ac.q(phi)}')
        grad_q = ac.grad_q(phi)
        print(f'grad_q {grad_q}')
        np.testing.assert_almost_equal(grad_q, phi)
        pis = [ac.policy(varphi, i) for i in range(n_nodes)]
        print(f'policies {pis}')
        next_actions = ac.act(varphi)
        print(f'next_actions {next_actions}')
        env.next_step(actions)
        env.get_features(actions)

        actions = next_actions
        next_state = state

    env.next_step(actions)

    ksis = [ac.grad_log_policy(varphi, actions, i) for i in range(n_nodes)]
