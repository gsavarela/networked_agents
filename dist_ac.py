''' Distributed version of the actor-critic algorithm

    Loop version.

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

class DistributedActorCritic(object):

    def __init__(self, env):

        # Network
        # todo eliminate dependency.
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_agents = env.n_nodes
        self.n_phi = env.n_phi
        self.n_varphi = env.n_varphi
        self.seed = env.seed
        assert env.n_actions == 2

        # Parameters
        np.random.seed(self.seed)
        self.mu = np.zeros(self.n_agents)
        self.next_mu = np.zeros(self.n_agents)

        self.w = np.ones((self.n_agents, self.n_phi)) * (1/ self.n_phi)
        self.theta = np.ones((self.n_agents, self.n_varphi)) * (1/ self.n_varphi)
        self.log = defaultdict(list)
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
        * varphi: [n_actions, n_agents, n_varphi]
            Critic features

        Returns:
        --------
        * actions: [n_agents,]
            Boolean array with agents actions for time t.
        '''
        choices = []
        for i in range(self.n_agents):
            probs = self.policy(varphi, i)
            choices.append(int(np.random.choice(self.n_actions, p=probs)))
        return choices

    def update_mu(self, rewards):
        '''Tracks long-term mean reward

        Parameters:
        -----------
        * rewards: float<n_agents> 
            instantaneous rewards.
        '''
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * rewards

    def update(self, state, actions, rewards, next_state, next_actions, C):
        '''Updates actor and critic parameters

        Parameters:
        -----------
        * state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where 
            state[0]: phi represents the state at time t as seen by the critic.
            state[1]: varphi represents the state at time t as seen by the actor.
        * actions: np.array<n_agents>
            Actions for each agent at time t.
        * rewards: np.array<n_agents> 
            Instantaneous rewards for each of the agents.
        * next_state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where 
            next_state[0]: phi represents the state at time t+1 as seen by the critic.
            next_state[1]: varphi represents the state at time t+1 as seen by the actor.
        * next_actions: tuple(float<>, float<>)
            Actions for each agent at time t+1.
        '''
        # 1. Common knowledge at timestep-t
        phi, varphi = self.env.get_features(state, actions)
        next_phi, _ = self.env.get_features(next_state, next_actions)

        dq = self.grad_q(phi)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        advantages = []
        deltas = []
        grad_ws = []
        grad_thetas = []
        scores = []

        ws = [self.w.tolist()]
        thetas = [self.theta.tolist()]
        wtilde = np.zeros_like(self.w)
        # 2. Iterate agents on the network.
        for i in range(self.n_agents):
            # 2.1 Compute time-difference delta
            delta = rewards[i] - mu[i] + \
                    self.q(next_phi, i) - self.q(phi, i)

            # 2.2 Critic step
            grad_w = alpha * delta * dq 
            wtilde[i, :] = self.w[i, :] + grad_w # [n_phi,]

            # 3.3 Actor step
            adv = self.advantage(phi, varphi, state, actions, i)  # [n_varphi,]
            ksi = self.grad_log_policy(varphi, actions, i)     # [n_varphi,]
            grad_theta = (beta * adv * ksi)
            self.theta[i, :] += grad_theta # [n_varphi,]

            # Log step
            grad_ws.append(grad_w.tolist())
            grad_thetas.append(grad_theta.tolist())
            scores.append(ksi.tolist())

            advantages.append(adv)
            deltas.append(float(delta))
        # Consensus step.
        self.w = C @ wtilde

        # Log.
        ws.append(self.w.tolist())
        thetas.append(self.theta.tolist())

        self.n_steps += 1
        self.mu = self.next_mu
        return advantages, deltas, ws, grad_ws, thetas, grad_thetas, scores

    def q(self, phi, i):
        '''Q-function 

        Parameters:
        -----------
        * phi: np.array<n_phi>
            critic features

        Returns:
        --------
        * q: float
            q-value for agent i
        '''
        return self.w[i, :] @ phi

    def v(self, varphi, state, actions, i):
        '''Relative value-function

        A version of value-function where the effects of i-agent's
        have been averaged. 

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features
        * actions: np.array<n_agents>
            actions for agents
        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        --------
        * v: float
            value-function with averaged i
        '''
        probabilities = self.policy(varphi, i)

        ret = 0
        for j, aj in enumerate(range(self.n_actions)):
            _actions = [aj if k == i else ak for k, ak in enumerate(actions)] 
            phi_aj = self.env.get_phi(state, np.array(_actions))
            ret += probabilities[j] * self.q(phi_aj, i)
        return ret

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
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features


        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        -------
        * probabilities: np.array<n_actions>
            Stochastic policy
        '''
        # [n_varphi, n_actions]
        # [n_varphi] @ [n_varphi, n_actions] --> [n_actions]
        x = self.theta[i, :] @ varphi[:, i, :].T
        # [n_actions]
        z = softmax(x) 
        
        return z

    def grad_log_policy(self, varphi, actions, i):
        '''Computes gibbs distribution / Boltzman policies

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features

        * actions: np.array<n_agents>
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

    def advantage(self, phi, varphi, state, actions, i):
        '''Advantage agent i and time t

        The advantage for agent-i evaluates the `goodness` of 
        taking action a_i.


        Parameters:
        -----------
        * phi: np.array<n_phi>
            Critic features

        * varphi: np.array<n_actions, n_agents, n_varphi>
            Actor features

        * actions: np.array<n_agents>
            actions for agents

        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        -------
        * advantege f
            Score for policy of agent i.
        '''
        return self.q(phi, i) - self.v(varphi, state, actions, i)

    def get_q(self, phi):
        '''Q-function for each agent-i

        Parameters:
        -----------
        * phi: np.array<n_phi>
            Critic features

        Returns:
        -------
        * q: np.array<n_agents>
            Q value for each agent in the network.
        '''
        return [self.q(phi,  i) for i in range(self.n_agents)]

    def get_pi(self, varphi):
        '''The policy agent i at time-t and state given by varphi.

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            Actor features

        Returns:
        --------
        * global policy: list<list<n_actions>>
            List of policies for each agent.
        '''

        return [self.policy(varphi, i).tolist() for i in range(self.n_agents)]


if __name__ == '__main__':
    n_states=3
    n_actions=2
    n_agents=3
    n_phi=4
    n_varphi=2

    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_agents=n_agents,
        n_phi=n_phi,
        n_varphi=n_varphi
    )
    # consensus 
    consensus = build_consensus_matrix(env.adjacency, method='metropolis')
    print(consensus)
    # consensus should be a doubly stochastic matrix
    eigen_vector = np.ones((n_agents, 1), dtype=int)

    # left eigen_vector should equal itself.
    np.testing.assert_almost_equal(eigen_vector.T @ consensus, eigen_vector.T)

    # right eigen_vector should equal itself.
    np.testing.assert_almost_equal(consensus @ eigen_vector, eigen_vector)

    dac = DistributedActorCritic(env)

    first_private = env.get_features()
    actions = dac.act(first_private)
    print(actions)
    phi, varphi = env.get_features(actions)
    np.testing.assert_almost_equal(varphi, first_private)


    qs = [dac.q(phi,  i) for i in range(n_agents)]
    print(f'action_value-function {qs}')
    grad_q = dac.grad_q(phi)
    print(f'grad_q {grad_q}')
    np.testing.assert_almost_equal(grad_q, phi)
    pis = [dac.policy(varphi, i) for i in range(n_agents)]
    print(f'policies {pis}')
    vs = [dac.v(varphi, actions, i) for i in range(n_agents)]
    print(f'value-function {vs}')
    # np.testing.assert_almost_equal(np.stack(qs), np.stack(vs))
    advs = [dac.advantage(phi, varphi, actions, i) for i in range(n_agents)]
    print(f'advantages {advs}')
    next_actions = dac.act(varphi)
    print(f'next_actions {next_actions}')

    ksis = [dac.grad_log_policy(varphi, actions, i) for i in range(n_agents)]
    # print(f'GradientLogPolicy:\t{ksis}')

    # params = dac.get_actor()
    # print(f'Actor:\t{params}')

    # params = dac.get_critic()
    # print(f'Critic:\t{params}')

    # params = dac.get_probabilities(state)
    # print(f'Probabilities:\t{params}')
    # params = dac.get_values(state, actions)
    # print(f'Values:\t{params}')
    # params = dac.get_qs(state, actions)
    # print(f'Qs:\t{params}')
    # params = dac.get_advantages(state, actions)
    # print(f'Advantages:\t{params}')
    # N = int(10e5)
    # for i in range(N):
    #     dac.update(state, actions, rewards, next_states)
