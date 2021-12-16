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
def replace(x, pos, elem): x[pos] = elem; return x


class DistributedActorCritic(object):

    def __init__(self, env):

        # Network
        # todo eliminate dependency.
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_agents = env.n_nodes
        self.n_shared = env.n_shared
        self.n_private = env.n_private
        assert env.n_actions == 2

        # Parameters
        self.mu = np.zeros(self.n_agents)
        self.next_mu = np.zeros(self.n_agents)
        self.w = np.random.randn(self.n_agents, self.n_shared)
        self.theta = np.random.randn(self.n_agents, self.n_private)
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
        self.n_steps = 0

    def act(self, private):
        '''Pick actions

        first to be called on the pipeline.

        Parameters:
        -----------
        * x: [n_agents, n_features * n_actions]
        features

        Returns:
        -----------
        * actions: [n_agents,]
        boolean array
        '''
        choices = np.zeros(self.n_agents, dtype=np.int32)
        action_set = np.arange(self.n_actions)
        for i in range(self.n_agents):
            probs = self.policy(private, i)
            choices[i] = np.random.choice(action_set, p=probs)
        return choices

    def update_mu(self, rewards):
        '''Tracks long-term mean reward

        second to be called on the pipeline.
        '''
        self.next_mu = (1 - self.alpha) * self.mu + self.alpha * rewards

    def update(self, state, actions, reward, next_state, next_actions, C):
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
        for i in range(self.n_agents):
            # 3.1 Compute time-difference delta
            delta = reward[i] - mu[i] + \
                    self.q(next_shared, i) - self.q(shared, i)

            # 3.2 Critic step
            # [n_shared,]
            weights.append(self.w[i, :] + alpha * (delta * dq))

            # 3.3 Actor step
            adv = self.advantage(shared, private, actions, i)
            ksi = self.grad_policy(private, actions, i)     # [n_shared]
            self.theta[i, :] += (beta * adv * ksi) # [n_shared]

            q_values.append(self.q(shared, i))

        # 4. Consensus step: broadcast weights
        # Send weights to the network
        # get weights from neighbors
        weights = np.stack(weights)

        # forms to test
        # self.w = np.einsum('ij, ijk -> jk', self.C, weights)
        # self.w = self.C @ weights
        for i in range(self.n_agents):
            self.w[i, :] = C[i, :] @ weights
        self.n_steps += 1
        self.mu = self.next_mu

        return q_values

    def q(self, shared, i):
        '''Q-function 

        Parameters:
        -----------
        * x: [n_agents, n_features * n_actions]
        features
        * i: int
        indicator for agent

        Returns:
        --------
        * q_value: float
        action-value function for agent i
        '''
        return self.w[i, :] @ shared

    def v(self, private, actions, i):
        def get_phi(x): return self.env.get_shared(x)
        probabilities = self.policy(private, i)
        _actions = deepcopy(actions)
        ret = 0
        for j, aj in enumerate(range(self.n_actions)):
            phi_aj = get_phi(replace(_actions, i, aj))
            ret += probabilities[j] * self.q(phi_aj, i)
        return ret

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

    def advantage(self, shared, private, actions, i):
        return self.q(shared, i) - self.v(private, actions, i)


   #  '''        debugging        '''
   #  def get_actor(self):
   #      return self._get_params([w.tolist() for w in self.ws])

   #  def get_critic(self):
   #      return self._get_params([theta.tolist() for theta in self.thetas])

   #  def get_probabilities(self, state):
   #      probs = []
   #      for i in range(self.n_agents):
   #          probs.append(self.policy(state, i).tolist())
   #      return self._get_params(probs)

   #  def get_values(self, state, actions):
   #      values = []
   #      for i in range(self.n_agents):
   #          values.append(float(self.v(state, actions, i)))

   #      return self._get_params(values)

   #  def get_qs(self, state, actions):
   #      qs = [float(self.q(state, actions, i)) for i in range(self.n_agents)]
   #      return self._get_params(qs)

   #  def get_advantages(self, state, actions):
   #      advs = [float(self.advantage(state, actions, i)) for i in range(self.n_agents)]
   #      return self._get_params(advs)

   #  def _get_params(self, params):
   #      return {tl: par for tl, par in zip(self.tl_ids, params)}

   #  """ Serialization """
   #  # Serializes the object's copy -- sets get_wave to null.
   #  def save_checkpoint(self, chkpt_dir_path, chkpt_num):
   #      class_name = type(self).__name__.lower()
   #      file_path = Path(chkpt_dir_path) / chkpt_num / f'{class_name}.chkpt'  
   #      file_path.parent.mkdir(exist_ok=True)
   #      with open(file_path, mode='wb') as f:
   #          dill.dump(self, f)

   #  # deserializes object -- except for get_wave.
   #  @classmethod
   #  def load_checkpoint(cls, chkpt_dir_path, chkpt_num):
   #      class_name = cls.__name__.lower()
   #      file_path = Path(chkpt_dir_path) / str(chkpt_num) / f'{class_name}.chkpt'  
   #      with file_path.open(mode='rb') as f:
   #          new_instance = dill.load(f)

   #      return new_instance

if __name__ == '__main__':
    n_states=3
    n_actions=2
    n_agents=3
    n_shared=4
    n_private=2

    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_agents=n_agents,
        n_shared=n_shared,
        n_private=n_private
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
    shared, private = env.get_features(actions)
    np.testing.assert_almost_equal(private, first_private)


    qs = [dac.q(shared,  i) for i in range(n_agents)]
    print(f'action_value-function {qs}')
    grad_q = dac.grad_q(shared)
    print(f'grad_q {grad_q}')
    np.testing.assert_almost_equal(grad_q, shared)
    pis = [dac.policy(private, i) for i in range(n_agents)]
    print(f'policies {pis}')
    vs = [dac.v(private, actions, i) for i in range(n_agents)]
    print(f'value-function {vs}')
    # np.testing.assert_almost_equal(np.stack(qs), np.stack(vs))
    advs = [dac.advantage(shared, private, actions, i) for i in range(n_agents)]
    print(f'advantages {advs}')
    next_actions = dac.act(private)
    print(f'next_actions {next_actions}')

    ksis = [dac.grad_policy(private, actions, i) for i in range(n_agents)]
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
