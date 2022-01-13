from copy import deepcopy
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import trange

# from environment import Environment
from environment import SemiDeterministicEnvironment as Environment
from dist_ac import DistributedActorCritic
from ac import ActorCritic

UPDATE_CENTRALIZED_KEYS = ['features', 'actions', 'rewards', 'next_features', 'next_actions']
UPDATE_DISTRIBUTED_KEYS =  UPDATE_CENTRALIZED_KEYS + ['consensus']
LOG_KEYS = ['w', 'grad_w', 'theta', 'grad_theta', 'scores']  

def get_agent(distributed=True):
    return DistributedActorCritic if distributed else ActorCritic
    
def get_label(distributed):
    return 'distributed' if distributed else 'centralized'

def dictfy_tr(tr, distributed):
    keys = UPDATE_DISTRIBUTED_KEYS if distributed else UPDATE_CENTRALIZED_KEYS
    tr1 = listfy(tr) 
    return dict(zip(keys, tr1))

def dictfy_log(log): return dict(zip(LOG_KEYS, log))

# turn contents -- possibly numpy arrays into lists.
def listfy(tr):
    return [rec(list(elem), 0) if isinstance(elem, Iterable) else elem for elem in tr]

def rec(root, ind):
    # base case out-of-bounds `ind`.
    if ind == len(root): return root

    # special case: tuples do not support item assignment
    # format and verify cases
    if isinstance(root[ind], tuple):
        root[ind] = list(root[ind])

    if isinstance(root[ind], np.ndarray): 
        # convert a numpy array into a list.
        root[ind] = root[ind].tolist()  
    elif isinstance(root[ind], Iterable):
        root[ind] = rec(root[ind], 0)
    # do nothing and move forward.
    return rec(root, ind + 1)

def get_joint_policy(agent, env):
    '''Computes the joint_policy of agents jointly voting 1.'''
    # State vs agents.
    joint_policies = [] 
    for i in range(env.n_states):
        policy = agent.get_pi(env.get_varphi(i))
        joint_policies.append([votes[-1] for votes in policy])
    return joint_policies 


def train(n_steps, n_episodes, seed):
    # TODO: Make parse_args
    # n_states = 20
    # n_actions = 2
    # n_nodes = 20
    # n_phi = 10  # critic features
    # n_varphi = 5 # actor's features

    # Mini problem
    n_states = 20
    n_actions = 2
    n_nodes = 2
    n_phi = 10
    n_varphi = 5
    variable_graph = True
    seed = 0

    # Instanciate environment
    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_nodes=n_nodes,
        n_phi=n_phi,
        n_varphi=n_varphi,
        seed=seed
    )

    np.random.seed(seed)
    consensus = env.get_consensus()
    phi_00 = env.get_phi(0, [0] * n_nodes) # arbitrary
    varphi_2 = env.get_varphi(2) # arbitrary

    print('Best action for every state')
    print(np.arange(n_states))
    print(env.best_actions)
    print(env.max_team_reward)

    results = {}
    for distributed in (False, True):

        # system of agents
        agent = get_agent(distributed)(env)
        globally_averaged_return = []
        agents_q_values = []
        agents_mus = []
        agents_advantages = []
        agents_deltas = []
        agents_transitions = []
        agents_logs = []

        for episode in trange(n_episodes, position=0):

            gen = env.loop(n_steps)
            first = True
            try:
                while True:
                    if first:
                        state, varphi = next(gen)
                        actions = agent.act(varphi) 
                        phi = env.get_phi(state, actions)
                        features = (phi, varphi)
                        first = False

                    next_state, next_rewards, done = gen.send(actions)
                    agent.update_mu(next_rewards)

                    next_actions = agent.act(env.get_varphi(next_state))
                    tr = [state, actions, next_rewards, next_state, next_actions]

                    if distributed:
                        if variable_graph:
                            tr.append(env.get_consensus())
                        else:
                            tr.append(consensus)
                
                    advantages, deltas, *log = agent.update(*tr)

                    agents_transitions.append(dictfy_tr(tr, distributed))
                    agents_logs.append(dictfy_log(log))

                    globally_averaged_return.append(np.mean(agent.mu))
                    agents_mus.append(agent.mu.tolist())
                    agents_q_values.append(agent.get_q(phi_00))
                    agents_advantages.append(advantages)
                    agents_deltas.append(deltas)
                    state, actions = next_state, next_actions

            except StopIteration as e:
                agent.reset()
                key = get_label(distributed)
                results[key] = {
                    'A': agents_advantages,
                    'J': globally_averaged_return,
                    'Q': agents_q_values,
                    'delta': agents_deltas,
                    'mu': agents_mus,
                    'pi': agent.get_pi(varphi_2),
                    'data': deepcopy(env.log),
                    'transitions': agents_transitions,
                    'logs': agents_logs,
                    'joint_policy': get_joint_policy(agent, env)
                }
    return results



if __name__ == '__main__':
    train(1000, 1, 0)

