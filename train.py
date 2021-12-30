from copy import deepcopy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import trange

from environment import Environment
from dist_ac import DistributedActorCritic
from ac import ActorCritic


def get_agent(distributed=True):
    return DistributedActorCritic if distributed else ActorCritic
    
def get_label(distributed):
    return 'distributed' if distributed else 'centralized'

def train(n_steps, n_episodes, seed):
    # # TODO: Make parse_args
    # n_states = 20
    # n_actions = 2
    # n_nodes = 20
    # n_phi = 10  # critic features
    # n_varphi = 5 # actor's features

    # worse
    n_states = 20
    n_actions = 2
    n_nodes = 2
    n_phi = 10
    n_varphi = 5
    is_time_variable_graph = True

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
    phi_00 = env.get_phi([0] * n_nodes, 0) # arbitrary
    varphi_2 = env.get_varphi(2) # arbitrary

    print('Best action for every state')
    print(np.arange(n_states))
    print(env.best_actions)
    print(env.max_team_reward)

    results = {}
    for distributed in (True, False):
        # system of agents
        agent = get_agent(distributed)(env)

        globally_averaged_return = []
        agents_q_values = []
        agents_mus = []
        agents_advantages = []
        agents_deltas = []

        for episode in trange(n_episodes, position=0):

            gen = env.loop(n_steps)
            first = True
            try:
                while True:
                    if first:
                        varphi = next(gen)
                        actions = agent.act(varphi) 
                        phi = env.get_phi(actions)
                        state = (phi, varphi)
                        first = False

                    next_state, next_rewards, done = gen.send(actions)
                    agent.update_mu(next_rewards)

                    next_actions = agent.act(next_state[-1])
                    tr = [state, actions, next_rewards, next_state, next_actions]
                    if distributed:
                        if is_time_variable_graph:
                            tr.append(env.get_consensus())
                        else:
                            tr.append(consensus)
                
                    advantages, deltas = agent.update(*tr)
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
                    'data': deepcopy(env.log)
                }
    return results



if __name__ == '__main__':
    train(1000, 1, 0)

