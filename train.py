import json
from collections import defaultdict
import numpy as np

from pathlib import Path
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
    # n_shared = 10
    # n_private = 5

    # worse
    n_states = 2
    n_actions = 2
    n_nodes = 2
    n_shared = 10
    n_private = 5
    is_time_variable_graph = False

    # Instanciate
    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_nodes=n_nodes,
        n_shared=n_shared,
        n_private=n_private,
        seed=seed
    )


    np.random.seed(seed)
    consensus = env.get_consensus()
    sa_00 = env.shared[0, 0, :] # arbitrary
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
                        private = next(gen)
                        actions = agent.act(private) 
                        shared = env.get_shared(actions)
                        state = (shared, private)
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
                    agents_q_values.append(agent.get_q(sa_00))
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
                }
    return results


if __name__ == '__main__':
    train(1000, 10, 0)

