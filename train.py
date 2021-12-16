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
    n_nodes = 5
    n_shared = 10
    n_private = 5
    np.random.seed(seed)

    # Instanciate
    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_nodes=n_nodes,
        n_shared=n_shared,
        n_private=n_private
    )

    results = {}
    for distributed in (True, False):

        # system of agents
        agent = get_agent(distributed)(env)

        globally_averaged_return = []
        agents_q_values = []
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
                    if distributed: tr.append(env.get_consensus())
                
                    q_values = agent.update(*tr)
                    globally_averaged_return.append(np.mean(agent.mu))
                    agents_q_values.append(q_values)
                    state, actions = next_state, next_actions

            except StopIteration as e:
                agent.reset()
                key = get_label(distributed)
                results[key] = (globally_averaged_return, agents_q_values)
    return results


if __name__ == '__main__':
    train(1000, 10, 0)

