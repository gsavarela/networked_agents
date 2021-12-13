import json
from collections import defaultdict
import numpy as np

from pathlib import Path
from tqdm import trange

from environment import Environment
from dist_ac import DistributedActorCritic


def train(n_steps, n_episodes, seed):
    # TODO: Make parse_args
    n_states = 20
    n_actions = 2
    n_agents = 20
    n_shared = 10
    n_private = 5

    np.random.seed(seed)
    # Instanciate
    env = Environment(
        n_states=n_states,
        n_actions=n_actions,
        n_agents=n_agents,
        n_shared=n_shared,
        n_private=n_private
    )
    dac = DistributedActorCritic(env)

    globally_averaged_return = []
    agents_q_values = []
    log = []
    for episode in trange(n_episodes, position=0):

        gen = env.loop(n_steps)
        team_rewards = []
        first = True
        data = defaultdict(list)
        try:
            while True:
                if first:
                    private = next(gen)
                    actions = dac.act(private) 
                    shared = env.get_shared(actions)
                    state = (shared, private)
                    first = False


                next_state, next_rewards, done, consensus = gen.send(actions)
                dac.update_mu(next_rewards)

                next_actions = dac.act(next_state[-1])
                q_values = dac.update(state, actions, next_rewards, next_state, next_actions, consensus)
                globally_averaged_return.append(np.mean(dac.mu))
                agents_q_values.append(q_values)
                state, actions = next_state, next_actions
        except StopIteration as e:
            log.append(data)
            dac.reset()

    return globally_averaged_return, agents_q_values


if __name__ == '__main__':
    train(1000, 10, 0)

