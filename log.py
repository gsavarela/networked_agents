
def logger(state, actions, rewards, dac, env, td, data):
    # transitions.
    data['rewards'].append(rewards.tolist())
    data['actions'].append(actions.tolist())
    data['actions2'].append(env.to_index(actions))
    data['shared'].append(state[0].tolist())
    data['private'].append(state[-1].tolist())
    data['hidden'].append(int(env.state))

    # Hyperparameters.
    data['alpha'].append(dac.alpha)
    data['beta'].append(dac.beta)

    # Objective variables.
    data['mu'].append(dac.mu.tolist())
    data['w'].append(dac.w.tolist())
    data['theta'].append(dac.theta.tolist())

    # n_step
    data['n_step'].append(dac.n_steps)

    # Time-difference variables.
    data['grad_Q'].append(td['grad_Q'])
    data['Qt'].append(td['Qt'])
    data['Vt'].append(td['Vt'])
    data['Qt+1'].append(td['Qt+1'])
    data['delta'].append(td['delta'])
    data['w_local'].append(td['w_local'])
    data['adv'].append(td['adv'])
    data['ksi'].append(td['ksi'])
    data['next_mu'].append(td['next_mu'])
