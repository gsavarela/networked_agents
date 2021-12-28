import json
from pathlib import Path

import numpy as np
from numpy.random import choice
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm

FIGURE_X = 6.0
FIGURE_Y = 4.0

STD_CURVE_COLOR = (0.88,0.70,0.678)
MEAN_CURVE_COLOR = (0.89,0.282,0.192)
SMOOTHING_CURVE_COLOR = (0.33,0.33,0.33)
CENTRALIZED_AGENT_COLOR = (0.2, 1.0, 0.2)

def globally_averaged_plot(centralized_J, distributed_J, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')

    globally_averaged_return = np.array(distributed_J)
    n_runs, n_steps = globally_averaged_return.shape


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(globally_averaged_return, axis=0)

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)
    plt.plot(X,Y, label=f'Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')
    plt.plot(X,np.mean(np.array(centralized_J), axis=0), c=CENTRALIZED_AGENT_COLOR, label='Centralized')

    plt.xlabel('Time')
    plt.ylabel('Globally Averaged Return J')
    plt.legend(loc=4)

    file_name = (results_path / 'globally_averaged_return.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'globally_averaged_return.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    

def advantages_plot(decentralized_A, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')
    advantages = np.array(decentralized_A)

    n_runs, n_steps, n_agents = advantages.shape

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(advantages, axis=0) # average number of runs
    labels = [f'agent {_id}' for _id in range(n_agents)]
    plt.plot(X,Y, label=labels)

    lowess = sm.nonparametric.lowess(np.average(Y, axis=-1), X, frac=0.10)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label=f'smoothed')

    plt.xlabel('Timesteps')
    plt.ylabel('Advantages')
    plt.legend(loc='upper right')

    file_name = (results_path / 'advantages.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'advantages.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def delta_plot(centralized_deltas, decentralized_deltas, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')
    deltas = np.array(decentralized_deltas)

    n_runs, n_steps, n_agents = deltas.shape

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(deltas, axis=0) # average number of runs
    T = np.average(np.array(centralized_deltas), axis=0) 
    labels = [f'agent {_id}' for _id in range(n_agents)]

    plt.plot(X,Y, label=labels)

    lowess = sm.nonparametric.lowess(np.average(Y, axis=-1), X, frac=0.10)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label=f'smoothed')
    plt.plot(X, T, c=CENTRALIZED_AGENT_COLOR, label='Centralized')

    plt.xlabel('Timesteps')
    plt.ylabel('Deltas')
    plt.legend(loc='upper right')

    file_name = (results_path / 'deltas.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'deltas.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def mu_plot(centralized_mus, decentralized_mus, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')
    mus = np.array(decentralized_mus)

    n_runs, n_steps, n_agents = mus.shape

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(mus, axis=0) # average number of runs
    T = np.average(np.array(centralized_mus), axis=0) 
    labels = [f'agent {_id}' for _id in range(n_agents)]

    plt.plot(X,Y, label=labels)

    lowess = sm.nonparametric.lowess(np.average(Y, axis=-1), X, frac=0.10)
    plt.plot(X,np.average(Y, axis=-1), c=MEAN_CURVE_COLOR, label=f'Mean')
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label=f'Smoothed')
    plt.plot(X, T, c=CENTRALIZED_AGENT_COLOR, label='Centralized')

    plt.xlabel('Timesteps')
    plt.ylabel('Mus')
    plt.legend(loc='upper right')

    file_name = (results_path / 'mus.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'mus.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def pi_plot(centralized_pis, decentralized_pis, results_path=None):
    # plt.rcParams['text.usetex'] = True 
    if results_path is None:
        results_path = Path('data/results')

    decentralized_pis = np.array(decentralized_pis)
    n_runs, n_agents, n_probs = decentralized_pis.shape

    decentralized_pis = np.average(decentralized_pis, axis=0).reshape(-1)


    centralized_pis = np.array(centralized_pis)
    n_runs, n_agents, n_probs = centralized_pis.shape
    centralized_pis = np.average(centralized_pis, axis=0).reshape(-1)
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    X = np.arange(1, n_agents * n_probs + 1, 1)
    labels = [i if i % 5 == 0 else None for i in range(1, n_agents * n_probs + 1)]

    rects1 = ax.bar(X - width/2, centralized_pis.tolist(), width, label='Centralized')
    rects2 = ax.bar(X + width/2, decentralized_pis.tolist(), width, label='Distributed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel(r'$\displaystyle \text{Probabilities} \pi(s, a^i)')
    ax.set_ylabel('Probabilities pi(s, a^i)')
    ax.set_xlabel('(s, a^i)')
    ax.set_title('s = 2')
    ax.set_xticks(X)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc='upper right')

    fig.tight_layout()
    file_name = (results_path / 'pis.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'pis.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

def q_values_plot(centralized_Q, decentralized_Q, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')
    q_values = np.array(decentralized_Q)

    n_runs, n_steps, n_agents = q_values.shape

    # draw tree agents
    n_choice = min(n_agents, 3)
    agent_ids = sorted(choice(n_agents, size=n_choice, replace=False).tolist())
    
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(q_values, axis=0) # average number of runs
    Y = Y[:, agent_ids]
    labels = [f'agent {agent_id}' for agent_id in agent_ids]

    plt.plot(X,Y, label=labels)


    labels = '-'.join([str(int(_id)) for _id in agent_ids])
    lowess = sm.nonparametric.lowess(np.average(Y, axis=1), X, frac=0.10)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label=f'average {labels}')

    plt.plot(X, np.average(centralized_Q, axis=0), c=CENTRALIZED_AGENT_COLOR, label='Centralized')
    plt.xlabel('Timesteps')
    plt.ylabel('Relative Q-values')
    plt.legend(loc='upper right')

    file_name = (results_path / 'q_values.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'q_values.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
