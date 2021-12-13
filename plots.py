import json
from pathlib import Path

import numpy as np
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

def globally_averaged_plot(globally_averaged_return, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')

    globally_averaged_return = np.array(globally_averaged_return)
    n_runs, n_steps = globally_averaged_return.shape


    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)


    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(globally_averaged_return, axis=0)
    Y_std = np.std(globally_averaged_return, axis=0)


    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label=f'Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if globally_averaged_return.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Time')
    plt.ylabel('Globally Averaged Return J')
    plt.legend(loc=4)

    file_name = (results_path / 'globally_averaged_return.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'globally_averaged_return.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    
def q_values_plot(q_values, results_path=None):
    
    if results_path is None:
        results_path = Path('data/results')
    q_values = np.array(q_values)

    n_runs, n_steps, n_agents = q_values.shape

    # draw tree agents
    agent_ids = np.random.choice(np.arange(n_agents), size=3, replace=False)
    
    
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.linspace(1, n_steps, n_steps)
    Y = np.average(q_values, axis=0) # average number of runs
    Y = Y[:, agent_ids]
    labels = [f'agent {agent_id}' for agent_id in agent_ids]

    plt.plot(X,Y, label=labels)

    plt.xlabel('Timesteps')
    plt.ylabel('Relative Q-values')


    lowess = sm.nonparametric.lowess(np.average(Y, axis=1), X, frac=0.10)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Agent average')
    plt.legend(loc='upper right')

    file_name = (results_path / 'q_values.pdf').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    file_name = (results_path / 'q_values.png').as_posix()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)