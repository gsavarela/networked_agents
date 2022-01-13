import json
from pathlib import Path

import numpy as np 
import pandas as pd

def flatten(items, ignore_types=(str, bytes)):
    """

    Usage:
    -----
    > items = [1, 2, [3, 4, [5, 6], 7], 8]

    > # Produces 1 2 3 4 5 6 7 8
    > for x in flatten(items):
    >         print(x)

    Ref:
    ----

    David Beazley. `Python Cookbook.'
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

# Creates a pandas-dataframe with transitions.
def transitions_df(results, agent_type='distributed'):
    if not agent_type in ('centralized', 'distributed'):
        raise KeyError('agent_type must be in `centralized` or `distributed`')

    if len(results) > 1:
        print('Processing only the first epoch')

    res = results[0][agent_type]
    n_states = res['data']['n_states']
    n_actions = res['data']['n_actions']
    n_nodes = res['data']['n_nodes']
    n_varphi = res['data']['n_varphi']
    n_phi = res['data']['n_phi']

    first = True
    for tr in res['transitions']:
        for key, val in tr.items():
            import ipdb; ipdb.set_trace()



if __name__ == '__main__':
    dir_path = Path('data/results/20220112193533.025359')
    results_path = dir_path / 'results.json'

    with results_path.open('r') as f: res = json.load(f)
    df_tr = transitions_df(res)

    


    



