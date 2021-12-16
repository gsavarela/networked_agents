import sys
import json
from pathlib import Path
from datetime import datetime
import multiprocessing
from multiprocessing.pool import Pool

from train import train
from plots import globally_averaged_plot, q_values_plot

def fn(args): return train(*args)
# helps transform a list of dictionaries into a pair of lists
def gn(adict, pos): return (adict['centralized'][pos], adict['distributed'][pos])
def unwrap(alist, pos): return zip(*[gn(elem, pos) for elem in alist])

def main(n_runs, n_processors, n_steps, n_episodes):


    results_path = Path('data/results')
    results_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    print(f'Experiment timestamp: {timestamp}\n')
    

    base_args = (n_steps, n_episodes)
    train_args = [base_args + (n_run * 10,) for n_run in range(n_runs)]

    if n_processors > 1:
        pool = Pool(n_processors)
        results = pool.map(fn, train_args)
        pool.close()
        pool.join()
    else:
        results = []
        for args in train_args:
            results.append(train(*args))


    results_path = results_path / timestamp
    results_path.mkdir(exist_ok=True)
    sys.stdout.write(str(results_path))

    
    # get globally averaged return
    with (results_path / 'results.json').open('w') as f:
        json.dump(results, f)

    centralized_J, decentralized_J = unwrap(results, 0) 
    globally_averaged_plot(centralized_J, decentralized_J, results_path)

    centralized_Q, decentralized_Q = unwrap(results, 1) 
    q_values_plot(centralized_Q, decentralized_Q, results_path)

    return results, str(results_path)

if __name__ == '__main__':
    results, results_path = main(5, 1, 15000, 1)

