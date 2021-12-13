import sys
import json
from pathlib import Path
from datetime import datetime
import multiprocessing
from multiprocessing.pool import Pool

from train import train
from plots import globally_averaged_plot, q_values_plot

def fn(args): return train(*args)
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

    globally_averaged_return, q_values = zip(*results)
    with (results_path / 'globally_averaged_return.json').open('w') as f:
        json.dump(globally_averaged_return, f)
    globally_averaged_plot(globally_averaged_return, results_path)
    with (results_path / 'q_values.json').open('w') as f:
        json.dump(q_values, f)
    q_values_plot(q_values, results_path)
    return results, str(results_path)

if __name__ == '__main__':
    results, results_path = main(5, 1, 15000, 1)

