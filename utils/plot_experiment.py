import os
import pandas as pd
from utils.plot_performance_metrics import plot_performance_metrics

def plot_experiment(config):

    '''
    Arrange data from an experiment and then plot it.
    '''

    # list all .csv
    files = os.listdir('metrics')

    # filter csv files
    exp_files = [file for file in files if config in file]

    # csv union
    exp_data = pd.DataFrame()
    for file in exp_files:
        file_data = pd.read_csv(f'metrics/{file}', sep = '\t')
        exp_data = pd.concat((exp_data, file_data), axis = 1)

    # obtain aggregate metrics
    steps = exp_data['steps'].mean(axis = 1)
    avg_reward = exp_data['avg_reward'].mean(axis = 1)
    std_reward = exp_data['avg_reward'].std(axis = 1)
    success_rate = exp_data['success_rate'].mean(axis = 1)
    std_success = exp_data['success_rate'].std(axis = 1)

    # plot
    plot_performance_metrics(steps, avg_reward, std_reward, success_rate, std_success, config)