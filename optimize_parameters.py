import gym
import yaml

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from experts.pid_controller import PIDController
from utils.test_agent import test_agent

if __name__ == '__main__':

    # parameters of optimization
    env = gym.make('CartPole-v0')
    n_trials = 100

    # function to optimize
    def objective(trial):
        params = {
            "kp": trial.suggest_float("kp", -1, 1),
            "ki": trial.suggest_float("ki", -1, 1),
            "kd": trial.suggest_float("kd", -1, 1),
        }
        
        agent = PIDController(**params)
        
        return test_agent(env, agent)
        
    # optimize parameters
    study = optuna.create_study(direction = 'maximize', sampler = TPESampler())
    study.optimize(objective, n_trials = n_trials, show_progress_bar = True)

    # print results
    best_trial = study.best_trial
    print(f'Best params: {best_trial.params}')
    print(f'Best score: {best_trial.value}')

    # export parameters
    with open('experts/params/pid_parameters.yaml', 'w') as outfile:
        yaml.dump(best_trial.params, outfile, default_flow_style=False)