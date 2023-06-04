import gym
import yaml
import time
import os
import torch

from replay_buffer import ReplayBuffer
from conservative_q_learning import ConservativeDeepQNetworkAgent
from utils.run_args import run_args
from train_agent import train_agent

if __name__ == '__main__':

    # load run args
    r_args = run_args() 
    
    filter_env = r_args['env']
    filter_config = r_args['filter_config']
    n_trials = r_args['n_trials']
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    # list configs
    configs = sorted(os.listdir('configs'))

    # filter configs if specified
    if filter_env or filter_config:
        
        env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] # filter by environment
        filtered_configs = [config for config in configs if config in filter_config] # filter by config
        
        final_configs = set(env_configs + filtered_configs) # filter configs
        configs = [config for config in configs if config in final_configs] # apply filter

    print('Running experiments on the following configs: ', configs)

    for trial in range(1, n_trials + 1):
        start_time = time.time()
        # for every config file
        for config in configs:
            # load config
            with open(f"configs/{config}", 'r') as file:
                args = yaml.safe_load(file)

            # experiment name
            exp_name = f"{args['env'][:-3]}_{args['exp_id']}_{trial}"

            env_name = args['env']
            env = gym.make(env_name)
            
            if env_name == 'CartPole-v0':
                from experts.pid_controller import PIDController
                
                # load PID params
                with open(f"experts/params/pid_parameters.yaml", 'r') as file:
                    pid_params = yaml.safe_load(file)
                
                expert_agent = PIDController(**pid_params, dt = 0.02)
            
            elif env_name == 'MountainCar-v0':
                from experts.mcar_policy import MCarExpertPolicy
                expert_agent = MCarExpertPolicy()

            else: 
                raise NotImplementedError()

            nb_states = env.observation_space.shape[0]
            nb_actions = env.action_space.n
            
            agent = ConservativeDeepQNetworkAgent(dim_states=nb_states, dim_actions=nb_actions, lr=0.01, gamma=0.99, alpha=args['alpha'], device = device)
            replay_buffer = ReplayBuffer(dim_states=nb_states, dim_actions=nb_actions, max_size=100000, sample_size=128)

            train_agent(env=env, agent=agent, expert_agent=expert_agent, 
                        replay_buffer=replay_buffer, rollout_episodes=args['nb_rollouts'], training_iters=20000, exp_name = exp_name)