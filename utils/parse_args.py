import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument('--env_name', type = str, default = 'CartPole-v0', help = 'environment to run the experiment') 
    parser.add_argument('--nb_rollouts', type = int, default = 10, help = 'number of rollouts')
    parser.add_argument('--alpha', type = float, default = 0.0, help = 'alpha of the experiment')
    parser.add_argument('--exp_name', type = str, default = 'experiment', help = 'name of the experiment')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args