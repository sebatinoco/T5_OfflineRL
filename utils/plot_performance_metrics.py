import matplotlib.pyplot as plt

def plot_performance_metrics(tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec, std_success_vec, exp_id, n_points = 100):
    
    # set n_points
    plot_steps = len(tr_steps_vec) // n_points if len(tr_steps_vec) // n_points > 1 else 1
    
    # keep just n_points
    tr_steps_vec = tr_steps_vec[::plot_steps]
    avg_reward_vec = avg_reward_vec[::plot_steps]
    std_reward_vec = std_reward_vec[::plot_steps]
    success_rate_vec = success_rate_vec[::plot_steps]
    std_success_vec = std_success_vec[::plot_steps]
    
    # plot
    fig, ax = plt.subplots(1, 2, figsize = (15, 4))

    # left chart
    ax[0].errorbar(tr_steps_vec, avg_reward_vec, yerr = std_reward_vec, marker = '.', color = 'C0')
    ax[0].set(xlabel = 'Training Iteration', ylabel = 'Avg Reward')
    ax[0].grid('on')

    # right chart
    #ax[1].errorbar(tr_steps_vec, success_rate_vec, yerr = std_success_vec, marker = '.', color = 'C1')
    ax[1].errorbar(tr_steps_vec, success_rate_vec, marker = '.', color = 'C1')
    ax[1].set(xlabel = 'Training Iteration', ylabel = 'Avg Success Rate')
    ax[1].grid('on')
    
    # save figure
    plt.savefig(f'figures/agg_experiments/{exp_id}.pdf')
    plt.close()