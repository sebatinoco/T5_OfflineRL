import numpy as np

def test_agent(env, agent, nb_episodes=30):

    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    for episode in range(nb_episodes):

        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:
                
            action = agent.select_action(ob_t)
            
            ob_t1, reward, done, _ = env.step(action)

            ob_t = ob_t1
            episode_reward += reward
            
            nb_steps += 1

            if done:
                if nb_steps == 200:
                    success_rate += 1.
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
    
    return np.average(ep_rewards)
