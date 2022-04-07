from Hvac import Hvac
from DDQN import DQN
import numpy as np
import matplotlib.pyplot as plt


def On_off_policy(state, action):
    # 制冷，低于最小温度，功率为0，高于最大功率，功率为1
    T_min = 24 * 1.8 + 32
    T_max = 26 * 1.8 + 32
    indoor_temperature = state[1]
    if indoor_temperature > T_max:
        action = 2
    elif indoor_temperature < T_min:
        action = 2
    return action


def train():
    episode_rewards = []
    for episode in range(5000):
        state = env.reset(True)
        episode_reward = 0
        # print('---------------episode ', episode, '-------------------')
        for step in range(30):
            action = RL.choose_action(state) * 0.1

            next_state, reward, done, info = env.step(action)
            # print('outTemp:', state[0], 'inTemp:', state[1], 'price:', state[2], 'daynum', env.day_number, 'time_step:',
            #       state[3])
            # print('action:', action)
            # print('outTemp:', next_state[0], 'inTemp:', next_state[1], 'price:', next_state[2], 'daynum:',
            #       env.day_number, 'time_step:', next_state[3])
            # print('reward:', reward)
            state = next_state
            episode_reward += reward
            # print('step ', step, 'reward ', reward)
            if done:
                break
        print('episode ', episode, 'episode_reward ', episode_reward)
        episode_rewards.append(episode_reward)
    file = open('./logs/train_reward.txt', 'w')
    file.write(str(episode_rewards))
    file.close()
    plt.plot(episode_rewards, c='b')
    plt.savefig('./logs/training_reward_v1.png')
    plt.title('training_reward')
    RL.save('./logs/model_v1.h5')


# on_off_policy = True 则采用on-off策略
def test(on_off_policy=False):
    reward_list = []
    action = 0
    for episode in range(121):
        episode_reward = 0
        state = env.reset(False, episode)

        for step in range(30):
            if on_off_policy:
                action = On_off_policy(state, action)
            else:
                action = RL.choose_action(state) * 0.1

            next_state, reward, done, info = env.step(action)

            # print('outTemp:', state[0], 'inTemp:', state[1], 'price:', state[2], 'daynum', env.day_number, 'time_step:',
            #       state[3])
            # print('action:', action)
            # print('outTemp:', next_state[0], 'inTemp:', next_state[1], 'price:', next_state[2], 'daynum:',
            #       env.day_number, 'time_step:', next_state[3])
            # print('reward:', reward)

            state = next_state
            episode_reward += reward
            if done:
                break
        print('test episode ', episode, 'episode_reward ', episode_reward)
        reward_list.append(episode_reward)
    return reward_list


if __name__ == "__main__":
    env = Hvac()
    # HVAC的输入功率范围为[0, hvac_p_cap]
    # 以0.1为步长
    n_features = 4
    n_actions = 2 / 0.1 + 1
    RL = DQN(n_actions, n_features)
    train()
    # 比较on-off policy 和 DDQN算法的性能
    baseline_reward_list = test(True)
    DDQN_reward_list = test()

    # 保存到文件
    file = open('./logs/baseline_reward_list_v1.txt', 'w')
    file.write(str(baseline_reward_list))
    file.close()
    file = open('logs/DDQN_reward_list_v1.txt', 'w')
    file.write(str(DDQN_reward_list))
    file.close()

    # 画图
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(baseline_reward_list, c='r')
    ax1.set_title('on-off-policy')
    ax2.plot(DDQN_reward_list, c='g')
    ax2.set_title('DDQN')
    plt.savefig('./logs/test1.png')
