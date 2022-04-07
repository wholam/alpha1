import numpy as np
import pandas as pd


class Hvac:
    """
    state = [outdoor temperature, indoor temperature, price, time_step]
    """

    def __init__(self):
        self.day_number = None
        self.time_step = None
        self.current_state = None
        self.df_outTemp = pd.read_csv(U"./data/temp_modified.csv")
        self.df_price = pd.read_csv(U"./data/price_modified.csv")
        self.hvac_p_cap = 2  # hvac的最大输入功率，单位KW

        self.Ewuxilong = 0.7
        self.eta_hvac = 2.5
        self.A = 0.14

        # 室内温度范围[66.2, 75.2]
        self.T_min = 19 * 1.8 + 32
        self.T_max = 24 * 1.8 + 32

    # 重置环境状态
    def reset(self, isRandom=True, day_num=0):
        if isRandom:
            self.day_number = np.random.randint(0, 121)
        else:
            self.day_number = day_num

        self.time_step = 0

        outdoor_temperature = float(self.df_outTemp[self.get_key(0)][self.day_number])
        indoor_temperature = 23 * 1.8 + 32
        price = float(self.df_price[self.get_key(0)][self.day_number])
        initial_state = [outdoor_temperature, indoor_temperature, price, self.time_step]
        initial_state = np.asarray(initial_state)
        self.current_state = initial_state
        return initial_state

    # 下一状态, action是一个值，为hvac的输入功率
    def step(self, action):
        next_state, reward, done = self.get_next_state(self.current_state, action)
        self.current_state = next_state
        return next_state, reward, done, None

    # 获取下一个状态
    def get_next_state(self, state, action):
        current_outdoorTemp = state[0]
        current_indoorTemp = state[1]

        T_next = self.Ewuxilong * current_indoorTemp + (1 - self.Ewuxilong) * (
                current_outdoorTemp - self.eta_hvac * action / self.A)

        reward = self.get_reward(action, T_next)

        done = False
        self.time_step += 1
        if self.time_step > 23:
            self.time_step = 0
            self.day_number += 1
            done = True

        next_state = [self.get_outdoorTemp(self.day_number, self.time_step), T_next,
                      self.get_price(self.day_number, self.time_step), self.time_step]
        next_state = np.asarray(next_state)
        return next_state, reward, done

    # 获取奖励
    def get_reward(self, action, T_next):
        # c1 为hvac电价费用 c2为温度偏离惩罚
        c1 = action * self.get_price(self.day_number, self.time_step)
        c2 = (max(0, self.T_min - T_next) + max(0, T_next - self.T_max))
        # print('c1=', c1, 'c2=', c2)
        return -15 * c1 - c2

    # 转换时刻
    @staticmethod
    def get_key(time_step):
        time_step = str(time_step)
        return time_step + ':00'

    # 获取电价
    def get_price(self, day_number, time_step):
        time_step = self.get_key(time_step)
        return self.df_price[time_step][day_number]

    # 获取室外温度
    def get_outdoorTemp(self, day_number, time_step):
        time_step = self.get_key(time_step)
        return self.df_outTemp[time_step][day_number]


if __name__ == '__main__':
    env = Hvac()
    for episode in range(100):
        print('-----------------------episode ', episode, '-------------------------------------------')
        state = env.reset()
        for step in range(30):
            print('-----------------------step ', step, '-------------------------------------------')
            action = np.random.uniform(0, 1)
            action = action * env.hvac_p_cap
            next_state, reward, done, info = env.step(action)
            print('outTemp:', state[0], 'inTemp:', state[1], 'price:', state[2], 'daynum', env.day_number, 'time_step:', state[3])
            print('action:', action)
            print('outTemp:', next_state[0], 'inTemp:', next_state[1], 'price:', next_state[2], 'daynum:', env.day_number, 'time_step:', next_state[3])
            print('reward:', reward)
            state = next_state
            if done:
                break
