from collections import deque
import random

import gym
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            batch_size=32,
            e_greedy=0.2,
            e_greedy_increment=0.00001,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.replay_memory = deque(maxlen=1000)
        self.batch_size = batch_size
        self.e_greedy_increment = e_greedy_increment  # epsilon的增量，随机采样，大于epsilon则采取随机动作
        self.epsilon = e_greedy
        self.model = self.create_q_model()
        self.model_target = self.create_q_model()
        self.step = 0

    def create_q_model(self):
        model = keras.Sequential([
            layers.Dense(64, input_shape=(1, self.n_features), activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        # sgd = optimizers.SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model


    def choose_action(self, state):
        if self.epsilon > 0:
            self.epsilon -= self.e_greedy_increment
        if np.random.uniform() <= self.epsilon:  # 采用随机动作
            return np.random.randint(self.n_actions)
        actions_value = self.model.predict(state.reshape(1, 1, self.n_features))
        return np.argmax(actions_value)

    def learn(self):
        if self.step % 100 == 0:  # 每100步同步一下target网络参数
            RL.update_model_target()

        if self.step % 5 == 0 and len(self.replay_memory) >= 200:
            batch = random.sample(self.replay_memory, self.batch_size)
            Obs, Action, Reward, N_s, Done = [], [], [], [], []
            for (obs, action, reward, n_s, done) in batch:
                Obs.append(obs)
                Action.append(action)
                Reward.append(reward)
                N_s.append(n_s)
                Done.append(done)

            Obs = np.array(Obs).astype("float32")
            Action = np.array(Action).astype("int64")
            Reward = np.array(Reward).astype("float32")
            N_s = np.array(N_s).astype("float32")
            Done = np.array(Done).astype("float32")

            Q = self.model.predict(Obs.reshape(self.batch_size, 1, self.n_features))

            # DQN
            # Q_ = self.model_target.predict(N_s.reshape(self.batch_size, 1, self.n_features))
            #
            # for i in range(self.batch_size):
            #     Q[i][0][Action[i]] = (Reward[i] + self.reward_decay * np.max(Q_[i][0]))

            # DDQN
            Q_ = self.model.predict(N_s.reshape(self.batch_size, 1, self.n_features))
            Q_next = self.model_target.predict(N_s.reshape(self.batch_size, 1, self.n_features))
            for i in range(self.batch_size):
                Qnext = Q_next[i][0][np.argmax(Q_[i][0])]
                Q[i][0][Action[i]] = (Reward[i] + self.reward_decay * Qnext)

            # 训练决策网络
            self.model.fit(Obs.reshape(self.batch_size, 1, self.n_features), Q, verbose=0)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def update_model_target(self):
        self.model_target.set_weights(self.model.get_weights())

    def save(self, path):
        self.model_target.save(path)


def train():
    rewards = []
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        while True:
            # env.render()
            RL.step += 1
            action = RL.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            RL.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            RL.learn()
            if done:
                rewards.append(episode_reward)
                print("{} Episode, score = {} max = {}".
                      format(episode + 1, episode_reward, np.max(rewards)))
                break
        # 提前终止训练
        if len(rewards) >= 6 and np.mean(rewards[-6:]) >= 180:
            break
    RL.save()
    plt.plot(np.array(rewards), c='r')
    plt.show()


def test():
    model = keras.models.load_model('./logs/' + name + 'cartpole.h5')
    score = 0
    for i in range(10):
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(model.predict(s.reshape(1, 1, 4)))
            next_s, reward, done, info = env.step(a)
            score += reward
            s = next_s
            if done:
                break
    print('average score = {}'.format(score / 10))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    name = 'test_1_'
    RL = DQN(env.action_space.n, env.observation_space.shape[0])
    # train()
    test()
