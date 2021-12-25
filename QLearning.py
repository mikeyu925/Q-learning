import numpy as np
import pandas as pd
import time

np.random.seed(2)

class QLearningTable:
    # 构造函数
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # 可用动作 是一个列表 本题是[0~3]
        self.lr = learning_rate    # 学习率
        self.gamma = reward_decay    # 奖励递减值
        self.epsilon = e_greedy  # 贪婪度 greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # Q-表 初始全为0，并初始化columns为执行策略


    def choose_action(self, observation):
        self.check_state_exist(observation)  # 查看是否有当前状态
        # action selection
        if np.random.uniform() < self.epsilon:  # 生成一个随机数 判断是否贪婪操作
            # choose best action
            state_action = self.q_table.loc[observation, :]  # 得到val of action
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index) #因为可能有多个值相同的action，随机挑一个
        else:  # 防止进入局部最优解出不来
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    # 查看是否存在当前状态，没有则添加进 q_table
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )