'''
    Deep Q_learning for 2d game with only one net
'''
import tensorflow as tf
import numpy as np
import time
import tkinter as tk
from tkinter import *
from tkinter.scrolledtext import ScrolledText
import matplotlib.pylab as plt

ACTIONS = 4
ROWS, COLS = 4, 4
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
FEATURES = 2
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
EPOCHS = 1000
STEP_TIME = 0.5
FRESH_TIME = 0.1
PUNISHMENT = 0.0
BATCH_SIZE = 32
MEMEORY_SIZE = 2000
Reward = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Deep Q_learning_1')
        self.geometry('500x700+500+10')
        self.canvas = self.build_canvas()
        self.Text = self.build_Text()
        self.reset()

    def build_canvas(self):
        canvas = tk.Canvas(self, width=400, height=400)
        self.rt = np.zeros((ROWS, COLS), dtype=np.int)
        for i in range(ROWS):
            for j in range(COLS):
                self.rt[i, j] = canvas.create_rectangle(i * 100, j * 100, (i + 1) * 100, (j + 1) * 100, fill='green')

        canvas.itemconfig(self.rt[1, 2], fill='black')
        canvas.itemconfig(self.rt[2, 1], fill='black')
        canvas.itemconfig(self.rt[2, 2], fill='blue')
        canvas.pack()

        return canvas

    def build_Text(self):
        T1 = ScrolledText(self, wrap=tk.WORD, font=('arial', 13))
        T1.pack(expand=1, fill=BOTH)
        return T1

    def reset(self):
        time.sleep(STEP_TIME)
        for i in range(ROWS):
            for j in range(COLS):
                if i == 0 and j == 0:
                    self.canvas.itemconfig(self.rt[i, j], fill='red')
                elif (i == 1 and j == 2) or (i == 2 and j == 1):
                    self.canvas.itemconfig(self.rt[i, j], fill='black')
                elif i == 2 and j == 2:
                    self.canvas.itemconfig(self.rt[i, j], fill='blue')
                else:
                    self.canvas.itemconfig(self.rt[i, j], fill='green')
        self.update()

    def renew(self, S_Last, S, epoch, step_counter):
        if not S_Last is None:
            self.canvas.itemconfig(self.rt[S_Last], fill='green')
        self.canvas.itemconfig(self.rt[S], fill='red')

        if S == (1, 2) or S == (2, 1) or S == (2, 2):
            interaction = 'Epoch %d/%d: total_steps=%d' % (epoch + 1, EPOCHS, step_counter)
            if S == (1, 2) or S == (2, 1):
                interaction += '  Failure'
            else:
                interaction += '  Success'
            self.Text.insert(END, interaction + '\n')
            self.Text.see(END)
            self.update()
            self.reset()
        else:
            self.update()
            time.sleep(FRESH_TIME)


class QLAgent():
    def __init__(self, Graph_Write=False):
        self.store = np.zeros((MEMEORY_SIZE, FEATURES * 2 + 2))
        self.build_net()

        if Graph_Write:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            writer.flush()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_hist = []

    def choose_action(self, state):
        if np.random.uniform() < EPSILON:
            state = np.array(state)[np.newaxis, :]
            action_value = self.sess.run(self.q_out, feed_dict={self.s: state})[0]
            c = np.random.permutation(ACTIONS)
            action = c[action_value[c].argmax()]
        else:
            action = np.random.randint(0, ACTIONS)
        return action

    def get_env_feedback(self, S, A):
        if A == UP:  # up
            if S[0] == 0:
                S_next = S
                R = PUNISHMENT
            else:
                S_next = (S[0] - 1, S[1])
                R = Reward[S_next]
        if A == DOWN:  # down
            if S[0] == ROWS - 1:
                S_next = S
                R = PUNISHMENT
            else:
                S_next = (S[0] + 1, S[1])
                R = Reward[S_next]
        if A == LEFT:  #
            if S[1] == 0:
                S_next = S
                R = PUNISHMENT
            else:
                S_next = (S[0], S[1] - 1)
                R = Reward[S_next]
        if A == RIGHT:
            if S[1] == COLS - 1:
                S_next = S
                R = PUNISHMENT
            else:
                S_next = (S[0], S[1] + 1)
                R = Reward[S_next]
        if S_next == (1, 2) or S_next == (2, 1) or S_next == (2, 2):
            is_terminal = True
        else:
            is_terminal = False

        return S_next, R, is_terminal

    def build_net(self):
        # -------------- build eval network ------------------
        self.s = tf.placeholder(tf.float32, [None, FEATURES], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, ACTIONS], name='Q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                                                          tf.random_normal_initializer(0., 0.3), \
                                                          tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [FEATURES, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, ACTIONS], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [ACTIONS], initializer=b_initializer, collections=c_names)
                self.q_out = tf.matmul(l1, w2) + b2

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_out))
        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

    def store_transition(self, S, A, R, S_next):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((S, [A, R], S_next))
        self.store[self.memory_counter % MEMEORY_SIZE] = transition
        self.memory_counter += 1

    def learn(self):
        batch_memory = self.store[
            np.random.permutation(MEMEORY_SIZE)[:BATCH_SIZE]] \
            if self.memory_counter > MEMEORY_SIZE else self.store[
            np.random.permutation(self.memory_counter)[:BATCH_SIZE]]

        q_out = self.sess.run(self.q_out, feed_dict={self.s: batch_memory[:, :FEATURES]})
        q_next = self.sess.run(self.q_out, feed_dict={self.s: batch_memory[:, -FEATURES:]})

        q_target = q_out.copy()

        q_target[np.arange(BATCH_SIZE, dtype=np.int), batch_memory[:, FEATURES].astype(np.int)] = \
            batch_memory[:, FEATURES + 1] + LAMBDA * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :FEATURES], self.q_target: q_target})
        self.cost_hist.append(self.cost)

    def plot_cost(self):
        plt.plot(self.cost_hist)
        plt.show()


class usr():
    def __init__(self):
        self.env = Env()
        self.agent = QLAgent(Graph_Write=False)

    def run(self):
        step = 0
        for epoch in range(EPOCHS):
            step_counter = 0
            S = 0, 0
            is_terminal = False
            self.env.renew(None, S, epoch, step_counter)

            while not is_terminal:
                A = self.agent.choose_action(S)
                S_next, R, is_terminal = self.agent.get_env_feedback(S, A)

                self.agent.store_transition(S, A, R, S_next)

                if step > BATCH_SIZE:
                    self.agent.learn()

                S_tmp = S
                S = S_next

                step += 1

                if S != S_tmp:
                    step_counter += 1
                    self.env.renew(S_tmp, S, epoch, step_counter)
        self.env.mainloop()
        self.agent.plot_cost()


if __name__ == '__main__':
    usr().run()
