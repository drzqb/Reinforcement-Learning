'''
    Deterministic Policy Gradient for CartPole
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped

GAMMA = 0.99
LR_A = 0.001
LR_C = 0.001

FEATURES = env.observation_space.shape[0]
ACTIONS = env.action_space.n
EPOCHS = 3000
HIDDEN = 40
BATCH_SIZE = 32


class Actor():
    def __init__(self, sess):
        self.sess = sess
        self.build_net()

    def build_net(self):
        with tf.variable_scope("Actor"):
            with tf.variable_scope('Input'):
                self.tf_s = tf.placeholder(tf.float32, [None, FEATURES], 's')
                self.tf_a = tf.placeholder(tf.int32, [None], name="actions_index")
                self.tf_av = tf.placeholder(tf.float32, [None], name="target_eval")

            with tf.variable_scope('Layer'):
                self.all_act_prob = Sequential([
                    Dense(HIDDEN, activation='relu'),
                    Dense(HIDDEN, activation='relu'),
                    Dense(ACTIONS, activation='softmax')
                ])(self.tf_s)

            with tf.name_scope('loss'):
                log_act_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_a, ACTIONS), 1)
                loss = tf.reduce_mean(log_act_prob * self.tf_av)

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=LR_A).minimize(loss)

    def choose_action(self, S):
        prob_weight = self.sess.run(self.all_act_prob, feed_dict={self.tf_s: S[np.newaxis, :]})
        action = np.random.choice(ACTIONS, p=np.ravel(prob_weight))
        return action

    def choose_action_deterministic(self, S):
        prob_weight = self.sess.run(self.all_act_prob, feed_dict={self.tf_s: S[np.newaxis, :]})
        return np.argmax(prob_weight[0])

    def learn(self, store_s, store_a, av):
        self.sess.run(self.train_op, feed_dict={
            self.tf_s: store_s,
            self.tf_a: store_a,
            self.tf_av: av
        })


class Critic():
    def __init__(self, sess):
        self.sess = sess
        self.build_net()

    def build_net(self):
        with tf.variable_scope("Critic"):
            with tf.variable_scope('Input'):
                self.tf_s = tf.placeholder(tf.float32, [None, FEATURES], 's')
                self.tf_target = tf.placeholder(tf.float32, [None], 'discounted_r')

            with tf.variable_scope('Layer'):
                self.eval = tf.reduce_sum(Sequential([
                    Dense(HIDDEN, activation='relu'),
                    Dense(HIDDEN, activation='relu'),
                    Dense(1)
                ])(self.tf_s), axis=1)

            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.square(self.tf_target - self.eval))

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=LR_C).minimize(loss)

    def learn(self, store_s, target):
        self.sess.run(self.train_op, feed_dict={self.tf_s: store_s, self.tf_target: target})


class usr():
    def __init__(self, output_graph=True):
        self.store_s = []
        self.store_a = []
        self.store_r = []

        self.sess = tf.Session()
        self.env = env
        self.actor = Actor(self.sess)
        self.critic = Critic(self.sess)
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            writer.flush()

        self.Reward = []

    def store_transition(self, S, A, R):
        self.store_s.append(S)
        self.store_a.append(A)
        self.store_r.append(R)

    def calculate_target(self):
        discount_r = np.zeros_like(self.store_r)
        tmp = 0.0

        for t in reversed(range(len(self.store_r))):
            tmp = tmp * GAMMA + self.store_r[t]
            discount_r[t] = tmp

        return discount_r

    def store_reset(self):
        self.store_s = []
        self.store_a = []
        self.store_r = []

    def run(self):
        for epoch in range(1, EPOCHS + 1):
            step_counter = 0
            S = self.env.reset()
            is_terminal = False

            while not is_terminal:
                # self.env.render()
                A = self.actor.choose_action(S)

                S_next, R, is_terminal, info = self.env.step(A)

                if is_terminal: R = -20.

                self.store_transition(S, A, R)

                if is_terminal:
                    target = self.calculate_target()

                    total_batch = len(self.store_s) // BATCH_SIZE + 1
                    index = np.array_split(np.random.permutation(len(self.store_s)), total_batch)

                    t_s = np.array(self.store_s)
                    t_a = np.array(self.store_a)

                    for id in index:
                        eval = self.sess.run(self.critic.eval, feed_dict={self.critic.tf_s: t_s[id]})

                        av = target[id] - eval

                        self.actor.learn(t_s[id], t_a[id], av)
                        self.critic.learn(t_s[id], target[id])

                    self.store_reset()
                    print('Epoch %d/%d Reward:%d' % (epoch, EPOCHS, step_counter))
                    self.Reward.append(step_counter)

                else:
                    S = S_next
                    step_counter += 1

            if epoch % 10 == 0:
                for test in range(1, 11):
                    step_counter = 0
                    S = self.env.reset()
                    is_terminal = False

                    while not is_terminal:
                        self.env.render()
                        A = self.actor.choose_action_deterministic(S)

                        S_next, _, is_terminal, _ = self.env.step(A)

                        if is_terminal:
                            print('Test %d Reward:%d' % (test, step_counter))
                        else:
                            S = S_next
                            step_counter += 1

        self.plot_reward()

    def plot_reward(self):
        plt.subplot(121)
        plt.plot(self.Reward)
        plt.title('Reward')
        plt.subplot(122)
        self.average_reward = [np.mean(self.Reward[i * 10:(i + 1) * 10]) for i in range(len(self.Reward) // 10)]
        plt.plot(self.average_reward)
        plt.title('Average Reward')
        plt.show()


if __name__ == '__main__':
    usr().run()
