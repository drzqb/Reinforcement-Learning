'''
    Actor-Critic for CartPole
'''
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

env = gym.make('CartPole-v0')
env = env.unwrapped

ACTIONS = env.action_space.n
FEATURES = env.observation_space.shape[0]
EPOCHS = 3000
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions

        with tf.variable_scope('Actor'):
            with tf.name_scope('Input'):
                self.s = tf.placeholder(tf.float32, [1, n_features], 'state')
                self.a = tf.placeholder(tf.int32, None, 'act')
                self.td_error = tf.placeholder(tf.float32, None, 'td_error')

            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = log_prob * self.td_error

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict=feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        return np.random.choice(np.arange(self.n_actions), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        with tf.variable_scope('Critic'):
            with tf.name_scope('Input'):
                self.s = tf.placeholder(tf.float32, [1, n_features], "state")
                self.v_next = tf.placeholder(tf.float32, [1, 1], "v_next")
                self.r = tf.placeholder(tf.float32, None, 'r')

            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

            with tf.variable_scope('TD_error'):
                self.td_error = self.r + GAMMA * self.v_next - self.v
                self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_next):
        s, s_next = s[np.newaxis, :], s_next[np.newaxis, :]

        v_next = self.sess.run(self.v, {self.s: s_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_next: v_next, self.r: r})
        return td_error


class usr():
    def __init__(self, output_graph=True):
        self.sess = tf.Session()
        self.env = env
        self.actor = Actor(self.sess, n_features=FEATURES, n_actions=ACTIONS, lr=LR_A)
        self.critic = Critic(self.sess, n_features=FEATURES, lr=LR_C)
        self.sess.run(tf.global_variables_initializer())

        if output_graph:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            writer.flush()

        self.Reward = []

    def run(self):
        for epoch in range(EPOCHS):
            step_counter = 0
            S = self.env.reset()
            is_terminal = False

            while not is_terminal:
                self.env.render()
                A = self.actor.choose_action(S)

                S_next, R, is_terminal, info = self.env.step(A)

                if is_terminal: R = -20.

                td_error = self.critic.learn(S, R, S_next)
                self.actor.learn(S, A, td_error)

                S = S_next

                step_counter += 1

                if is_terminal:
                    print('Epoch %d/%d Reward:%d' % (epoch, EPOCHS, step_counter))
                    self.Reward.append(step_counter)

        self.plot_reward()

    def plot_cost_reward(self):
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
