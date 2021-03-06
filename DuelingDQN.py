'''
    Dueling DQN for 2d game CartPole
'''
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env = env.unwrapped

ACTIONS = env.action_space.n
FEATURES = env.observation_space.shape[0]
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
EPOCHS = 1000
BATCH_SIZE = 32
MEMEORY_SIZE = 2000


class QLAgent():
    def __init__(self, Graph_Write=False):
        self.store = np.zeros((MEMEORY_SIZE, FEATURES * 2 + 2))
        self.build_net()

        if Graph_Write:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            writer.flush()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.replace_target_params()

        self.cost_hist = []
        self.Reward = []
        self.replace_target_iter = 100
        self.learn_step_counter = 0

    def choose_action(self, state):
        if np.random.uniform() < EPSILON:
            state = state[np.newaxis, :]
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]
            c = np.random.permutation(ACTIONS)
            action = c[action_value[c].argmax()]
        else:
            action = np.random.randint(0, ACTIONS)
        return action

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

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, ACTIONS], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [ACTIONS], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            with tf.name_scope('Q'):
                self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))

        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

        # --------------- build target network -------------------------
        self.s_next = tf.placeholder(tf.float32, [None, FEATURES], name='s_next')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [FEATURES, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_next, w1) + b1)

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, ACTIONS], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [ACTIONS], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            with tf.name_scope('Q'):
                self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1,keepdims=True))

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def replace_target_params(self):
        self.sess.run(self.replace_target_op)

    def store_transition(self, S, A, R, S_next):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((S, [A, R], S_next))
        self.store[self.memory_counter % MEMEORY_SIZE] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_params()

        batch_memory = self.store[
            np.random.permutation(MEMEORY_SIZE)[:BATCH_SIZE]] \
            if self.memory_counter > MEMEORY_SIZE else self.store[
            np.random.permutation(self.memory_counter)[:BATCH_SIZE]]

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_next: batch_memory[:, -FEATURES:],
                                                  self.s: batch_memory[:, :FEATURES]})
        q_eval_next = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, -FEATURES:]})

        q_target = q_eval.copy()

        q_target[np.arange(BATCH_SIZE, dtype=np.int), batch_memory[:, FEATURES].astype(np.int)] = \
            batch_memory[:, FEATURES + 1] + LAMBDA * q_next[
                np.arange(BATCH_SIZE, dtype=np.int), np.argmax(q_eval_next, axis=1)]

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :FEATURES], self.q_target: q_target})
        self.cost_hist.append(self.cost)
        self.learn_step_counter += 1

    def plot_cost_reward(self):
        plt.subplot(131)
        plt.plot(self.cost_hist)
        plt.title('cost')
        plt.subplot(132)
        plt.plot(self.Reward)
        plt.title('Reward')
        plt.subplot(133)
        self.average_reward = [np.mean(self.Reward[i * 10:(i + 1) * 10]) for i in range(len(self.Reward) // 10)]
        plt.plot(self.average_reward)
        plt.title('Average Reward')
        plt.show()


class usr():
    def __init__(self):
        self.env = env
        self.agent = QLAgent(Graph_Write=True)

    def run(self):
        step = 0
        for epoch in range(EPOCHS):
            step_counter = 0
            S = self.env.reset()
            is_terminal = False

            while not is_terminal:
                self.env.render()
                A = self.agent.choose_action(S)

                S_next, R, is_terminal, info = self.env.step(A)

                x, x_dot, theta, theta_dot = S_next
                r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
                r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
                R = r1 + r2

                self.agent.store_transition(S, A, R, S_next)

                if step > BATCH_SIZE:
                    self.agent.learn()

                if is_terminal:
                    print('Epoch %d/%d Reward:%d' % (epoch, EPOCHS, step_counter))
                    self.agent.Reward.append(step_counter)
                S = S_next

                step += 1
                step_counter += 1
        self.agent.plot_cost_reward()


if __name__ == '__main__':
    usr().run()
