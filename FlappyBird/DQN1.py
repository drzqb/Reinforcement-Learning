'''
    Deep Q_learning for 2d game FlappyBird with four images
'''
import numpy as np
import tensorflow as tf
import cv2
import sys
import matplotlib.pylab as plt

sys.path.append('game')
import wrapped_flappy_bird as env

tf.flags.DEFINE_string('mode', 'train', 'The mode of train or predict as follows: '
                                        'train'
                                        'play')
FLAGS = tf.flags.FLAGS

np.random.seed(1)
tf.set_random_seed(1)

WIDTH, HEIGHT = 80, 80
ACTIONS = 2
ACTION = np.array([[1, 0], [0, 1]], dtype=np.int32)
INITIAL_EPSILON = 0.0
EPSILON = 1.0
ALPHA = 0.1
LAMBDA = 0.99
OBSERVE = 100
EXPLORE = 10000
EPOCHS = 2000000
BATCH_SIZE = 8
MEMORY_SIZE = 20000
SAVE_MODEL_PER = 100
LEARNING_RATE = 1e-4


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def convert(S):
    S = cv2.cvtColor(cv2.resize(S, (HEIGHT, WIDTH)), cv2.COLOR_BGR2GRAY)
    _, S = cv2.threshold(S, 1, 255, cv2.THRESH_BINARY)
    return S / 255.


class QLAgent():
    def __init__(self, Graph_Write=False):
        self.build_net()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=1)
        if FLAGS.mode == 'train':
            self.epsilon = INITIAL_EPSILON
            self.epsilon_increment = (EPSILON - INITIAL_EPSILON) / EXPLORE
            self.store = []

            if Graph_Write:
                writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='FB2_DQN')
                writer.flush()
            self.sess.run(tf.global_variables_initializer())
            self.cost_hist = []
            self.Reward = []
        elif FLAGS.mode == 'play':
            self.saver.restore(self.sess, 'saved_networks/FB2_DQN')

    def build_net(self):
        w_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        w_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        w_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        w_fc1 = weight_variable([1600, 512])
        b_fc1 = weight_variable([512])

        w_fc2 = weight_variable([512, ACTIONS])
        b_fc2 = weight_variable([ACTIONS])

        self.s = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 4], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, ACTIONS], name='Q_target')

        h_conv1 = tf.nn.relu(conv2d(self.s, w_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1) + b_conv3)

        h_flatten = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_flatten, w_fc1) + b_fc1)

        self.q_eval = tf.matmul(h_fc1, w_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval), axis=-1))
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def choose_action(self, state):
        if np.random.uniform() <= self.epsilon:
            state = np.reshape(state, (1, HEIGHT, WIDTH, 4))
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]
            action = action_value.argmax()
        else:
            action = 0
        return action

    def epsilon_change(self, t):
        if t >= OBSERVE:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < EPSILON else EPSILON

    def choose_action_deterministic(self, state):
        state = np.reshape(state, (1, HEIGHT, WIDTH, 4))
        action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]
        action = np.argmax(action_value)
        return action

    def store_transition(self, S, A, R, S_next, is_terminal):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = [S, A, R, S_next, is_terminal]
        self.store.append(transition)
        self.memory_counter += 1
        if self.memory_counter > MEMORY_SIZE:
            self.store.pop(0)

    def learn(self):
        index = np.random.permutation(np.minimum(self.memory_counter, MEMORY_SIZE))[:BATCH_SIZE]
        batch_s = [self.store[id][0] for id in index]
        batch_a = [self.store[id][1] for id in index]
        batch_r = [self.store[id][2] for id in index]
        batch_s_next = [self.store[id][3] for id in index]
        batch_is_terminal = [self.store[id][4] for id in index]

        q_next = self.sess.run(self.q_eval, feed_dict={self.s: batch_s_next})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_s})

        q_target = q_eval.copy()

        for i in range(BATCH_SIZE):
            if batch_is_terminal[i]:
                q_target[i, batch_a[i]] = batch_r[i]
            else:
                q_target[i, batch_a[i]] = batch_r[i] + LAMBDA * np.max(q_next[i])

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: batch_s, self.q_target: q_target})
        self.cost_hist.append(self.cost)

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


class Usr():
    def __init__(self):
        self.env = env
        self.agent = QLAgent(Graph_Write=False)

    def train(self):
        step = 0
        counter_max = 0
        for epoch in range(1, EPOCHS + 1):
            step_counter = 0
            game_state = self.env.GameState()
            S, R, is_terminal = game_state.frame_step(ACTION[0])
            S = convert(S)
            S = np.stack((S, S, S, S), axis=2)

            while not is_terminal:
                A = self.agent.choose_action(S)
                S_next, R, is_terminal = game_state.frame_step(ACTION[A])

                S_next = np.append(np.expand_dims(convert(S_next), 2), S[:, :, :3], axis=2)

                self.agent.store_transition(S, A, R, S_next, is_terminal)

                S = S_next
                step_counter += 1

                step += 1
                self.agent.epsilon_change(step)

            print('Epoch %d/%d Reward:%d Epsilon:%f' % (epoch, EPOCHS, step_counter, self.agent.epsilon))
            self.agent.Reward.append(step_counter)
            if step_counter > counter_max and self.agent.epsilon == 1.0:
                counter_max = step_counter
                self.agent.saver.save(self.agent.sess, 'saved_networks/FB2_DQN')
                print('model saved successfully!')
            if step > OBSERVE:
                for l in range(min(step_counter, 50)):
                    self.agent.learn()

        self.agent.plot_cost_reward()

    def play(self):
        for test_step in range(1, 1001):
            step_counter = 0
            game_state = self.env.GameState()
            S, _, is_terminal = game_state.frame_step(ACTION[0])
            S = convert(S)
            S = np.stack((S, S, S, S), axis=2)

            while not is_terminal:
                A = self.agent.choose_action_deterministic(S)
                S_next, R, is_terminal = game_state.frame_step(ACTION[A])

                S_next = np.append(np.expand_dims(convert(S_next), 2), S[:, :, :3], axis=2)

                S = S_next
                step_counter += 1
            print('Test %d Reward:%d' % (test_step, step_counter))


def main(unused_argvs):
    usr = Usr()

    if FLAGS.mode == 'train':
        usr.train()
    elif FLAGS.mode == 'play':
        usr.play()


if __name__ == '__main__':
    tf.app.run()
