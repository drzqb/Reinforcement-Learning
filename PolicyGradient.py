'''
    PolicyGradient for CartPole
'''
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

np.random.seed(1)
tf.set_random_seed(1)

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

ACTIONS = env.action_space.n
FEATURES = env.observation_space.shape[0]
EPOCHS = 3000


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.02,
                 reward_decay=0.99,
                 output_graph=True):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.store_s = []
        self.store_a = []
        self.store_r = []
        self.Reward = []

        self.build_net()
        self.sess = tf.Session()

        if output_graph:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
            writer.flush()

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        with tf.name_scope('inputs'):
            self.tf_s = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_a = tf.placeholder(tf.int32, [None], name="actions_index")
            self.tf_r = tf.placeholder(tf.float32, [None], name="actions_Reward")

        layer = tf.layers.dense(
            inputs=self.tf_s,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_a, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_r)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, S):
        prob_weight = self.sess.run(self.all_act_prob, feed_dict={self.tf_s: S[np.newaxis, :]})
        action = np.random.choice(range(self.n_actions), p=np.ravel(prob_weight))
        return action

    def store_transition(self, S, A, R):
        self.store_s.append(S)
        self.store_a.append(A)
        self.store_r.append(R)

    def learn(self):
        discounted_norm_r = self.discount_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_s: self.store_s,
            self.tf_a: self.store_a,
            self.tf_r: discounted_norm_r
        })
        self.store_s = []
        self.store_a = []
        self.store_r = []

    def discount_and_norm_rewards(self):
        discount_r = np.zeros_like(self.store_r)
        tmp = 0.0

        for t in reversed(range(len(self.store_r))):
            tmp = tmp * self.gamma + self.store_r[t]
            discount_r[t] = tmp

        return (discount_r - np.mean(discount_r)) / np.std(discount_r)

    def plot_cost_reward(self):
        plt.subplot(121)
        plt.plot(self.Reward)
        plt.title('Reward')
        plt.subplot(122)
        self.average_reward = [np.mean(self.Reward[i * 10:(i + 1) * 10]) for i in range(len(self.Reward) // 10)]
        plt.plot(self.average_reward)
        plt.title('Average Reward')
        plt.show()


class usr():
    def __init__(self):
        self.env = env
        self.agent = PolicyGradient(ACTIONS, FEATURES, output_graph=False)

    def run(self):
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

                self.agent.store_transition(S, A, R)

                if is_terminal:
                    self.agent.learn()
                    print('Epoch %d/%d Reward:%d' % (epoch, EPOCHS, step_counter))
                    self.agent.Reward.append(step_counter)
                S = S_next

                step_counter += 1
        self.agent.plot_cost_reward()


if __name__ == '__main__':
    usr().run()
