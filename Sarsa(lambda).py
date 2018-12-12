'''
    Sarsa(lambda) for 2d game
'''
import numpy as np
import time
import tkinter as tk
from tkinter import *
from tkinter.scrolledtext import ScrolledText

ACTIONS = 4
ROWS, COLS = 4, 4
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
DECAY = 0.8
EPOCHS = 1000
STEP_TIME = 0.5
FRESH_TIME = 0.1
PUNISHMENT = 0.0
Reward = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.title('Sarsa(lambda)')
        self.geometry('500x500+500+100')
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
            interaction = 'Episode %d/%d: total_steps=%d' % (epoch + 1, EPOCHS, step_counter)
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


class SarsaAgent():
    def __init__(self):
        self.q_table = np.zeros((ROWS, COLS, ACTIONS))
        self.e_table = np.zeros_like(self.q_table)

    def choose_action(self, state, table):
        state_actions = table[state]

        if np.random.uniform() < EPSILON:
            c = np.random.permutation(ACTIONS)
            action = c[state_actions[c].argmax()]
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
        return S_next, R

    def learn(self, S, A, S_next, R):
        q_predict = self.q_table[S][A]

        if S_next != (1, 2) and S_next != (2, 1) and S_next != (2, 2):
            A_next = self.choose_action(S_next, self.q_table)
            q_target = R + LAMBDA * self.q_table[S_next][A_next]
            is_terminal = False
        else:
            A_next = A
            q_target = R
            is_terminal = True

        delta = q_target - q_predict
        if S!=S_next:
            self.e_table[S][A] += 1

        self.q_table += ALPHA * delta * self.e_table
        self.e_table *= LAMBDA * DECAY
        return A_next, is_terminal


class usr():
    def __init__(self):
        self.env = Env()
        self.agent = SarsaAgent()

    def run(self):
        for epoch in range(EPOCHS):
            step_counter = 0

            self.agent.e_table *= 0.0
            S = 0, 0
            is_terminal = False
            self.env.renew(None, S, epoch, step_counter)
            A = self.agent.choose_action(S, self.agent.q_table)

            while not is_terminal:
                S_next, R = self.agent.get_env_feedback(S, A)

                A_next, is_terminal = self.agent.learn(S, A, S_next, R)

                S_tmp = S
                S = S_next
                A = A_next

                if S != S_tmp:
                    self.env.renew(S_tmp, S, epoch, step_counter + 1)
                    step_counter += 1
        return self.agent.q_table


if __name__ == '__main__':
    q_table = usr().run()
