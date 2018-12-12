'''
    Q_learning for 2d game
'''
import numpy as np
import time
import tkinter as tk
from tkinter import *
from tkinter.scrolledtext import ScrolledText

np.random.seed(2)

ROWS, COLS = 4, 4
ACTIONS = 4
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
EPOCHS = 1000
FRESH_TIME = 0.3

Reward = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])


def build_q_table(rows, cols, actions):
    table = np.zeros((rows, cols, actions))
    return table


def choose_action(state, table):
    state_actions = table[state]

    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action = np.random.randint(0, ACTIONS)
    else:
        action = state_actions.argmax()
    return action


def get_env_feedback(S, A):
    if A == UP:  # up
        if S[0] == 0:
            S_next = S
            R = 0
        else:
            S_next = (S[0] - 1, S[1])
            R = Reward[S_next]
    if A == DOWN:  # down
        if S[0] == ROWS - 1:
            S_next = S
            R = 0
        else:
            S_next = (S[0] + 1, S[1])
            R = Reward[S_next]
    if A == LEFT:  #
        if S[1] == 0:
            S_next = S
            R = 0
        else:
            S_next = (S[0], S[1] - 1)
            R = Reward[S_next]
    if A == RIGHT:
        if S[1] == COLS - 1:
            S_next = S
            R = 0
        else:
            S_next = (S[0], S[1] + 1)
            R = Reward[S_next]
    return S_next, R


def update_env(S, epoch, step_counter, t1):
    env_list = [['-', '-', '-', '-'], ['-', '-', '*', '-'], ['-', '*', 'T', '-'], ['-', '-', '-', '-']]
    env_list[S[0]][S[1]] = 'o'

    t1.delete(epoch * 5 + 1.0, END)
    if epoch > 0:
        t1.insert(END, '\n')

    for i in range(ROWS):
        interaction = ''.join(env_list[i])
        t1.insert(epoch * 5 + i + 1.0, interaction + '\n')
    t1.see(END)
    t1.update()

    if S == (1, 2) or S == (2, 1) or S == (2, 2):
        interaction = 'Episode %d: total_steps=%d' % (epoch + 1, step_counter)
        t1.insert(END, interaction + '\n')
        t1.see(END)
        t1.update()
        time.sleep(2)
    else:
        time.sleep(FRESH_TIME)


def rl():
    root = Tk()
    root.title('Treasure')
    root.geometry('500x500+500+100')

    T1 = ScrolledText(root, wrap=tk.WORD, font=('arial',20))
    T1.pack(expand=1, fill=BOTH)

    q_table = build_q_table(ROWS, COLS, ACTIONS)

    for epoch in range(EPOCHS):
        step_counter = 0
        S = 0, 0
        is_terminal = False
        update_env(S, epoch, step_counter, T1)
        while not is_terminal:
            A = choose_action(S, q_table)
            S_next, R = get_env_feedback(S, A)
            q_predict = q_table[S][A]

            if S_next != (1, 2) and S_next != (2, 1) and S_next != (2, 2):
                q_target = R + LAMBDA * q_table[S_next].max()
            else:
                q_target = R
                is_terminal = True

            q_table[S][A] += ALPHA * (q_target - q_predict)
            S = S_next

            update_env(S, epoch, step_counter + 1, T1)
            step_counter += 1
    root.mainloop()
    return q_table


if __name__ == '__main__':
    q_table = rl()
