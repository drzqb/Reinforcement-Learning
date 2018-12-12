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
STEP_TIME = 0.5
FRESH_TIME = 0.1
PUNISHMENT = -1000.0
Reward = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])


def build_q_table(rows, cols, actions):
    table = np.zeros((rows, cols, actions))
    return table


def choose_action(state, table):
    state_actions = table[state]

    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action = np.random.randint(0, ACTIONS)
    else:
        c = np.random.permutation(ACTIONS)
        action = c[state_actions[c].argmax()]

    return action


def get_env_feedback(S, A):
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


def update_env(S, epoch, step_counter, w, rt, t1):
    w.itemconfig(rt[S], fill='red')
    w.update()

    if S == (1, 2) or S == (2, 1) or S == (2, 2):
        interaction = 'Episode %d/%d: total_steps=%d' % (epoch + 1, EPOCHS, step_counter)
        if S == (1, 2) or S == (2, 1):
            interaction += '  Failure'
        else:
            interaction += '  Success'
        t1.insert(END, interaction + '\n')
        t1.see(END)
        t1.update()
        time.sleep(STEP_TIME)
    else:
        time.sleep(FRESH_TIME)
    if S == (1, 2) or S == (2, 1):
        w.itemconfig(rt[S], fill='black')
    elif S == (2, 2):
        w.itemconfig(rt[S], fill='blue')
    else:
        w.itemconfig(rt[S], fill='green')
    w.update()


def rl():
    root = Tk()
    root.title('Treasure')
    root.geometry('500x500+500+100')

    w = Canvas(root, width=400, height=400)
    w.pack()

    rt = np.zeros((ROWS, COLS), dtype=np.int)
    for i in range(ROWS):
        for j in range(COLS):
            rt[i, j] = w.create_rectangle(i * 100, j * 100, (i + 1) * 100, (j + 1) * 100, fill='green')

    w.itemconfig(rt[1, 2], fill='black')
    w.itemconfig(rt[2, 1], fill='black')
    w.itemconfig(rt[2, 2], fill='blue')
    T1 = ScrolledText(root, wrap=tk.WORD, font=('arial', 13))
    T1.pack(expand=1, fill=BOTH)

    q_table = build_q_table(ROWS, COLS, ACTIONS)

    for epoch in range(EPOCHS):
        step_counter = 0
        S = 0, 0
        is_terminal = False
        update_env(S, epoch, step_counter, w, rt, T1)
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

            update_env(S, epoch, step_counter + 1, w, rt, T1)
            step_counter += 1
    root.mainloop()
    return q_table


if __name__ == '__main__':
    q_table = rl()
