from math import floor
from Actions import *
import numpy as np

STANDARD_REWARD = 1
CRASH_REWARD = -20
END_REWARD = -20
ACTION_SUCCESSFUL_RATE = 0.8
N_ROW = 5
N_COL = 5
CRASH_BLOCKS = [(2, 2), (0, 2), (2, 1), (3, 1), (0, 3), (2, 4)]
INIT_STATE = (0, 0, 4, 4)


class TwoPlayerEnv:
    def __init__(self):
        self.n_rows = N_ROW
        self.n_cols = N_COL
        self.crash_blocks = CRASH_BLOCKS
        self.n_blocks = self.n_rows * self.n_cols

        self.states = self.__initialize_states()
        self.crash_states_p1, self.crash_states_p2 = self.__initialize_crash_states()
        self.terminal_states = self.__initialize_terminal_states()
        self.rewards = self.__initialize_rewards()

        self.init_state = self.rc2state(INIT_STATE[0], INIT_STATE[1], INIT_STATE[2], INIT_STATE[3])

    def play(self, player, opponent):
        state = self.init_state
        terminated = False
        cumulative_reward_player = 0
        while not terminated:
            action_player = player.choose_action(state)
            action_opponent = opponent.choose_action(state)

            row_player, col_player, row_opponent, col_opponent = self.state2rc(state)

            if self.__is_action_valid(row_player, col_player, action_player):
                row_player += get_movement(action_player)[0]
                col_player += get_movement(action_player)[1]
            if self.__is_action_valid(row_opponent, col_opponent, action_opponent):
                row_opponent += get_movement(action_opponent)[0]
                col_opponent += get_movement(action_opponent)[1]

            new_state = self.rc2state(row_player, col_player, row_opponent, col_opponent)

            reward_player = self.get_state_reward(new_state)
            reward_opponent = - self.get_state_reward(new_state)

            player.observe(reward_player, state, action_player, action_opponent, new_state)
            opponent.observe(reward_opponent, state, action_opponent, action_player, new_state)

            cumulative_reward_player += reward_player

            state = new_state
            if self.is_terminal_state(state):
                terminated = True

        return cumulative_reward_player

    def __is_action_valid(self, row, col, action):
        new_row = row + get_movement(action)[0]
        new_col = col + get_movement(action)[1]
        if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
            return True
        return False

    def is_crash_state(self, state):
        return state in self.crash_states_p1 or state in self.crash_states_p2

    def is_terminal_state(self, state):
        return state in self.terminal_states

    def get_n_states(self):
        '''
        Get the number of states of the game
        :return: number of states
        '''
        return len(self.states)

    def get_n_actions(self):
        return N_ACTIONS

    def get_state_reward(self, state):
        '''
        Get the reward of each state
        :param state: index of state
        :return: the reward of that state
        '''
        return self.rewards[state]

    def __initialize_crash_states(self):
        '''
        Get list of crash states [[p1_crash], [p2_crash]]
        :return: an array of crash states' index
        '''
        crash_states_p1 = []
        crash_states_p2 = []
        for crash_block in self.crash_blocks:
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if (i, j) != crash_block:
                        crash_states_p2.append(self.rc2state(i, j, crash_block[0], crash_block[1]))
                        crash_states_p1.append(self.rc2state(crash_block[0], crash_block[1], i, j))
        return crash_states_p1, crash_states_p2

    def __initialize_states(self):
        return np.arange(0, self.n_blocks * self.n_blocks)

    def __initialize_rewards(self):
        rewards = np.ones(self.get_n_states()) * STANDARD_REWARD
        for crash_state in self.crash_states_p1:
            rewards[crash_state] = CRASH_REWARD
        for crash_state in self.crash_states_p2:
            rewards[crash_state] = -CRASH_REWARD
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                rewards[self.rc2state(i, j, i, j)] = END_REWARD
        return rewards

    def __initialize_terminal_states(self):
        terminal_states = []
        for crash_state in self.crash_states_p1:
            terminal_states.append(crash_state)
        for crash_state in self.crash_states_p2:
            terminal_states.append(crash_state)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                terminal_states.append(self.rc2state(i, j, i, j))
        return terminal_states

    def rc2state(self, row1, col1, row2, col2):
        '''
        Convert row, col of player 1 and 2 to index of state
        :param row1: row of player1
        :param col1: col of player1
        :param row2: row of player2
        :param col2: col of player2
        :return: converted index of state
        '''
        state1 = row1 * self.n_cols + col1
        state2 = row2 * self.n_cols + col2
        state = state1 * self.n_blocks + state2
        return state

    def state2rc(self, state):
        '''
        Convert index of state to row, col of player 1 and 2
        :param state: index of state
        :return: row, col of player 1 and row, col of player 2
        '''
        state1 = floor(state / self.n_blocks)
        state2 = state % self.n_blocks
        row1 = floor(state1 / self.n_cols)
        col1 = state1 % self.n_cols
        row2 = floor(state2 / self.n_cols)
        col2 = state2 % self.n_cols
        return row1, col1, row2, col2

    def print_state(self, state):
        row1, col1, row2, col2 = self.state2rc(state)
        sep_line = ''
        for i in range(4 * N_ROW):
            sep_line += '-'
        print(sep_line)
        for i in range(self.n_rows):
            line = ''
            for j in range(self.n_cols):
                if i == row1 == row2 and j == col1 == col2:
                    line += (' O |')
                else:
                    if (i, j) in self.crash_blocks:
                        line += ' C |'
                    elif i == row1 and j == col1:
                        line += (' X |')
                    elif i == row2 and j == col2:
                        line += (' Y |')
                    else:
                        line += ('   |')
            print(line)
            print(sep_line)
