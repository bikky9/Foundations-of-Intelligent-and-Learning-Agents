import numpy as np
import matplotlib.pyplot as plt


class MDP:
    def currentState(self, i, j):
        return (i * self.columns) + j

    def cell(self, state):
        return state // self.columns, state % self.columns

    def __init__(self):
        self.rows = 7
        self.columns = 10
        self.startState = self.currentState(3, 0)
        self.numStates = self.rows * self.columns
        self.numActions = 4
        self.endState = self.currentState(3, 7)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        # self.Actions = {'up': 0, 'down': 1, 'right': 2, 'left': 3}

    def getNextState(self, state, action):
        row, column = self.cell(state)
        if action == 0:
            if self.rows > row - 1 - self.wind[column] >= 0:
                row = row - 1 - self.wind[column]
        elif action == 1:
            if self.rows > row + 1 - self.wind[column] >= 0:
                row = row + 1 - self.wind[column]
        elif action == 2:
            if self.columns > column + 1 >= 0:
                if self.rows > row - self.wind[column] >= 0:
                    row = row - self.wind[column]
                column = column + 1
        elif action == 3:
            if self.columns > column - 1 >= 0:
                if self.rows > row - self.wind[column] >= 0:
                    row = row - self.wind[column]
                column = column - 1
        nextState = self.currentState(row, column)
        reward = -1
        return reward, nextState

    def sarsa(self, epsilon=0.1, alpha=0.5):
        Q = [[0] * self.numActions] * self.numStates
        numEpisodes = 10
        timeStep = 0
        for episode in range(numEpisodes):
            print(episode)
            Episodes.append(episode)
            timeSteps.append(timeStep)
            state = self.startState
            if np.random.binomial(1, epsilon) == 1:
                action = np.random.randint(self.numActions)
            else:
                action = np.argmax(Q[state])
            while state != self.endState:
                reward, nextState = self.getNextState(state, action)
                timeStep += 1
                if np.random.binomial(1, epsilon) == 1:
                    nextAction = np.random.randint(self.numActions)
                else:
                    nextAction = np.argmax(Q[nextState])
                Q[state][action] += alpha * (reward + Q[nextState][nextAction] - Q[state][action])
                state = nextState
                action = nextAction


if __name__ == '__main__':
    mdp = MDP()
    fig, ax = plt.subplots()
    timeSteps = []
    Episodes = []
    mdp.sarsa()
    ax.plot(Episodes, timeSteps)
    plt.savefig("fig")
    plt.show()