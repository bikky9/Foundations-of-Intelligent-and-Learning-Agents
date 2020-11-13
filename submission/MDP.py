import numpy as np
import matplotlib.pyplot as plt
import argparse


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

    def getNextState(self, state, action, stochastic=False):
        row, column = self.cell(state)
        row -= self.wind[column]
        if stochastic and self.wind[column] != 0:
            outcome = np.random.binomial(1, [1 / 3.] * 3)
            if outcome[0] == 1:
                row = row + 1
            elif outcome[1] == 1:
                row = row - 1
        if action == 0:
            row = row - 1
        elif action == 1:
            row = row + 1
        elif action == 2:
            column = column + 1
        elif action == 3:
            column = column - 1
        elif action == 4:
            row = row - 1
            column = column + 1
        elif action == 5:
            row = row + 1
            column = column + 1
        elif action == 6:
            row = row - 1
            column = column - 1
        elif action == 7:
            row = row + 1
            column = column - 1
        if row < 0:
            row = 0
        if row >= self.rows:
            row = self.rows - 1
        if column < 0:
            column = 0
        if column >= self.columns:
            column = self.columns - 1
        nextState = self.currentState(row, column)
        reward = -1
        if nextState == self.endState:
            reward = 0
        return reward, nextState

    def sarsaMean(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        Episodes = [i for i in range(1, 171)]
        meanTimeSteps = []
        for seed in range(10):
            np.random.seed = seed
            timeSteps = self.sarsa(epsilon=epsilon, alpha=alpha, kingMoves=kingMoves, stochastic=stochastic)
            meanTimeSteps.append(timeSteps)
        return Episodes, np.mean(meanTimeSteps, axis=0)

    def sarsa(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        if kingMoves:
            self.numActions = 8
        else:
            self.numActions = 4
        Q = [[0 for i in range(self.numActions)] for j in range(self.numStates)]
        numEpisodes = 170
        timeSteps = []
        timeStep = 0
        for episode in range(1, numEpisodes + 1):
            timeSteps.append(timeStep)
            state = self.startState
            if np.random.binomial(1, epsilon) == 1:
                action = np.random.randint(self.numActions)
            else:
                action = np.argmax(Q[state])
            while state != self.endState:
                reward, nextState = self.getNextState(state, action, stochastic=stochastic)
                timeStep += 1
                if np.random.binomial(1, epsilon) == 1:
                    nextAction = np.random.randint(self.numActions)
                else:
                    nextAction = np.argmax(Q[nextState])
                Q[state][action] += alpha * (reward + Q[nextState][nextAction] - Q[state][action])
                state = nextState
                action = nextAction
        return timeSteps

    def ExpectedsarsaMean(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        Episodes = [i for i in range(1, 171)]
        meanTimeSteps = []
        for seed in range(10):
            np.random.seed = seed
            timeSteps = self.Expectedsarsa(epsilon=epsilon, alpha=alpha, kingMoves=kingMoves, stochastic=stochastic)
            meanTimeSteps.append(timeSteps)
        return Episodes, np.mean(meanTimeSteps, axis=0)

    def Expectedsarsa(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        if kingMoves:
            self.numActions = 8
        else:
            self.numActions = 4
        Q = [[0 for i in range(self.numActions)] for j in range(self.numStates)]
        numEpisodes = 170
        timeSteps = []
        timeStep = 0
        for episode in range(1, numEpisodes + 1):
            timeSteps.append(timeStep)
            state = self.startState
            if np.random.binomial(1, epsilon) == 1:
                action = np.random.randint(self.numActions)
            else:
                action = np.argmax(Q[state])
            while state != self.endState:
                reward, nextState = self.getNextState(state, action, stochastic=stochastic)
                timeStep += 1
                if np.random.binomial(1, epsilon) == 1:
                    nextAction = np.random.randint(self.numActions)
                else:
                    nextAction = np.argmax(Q[nextState])
                Q[state][action] += alpha * (reward + (1.0 / self.numActions) * np.sum(Q[nextState]) - Q[state][action])
                state = nextState
                action = nextAction
        return timeSteps

    def QlearningMean(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        Episodes = [i for i in range(1, 171)]
        meanTimeSteps = []
        for seed in range(10):
            np.random.seed = seed
            timeSteps = self.Qlearning(epsilon=epsilon, alpha=alpha, kingMoves=kingMoves, stochastic=stochastic)
            meanTimeSteps.append(timeSteps)
        return Episodes, np.mean(meanTimeSteps, axis=0)

    def Qlearning(self, epsilon=0.1, alpha=0.5, kingMoves=False, stochastic=False):
        if kingMoves:
            self.numActions = 8
        else:
            self.numActions = 4
        Q = [[0 for i in range(self.numActions)] for j in range(self.numStates)]
        numEpisodes = 170
        timeSteps = []
        timeStep = 0
        for episode in range(1, numEpisodes + 1):
            timeSteps.append(timeStep)
            state = self.startState
            if np.random.binomial(1, epsilon) == 1:
                action = np.random.randint(self.numActions)
            else:
                action = np.argmax(Q[state])
            while state != self.endState:
                reward, nextState = self.getNextState(state, action, stochastic=stochastic)
                timeStep += 1
                if np.random.binomial(1, epsilon) == 1:
                    nextAction = np.random.randint(self.numActions)
                else:
                    nextAction = np.argmax(Q[nextState])
                Q[state][action] += alpha * (reward + np.max(Q[nextState]) - Q[state][action])
                state = nextState
                action = nextAction
        return timeSteps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument('--agent', dest='agt', help="agent can be sarsa, expected_sarsa, q_learning")
    parser.add_argument('--king_moves', dest="king", help="True or false")
    parser.add_argument('--wind_stochasticity', dest="stoc", help="True of false")
    parser.add_argument('--epsilon', dest="epsilon", help="epsilon")
    parser.add_argument('--alpha', dest="alpha", help="alpha")

    args = parser.parse_args()
    agt = args.agt
    king = bool(args.king)
    stoc = bool(args.stoc)
    epsilon = args.epsilon
    alpha = args.alpha
    if epsilon is None:
        epsilon = 0.1
    else:
        epsilon = float(epsilon)
    if alpha is None:
        alpha = 0.5
    else:
        alpha = float(alpha)
    mdp = MDP()
    if agt is None:
        Episodes, timeSteps = mdp.sarsaMean(epsilon, alpha, kingMoves=False, stochastic=False)
        plt.plot(timeSteps, Episodes, label="sarsa")
        Episodes, timeSteps = mdp.sarsaMean(epsilon, alpha, kingMoves=True, stochastic=False)
        plt.plot(timeSteps, Episodes, label="sarsa_with_king_moves")
        Episodes, timeSteps = mdp.sarsaMean(epsilon, alpha, kingMoves=True, stochastic=True)
        plt.plot(timeSteps, Episodes, label="sarsa_with_king_moves_and_stochasticity")
        plt.legend()
        plt.savefig("sarsa_comparison")
        plt.clf()
        Episodes, timeSteps = mdp.sarsaMean(epsilon, alpha)
        plt.plot(timeSteps, Episodes, label="sarsa")
        Episodes, timeSteps = mdp.ExpectedsarsaMean(epsilon, alpha)
        plt.plot(timeSteps, Episodes, label="expected_sarsa")
        Episodes, timeSteps = mdp.QlearningMean(epsilon, alpha)
        plt.plot(timeSteps, Episodes, label="q_learning")
        plt.legend()
        plt.savefig("agent_comparison")
        plt.clf()
    elif agt == "sarsa":
        Episodes, timeSteps = mdp.sarsaMean(epsilon, alpha, kingMoves=king, stochastic=stoc)
        plt.plot(timeSteps, Episodes)
        if king and stoc:
            plt.savefig(agt + "_king" + "_stochastic")
        elif king:
            plt.savefig(agt + "_king")
        else:
            plt.savefig(agt)
        plt.clf()
    elif agt == "expected_sarsa":
        Episodes, timeSteps = mdp.ExpectedsarsaMean(epsilon, alpha, kingMoves=king, stochastic=stoc)
        plt.plot(timeSteps, Episodes)
        plt.savefig(agt)
        plt.clf()
    elif agt == "q_learning":
        Episodes, timeSteps = mdp.QlearningMean(epsilon, alpha, kingMoves=king, stochastic=stoc)
        plt.plot(timeSteps, Episodes)
        plt.savefig(agt)
        plt.clf()
