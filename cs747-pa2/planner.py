import argparse
import numpy as np
from collections import defaultdict
import pulp as pl


def valueIteration(numStates, numActions, startState, endStates, T, R, mdptype, discount):
    """
    MDP planning using Value Iteration Algorithm
    :param numStates: number of states in MDP, states are numbered from 0 to numStates-1
    :param numActions: number of Actions in MDP, actions are numbered from 0 to numActions-1
    :param startState: start state not required
    :param endStates: end state for episodic tasks
    :param T: Transition Table - contains probability for each transition
    :param R: Reward Table - contains reward associated with each transition
    :param mdptype: can be either of episodic or continuing
    :param discount: the discount factor of MDP - required for termination of value
    :return: Optimal Value, Optimal Policy
    """
    value = np.zeros(numStates)
    policy = np.zeros(numStates, dtype=int)

    while True:
        valueNext = np.zeros(numStates)
        for state, actionDict in T.items():
            valueNext[state] = float('-inf')
            for action, nextStateDict in actionDict.items():
                tempValue = np.sum(
                    [T[state][action][nextState] * (R[state][action][nextState] + discount * value[nextState])
                     for nextState in nextStateDict])
                if tempValue > valueNext[state]:
                    valueNext[state] = tempValue
                    policy[state] = action
        if np.allclose(value, valueNext, rtol=0, atol=1e-9):
            break
        value = valueNext[:]

    return value, policy


def howardPolicyIteration(numStates, numActions, startState, endStates, T, R, mdptype, discount):
    """
    MDP planning using Howard Policy Iteration Algorithm
    :param numStates: number of states in MDP, states are numbered from 0 to numStates-1
    :param numActions: number of Actions in MDP, actions are numbered from 0 to numActions-1
    :param startState: start state not required
    :param endStates: end state for episodic tasks
    :param T: Transition Table - contains probability for each transition
    :param R: Reward Table - contains reward associated with each transition
    :param mdptype: can be either of episodic or continuing
    :param discount: the discount factor of MDP - required for termination of value
    :return: Optimal Value, Optimal Policy
    """
    policy = np.zeros(numStates, dtype=int)

    while True:
        LpProb = pl.LpProblem('Solver')
        valueVars = [0] * numStates

        for i in range(numStates):
            valueVars[i] = pl.LpVariable("V" + str(i))

        for state in T:
            temp = []
            for nextState in T[state][policy[state]]:
                if nextState in endStates:
                    LpProb += valueVars[nextState] == 0
                temp.append(T[state][policy[state]][nextState] * (R[state][policy[state]][nextState] + discount * valueVars[nextState]))
            LpProb += valueVars[state] == pl.lpSum(temp)

        LpProb.solve(pl.PULP_CBC_CMD(msg=False))  # Solver
        value = np.asarray([pl.value(valueVars[i]) for i in range(numStates)])
        value = [i if i is not None else 0 for i in value]
        nextPolicy = np.zeros(numStates, dtype=int)
        for state, actionDict in T.items():
            temp = float('-inf')
            for action, nextStateDict in actionDict.items():
                tempValue = np.sum(
                    [T[state][action][nextState] * (R[state][action][nextState] + discount * value[nextState])
                     for nextState in nextStateDict])
                if tempValue > temp:
                    temp = tempValue
                    nextPolicy[state] = action
        if np.all(nextPolicy == policy):
            break
        policy = nextPolicy[:]

    return value, policy


def linearProgramming(numStates, numActions, startState, endStates, T, R, mdptype, discount):
    """
    MDP planning using Linear Programming
    :param numStates: number of states in MDP, states are numbered from 0 to numStates-1
    :param numActions: number of Actions in MDP, actions are numbered from 0 to numActions-1
    :param startState: start state not required
    :param endStates: end state for episodic tasks
    :param T: Transition Table - contains probability for each transition
    :param R: Reward Table - contains reward associated with each transition
    :param mdptype: can be either of episodic or continuing
    :param discount: the discount factor of MDP - required for termination of value
    :return: Optimal Value, Optimal Policy
    """
    LpProb = pl.LpProblem('Solver', pl.LpMinimize)
    valueVars = [0] * numStates

    for i in range(numStates):
        valueVars[i] = pl.LpVariable("V" + str(i))

    LpProb += pl.lpSum(valueVars)

    for state in T:
        for action in T[state]:
            temp = []
            for nextState in T[state][action]:
                if nextState in endStates:
                    LpProb += valueVars[nextState] == 0
                temp.append(T[state][action][nextState] * (R[state][action][nextState] + discount * valueVars[nextState]))
            LpProb += valueVars[state] >= pl.lpSum(temp)

    LpProb.solve(pl.PULP_CBC_CMD(msg=False))  # Solver

    value = np.asarray([pl.value(valueVars[i]) for i in range(numStates)])
    policy = np.zeros(numStates, dtype=int)
    for state, actionDict in T.items():
        temp = float('-inf')
        for action, nextStateDict in actionDict.items():
            tempValue = np.sum(
                [T[state][action][nextState] * (R[state][action][nextState] + discount * value[nextState])
                 for nextState in nextStateDict])
            if tempValue > temp:
                temp = tempValue
                policy[state] = action

    return value, policy


if __name__ == '__main__':
    # parser for MDP solver
    parser = argparse.ArgumentParser(description="MDP planner")
    parser.add_argument('--mdp', dest='mdp', help="provide the MDP file")
    parser.add_argument('--algorithm', dest='algo', help="provide which algorithm to use")
    args = parser.parse_args()
    mdp = args.mdp
    algo = args.algo
    # declare data structures for MDP solver
    numStates = None
    numActions = None
    startState = None
    endStates = list()
    T = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))
    R = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))
    mdptype = None
    discount = None
    # initialize above data structures
    mdpFile = open(mdp, 'r')
    for line in mdpFile.readlines():
        parameter = line.split()
        if parameter[0] == 'numStates':
            numStates = int(parameter[1])
        elif parameter[0] == 'numActions':
            numActions = int(parameter[1])
            # T = np.zeros((numStates, numActions, numStates))
            # R = np.zeros((numStates, numActions, numStates))
        elif parameter[0] == 'start':
            startState = int(parameter[1])
        elif parameter[0] == 'end':
            for i in range(1, len(parameter)):
                endStates.append(int(parameter[i]))
        elif parameter[0] == 'transition':
            s = int(parameter[1])
            a = int(parameter[2])
            s_prime = int(parameter[3])
            R[s][a][s_prime] = float(parameter[4])
            T[s][a][s_prime] = float(parameter[5])
            # R[s, a, s_prime] = float(parameter[4])
            # T[s, a, s_prime] = float(parameter[5])
        elif parameter[0] == 'mdptype':
            mdptype = parameter[1]
        elif parameter[0] == 'discount':
            discount = float(parameter[1])
    values = None
    policy = None
    if algo == 'vi':
        values, policy = valueIteration(numStates, numActions, startState, endStates, T, R, mdptype, discount)
    if algo == 'hpi':
        values, policy = howardPolicyIteration(numStates, numActions, startState, endStates, T, R, mdptype, discount)
    if algo == 'lp':
        values, policy = linearProgramming(numStates, numActions, startState, endStates, T, R, mdptype, discount)

    for state, value in enumerate(values):
        print(value, '\t', policy[state])
