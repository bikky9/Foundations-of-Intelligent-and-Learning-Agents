import argparse
import numpy as np


def genPath(grid, value_policy):
    gridFile = open(grid, 'r')
    maze = []
    for line in gridFile:
        row = []
        for word in line.split():
            row.append(int(word))
        maze.append(row)
    maze = np.asarray(maze)
    N = maze.shape[0]
    M = maze.shape[1]

    def getPos(t):
        return t[0] * M + t[1]

    def North(i, j):
        if i - 1 < 0:
            return i, j
        return i - 1, j

    def East(i, j):
        if j + 1 > M - 1:
            return i, j
        return i, j + 1

    def South(i, j):
        if i + 1 > N - 1:
            return i, j
        return i + 1, j

    def West(i, j):
        if j - 1 < 0:
            return i, j
        return i, j - 1

    value_policyFile = open(value_policy, 'r')
    policy = []
    for line in value_policyFile:
        value, action = line.split()
        policy.append(int(action))
    index = np.argwhere(maze == 2)[0]
    while maze[index[0], index[1]] != 3:
        if policy[getPos(index)] == 0:
            action = 'N'
            nextIndex = North(index[0], index[1])
        elif policy[getPos(index)] == 1:
            action = 'E'
            nextIndex = East(index[0], index[1])
        elif policy[getPos(index)] == 2:
            action = 'W'
            nextIndex = West(index[0], index[1])
        elif policy[getPos(index)] == 3:
            action = 'S'
            nextIndex = South(index[0], index[1])
        print(action, end=" ")
        index = nextIndex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decodes MDP solution to path")
    parser.add_argument('--grid', dest='grid', help="provide the grid file")
    parser.add_argument('--value_policy', dest='value_policy', help="value policy output file from MDP planner")
    args = parser.parse_args()
    grid = args.grid
    value_policy = args.value_policy
    genPath(grid, value_policy)
