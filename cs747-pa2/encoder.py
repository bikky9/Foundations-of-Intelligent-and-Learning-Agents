import argparse
import numpy as np


def getTransitions(maze, i, j):
    """
    get all possible transitions from a location i,j  from maze
    :param maze: maze object
    :param i: row
    :param j: column
    :return:
    """
    if maze[i, j] == 3 or maze[i, j] == 1:
        return
    reward_free = -1
    reward_end = 1e5
    N = maze.shape[0]
    M = maze.shape[1]

    def getPos(t):
        return t[0] * M + t[1]

    def North(i, j):
        if i - 1 < 0 or maze[i-1, j] == 1:
            return i, j
        return i - 1, j

    def East(i, j):
        if j + 1 > M - 1 or maze[i, j+1] == 1:
            return i, j
        return i, j + 1

    def South(i, j):
        if i + 1 > N - 1 or maze[i+1, j] == 1:
            return i, j
        return i + 1, j

    def West(i, j):
        if j - 1 < 0 or maze[i, j-1] == 1:
            return i, j
        return i, j - 1

    transitions = []

    nextPos = North(i, j)
    if maze[nextPos] != 3:
        print("transition", getPos((i, j)), 0, getPos(nextPos), reward_free, 1)
    elif maze[nextPos] == 3:
        print("transition", getPos((i, j)), 0, getPos(nextPos), reward_end, 1)

    nextPos = East(i, j)
    if maze[nextPos] != 3:
        print("transition", getPos((i, j)), 1, getPos(nextPos), reward_free, 1)
    elif maze[nextPos] == 3:
        print("transition", getPos((i, j)), 1, getPos(nextPos), reward_end, 1)

    nextPos = West(i, j)
    if maze[nextPos] != 3:
        print("transition", getPos((i, j)), 2, getPos(nextPos), reward_free, 1)
    elif maze[nextPos] == 3:
        print("transition", getPos((i, j)), 2, getPos(nextPos), reward_end, 1)

    nextPos = South(i, j)
    if maze[nextPos] != 3:
        print("transition", getPos((i, j)), 3, getPos(nextPos), reward_free, 1)
    elif maze[nextPos] == 3:
        print("transition", getPos((i, j)), 3, getPos(nextPos), reward_end, 1)


def genMDP(gridFile):
    """
    encodes given grid as an MDP
    :param gridFile: path to the grid file
    :return: void, prints the MDP to standard Output
    """
    gridFile = open(gridFile, 'r')
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

    print("numStates", N * M)
    print("numActions", 4)
    print("start", getPos(np.argwhere(maze == 2)[0]))
    print("end", getPos(np.argwhere(maze == 3)[0]))
    for i in range(N):
        for j in range(M):
            getTransitions(maze=maze, i=i, j=j)
    print("mdptype", "episodic")
    print("discount", 0.9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="encodes grid to an MDP")
    parser.add_argument('--grid', dest='grid', help="provide the grid file")
    args = parser.parse_args()
    grid = args.grid
    genMDP(grid)
