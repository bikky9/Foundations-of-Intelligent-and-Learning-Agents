import numpy as np
import matplotlib.pyplot as plt

horizons = [100, 400, 1600, 6400, 25600, 102400]
instances = ['../instances/i-1.txt', '../instances/i-2.txt', '../instances/i-3.txt']
# algorithms = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling']
algorithms = ['thompson-sampling', 'thompson-sampling-with-hint']
# algorithms = ['ucb', 'epsilon-greedy']
# algorithms = ['epsilon-greedy']
epsilons = [0.0001, 0.02, 0.5]


def get_data3(file):
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    reg_data = {}
    for ins in instances:
        reg_data[ins] = {}
        for alg in algorithms:
            reg_data[ins][alg] = {}
            for eps in epsilons:
                reg_data[ins][alg][eps] = {}

    for d in data:
        ins, alg, seed, eps, hor, reg = d.strip().split(', ')
        hor = int(hor)
        eps = float(eps)
        if hor not in reg_data[ins][alg][eps]:
            reg_data[ins][alg][eps][hor] = 0.0
        reg_data[ins][alg][eps][hor] += (float(reg) / 50)

    return reg_data


def get_data(file):
    f = open(file, 'r')
    data = f.readlines()
    f.close()

    reg_data = {}
    for ins in instances:
        reg_data[ins] = {}
        for alg in algorithms:
            reg_data[ins][alg] = {}

    for d in data:
        ins, alg, seed, eps, hor, reg = d.strip().split(', ')
        hor = int(hor)
        if hor not in reg_data[ins][alg]:
            reg_data[ins][alg][hor] = 0.0
        reg_data[ins][alg][hor] += (float(reg) / 50)

    return reg_data


def compare():
    reg_data1 = get_data('outputDataT1.txt')
    reg_data2 = get_data('outputDataT1_dil.txt')
    reg_data3 = get_data('outputDataT1_char.txt')
    for ins in instances:
        for alg in algorithms:
            print(ins, alg)
            l = reg_data1[ins][alg].values()
            l = [int(x) for x in l]
            print(l)
            l = reg_data2[ins][alg].values()
            l = [int(x) for x in l]
            print(l)
            l = reg_data3[ins][alg].values()
            l = [int(x) for x in l]
            print(l)
    return


def make_plots_T1():
    reg_data1 = get_data('outputDataT2.txt')

    # compare()

    for ins in instances:
        x = np.array(horizons)
        x = np.log(x)
        plt.xlabel('Horizon(log scale)')
        plt.title('Plots for ' + ins)
        for alg in algorithms:
            y = list(reg_data1[ins][alg].values())
            plt.plot(x, y, label=alg)
        plt.legend()
        plt.show()


def make_plots_T3():
    reg_data1 = get_data3('outputDataT3.txt')

    # compare()

    for ins in instances:
        x = np.array(horizons)
        x = np.log(x)
        plt.xlabel('Horizon(log scale)')
        plt.title('Plots for ' + ins)
        alg = 'epsilon-greedy'
        for eps in epsilons:
            y = list(reg_data1[ins][alg][eps].values())
            plt.plot(x, y, label=eps)
        plt.legend()
        plt.show()


make_plots_T1()
