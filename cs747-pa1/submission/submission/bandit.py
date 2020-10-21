import numpy as np
import argparse


def EpsilonGreedy(means, ep, hz, hzs):
    """
    simulates bandit using epsilon greedy algorithm
    :param means: list of true means of the arms
    :param ep: epsilon parameter of the algorithm - explore exploit tradeoff
    :param hz: horizon, number of time steps in simulation
    :param hzs: intermediate time steps to save
    :return: Regret after running epsilon greedy "horizon" number of times
    """
    n = len(means)
    cumulativeReward = 0
    empiricalMeans = [0] * n
    armPulls = [0] * n
    cumulativeRegretForHorizons = []

    # running epsilon greedy "horizon" number of times
    for t in range(hz):
        # with epsilon probability choose arm at random
        if np.random.binomial(1, ep) == 1:
            arm = np.random.randint(n)
        else:
            arm = np.argmax(empiricalMeans)

        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        empiricalMeans[arm] = (empiricalMeans[arm] * armPulls[arm] + reward) / (armPulls[arm] + 1)
        armPulls[arm] += 1

        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    maximumExpectedCumulativeReward = max(means) * hz
    cumulativeRegret = maximumExpectedCumulativeReward - cumulativeReward

    return cumulativeRegret, cumulativeRegretForHorizons


def ucb(means, hz, hzs):
    """
    simulates bandit using ucb algorithm
    :param means: list of true means of the arms
    :param hz: horizon, number of times steps in simulation
    :param hzs: intermediate time steps to save
    :return: Regret after running ucb "horizon" number of times
    """
    n = len(means)
    cumulativeReward = 0
    empiricalMeans = [0] * n
    armPulls = [0] * n
    cumulativeRegretForHorizons = []

    # running ucb "horizon" number of times
    for t in range(n):
        arm = t
        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        empiricalMeans[arm] = (empiricalMeans[arm] * armPulls[arm] + reward) / (armPulls[arm] + 1)
        armPulls[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    for t in range(n, hz):
        ucbValues = [(empiricalMeans[arm] + np.sqrt(2 * np.log(t) / armPulls[arm])) for arm in range(n)]
        arm = np.argmax(ucbValues)
        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        empiricalMeans[arm] = (empiricalMeans[arm] * armPulls[arm] + reward) / (armPulls[arm] + 1)
        armPulls[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    maximumExpectedCumulativeReward = max(means) * hz
    cumulativeRegret = maximumExpectedCumulativeReward - cumulativeReward

    return cumulativeRegret, cumulativeRegretForHorizons


def KL(x, y):
    """
    Return KL divergence between distributions x and y
    """
    if x == 0:
        return np.log(1 / (1 - y))
    elif x == 1:
        return np.log(1 / y)
    else:
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def KLucbvalue(empiricalMean, armPulls, t):
    """
    ucb-kl of arm at time t
    :param arm: index of arm
    :param empiricalMean: empirical mean of arm
    :param armPulls: count of arm pulls
    :param t: time t
    :return: ucb-kl
    """
    low = empiricalMean
    high = 1
    epsilon = 1e-3
    check = lambda q: armPulls * KL(empiricalMean, q) <= np.log(t)
    diff = float('inf')
    q = (low + high) / 2
    while diff > epsilon:
        if check(q):
            low = q
        else:
            high = q
        diff = abs(q - (low + high) / 2)
        q = (low + high) / 2

    return q


def KLucb(means, hz, hzs):
    """
    simulates bandit using KL-UCB algorithm
    :param means: list of true means of the arms
    :param hz: horizon, number of times steps in simulation
    :param hzs: intermediate time steps to save
    :return: Regret after running ucb "horizon" number of times
    """
    n = len(means)
    cumulativeReward = 0
    empiricalMeans = [0] * n
    armPulls = [0] * n
    cumulativeRegretForHorizons = []

    # running kl-ucb "horizon" number of times
    for t in range(n):
        arm = t
        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        empiricalMeans[arm] = (empiricalMeans[arm] * armPulls[arm] + reward) / (armPulls[arm] + 1)
        armPulls[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)
    for t in range(n, hz):
        ucbValues = [KLucbvalue(empiricalMeans[arm], armPulls[arm], t) for arm in range(n)]
        arm = np.argmax(ucbValues)
        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        empiricalMeans[arm] = (empiricalMeans[arm] * armPulls[arm] + reward) / (armPulls[arm] + 1)
        armPulls[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    maximumExpectedCumulativeReward = max(means) * hz
    cumulativeRegret = maximumExpectedCumulativeReward - cumulativeReward

    return cumulativeRegret, cumulativeRegretForHorizons


def thompsonSampling(means, hz, hzs):
    """
    simulates bandit using KL-UCB algorithm
    :param means: list of true means of the arms
    :param hz: horizon, number of times steps in simulation
    :param hzs: intermediate time steps to save
    :return: Regret after running ucb "horizon" number of times
    """
    n = len(means)
    cumulativeReward = 0
    armSuccesses = [0] * n
    armFailures = [0] * n
    cumulativeRegretForHorizons = []

    # running thompson-sampling "horizon" number of times
    for t in range(hz):
        betaSample = [np.random.beta(armSuccesses[arm] + 1, armFailures[arm] + 1) for arm in range(n)]
        arm = np.argmax(betaSample)
        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        if reward == 1:
            armSuccesses[arm] += 1
        else:
            armFailures[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    maximumExpectedCumulativeReward = max(means) * hz
    cumulativeRegret = maximumExpectedCumulativeReward - cumulativeReward

    return cumulativeRegret, cumulativeRegretForHorizons


def thompsonSamplingHint(means, hz, hzs, hint):
    """
    simulates bandit using KL-UCB algorithm
    :param means: list of true means of the arms
    :param hz: horizon, number of times steps in simulation
    :param hzs: intermediate time steps to save
    :param hint: list of true means in sorted order
    :return: Regret after running ucb "horizon" number of times
    """
    n = len(means)
    cumulativeReward = 0
    armSuccesses = [0] * n
    armFailures = [0] * n
    cumulativeRegretForHorizons = []

    # running thompson-sampling-with-hint "horizon" number of times
    for t in range(hz):
        betaDistributionMean = [(armSuccesses[arm] + 1) / (armSuccesses[arm] + armFailures[arm] + 2) for arm in
                                range(n)]
        betaDistributionVar = 0.01
        highestMean = hint[-1]
        check = [(betaDistributionMean[arm] - betaDistributionVar < highestMean < betaDistributionMean[arm] +
                  betaDistributionVar) for arm in range(n)]

        if np.sum(check) == 1:
            arm = np.argmax(check)
        else:
            betaSample = [np.random.beta(armSuccesses[arm] + 1, armFailures[arm] + 1) for arm in range(n)]
            arm = np.argmax(betaSample)

        reward = np.random.binomial(1, means[arm])
        cumulativeReward += reward
        if reward == 1:
            armSuccesses[arm] += 1
        else:
            armFailures[arm] += 1
        if t + 1 in hzs:
            cumulativeRegretForHorizons.append(max(means) * (t + 1) - cumulativeReward)

    maximumExpectedCumulativeReward = max(means) * hz
    cumulativeRegret = maximumExpectedCumulativeReward - cumulativeReward

    return cumulativeRegret, cumulativeRegretForHorizons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulates a bandit behaviour")
    parser.add_argument('--instance', dest='ins', help="bandit instance to simulate")
    parser.add_argument('--algorithm', dest='al', help="Algorithm to simulate eg. epsilon-greedy, UCB")
    parser.add_argument('--randomSeed', dest='rs', type=int, help="non-negative integer")
    parser.add_argument('--epsilon', dest='ep', type=float, help="epsilon")
    parser.add_argument('--horizon', dest='hz', type=int, help="provide horizon to stop")
    parser.add_argument('--horizons', dest='hzs', required=False, type=str, help="provide list of "
                                                                                 "horizons")

    args = parser.parse_args()
    ins = args.ins
    al = args.al
    rs = args.rs
    ep = args.ep
    hz = args.hz
    hzs = []
    if args.hzs is not None:
        hzs = [int(i) for i in args.hzs.split()]

    instanceFile = open(ins, 'r')
    lines = instanceFile.readlines()
    means = []
    for line in lines:
        means.append(float(line))

    np.random.seed(rs)
    REG = None
    REGS = None
    if al == "epsilon-greedy":
        REG, REGS = EpsilonGreedy(means, ep, hz, hzs)
    elif al == "ucb":
        REG, REGS = ucb(means, hz, hzs)
    elif al == "kl-ucb":
        REG, REGS = KLucb(means, hz, hzs)
    elif al == "thompson-sampling":
        REG, REGS = thompsonSampling(means, hz, hzs)
    elif al == "thompson-sampling-with-hint":
        hint = np.sort(means)
        REG, REGS = thompsonSamplingHint(means, hz, hzs, hint)
    if len(hzs) == 0:
        print(ins, al, rs, ep, hz, REG, sep=", ")
    else:
        for horizon in range(len(hzs)):
            print(ins, al, rs, ep, hzs[horizon], REGS[horizon], sep=", ")
