import os
from multiprocessing import Process

# For T1
banditInstances = ['../instances/i-1.txt', '../instances/i-2.txt', '../instances/i-3.txt']
algorithms1 = ['epsilon-greedy', 'ucb', "thompson-sampling", 'kl-ucb']
algorithms2 = ["thompson-sampling", "thompson-sampling-with-hint"]
algorithms3 = ["epsilon-greedy", "ucb"]
horizon = '"100 400 1600 6400 25600 102400"'
ep = 0.02
# os.system("rm outputDataT1.txt")
# os.system("rm outputDataT2.txt")
os.system("rm outputDataT3.txt")
# os.system("rm outputDataT4.txt")


def f(ins, al, min_seed, max_seed):
    for rs in range(min_seed, max_seed):
        print("random seed: ", rs)
        command = "python bandit.py --instance {ins} --algorithm {al} --randomSeed {rs} --epsilon {ep} " \
                  "--horizon {hz} --horizons {hzs}".format(ins=ins, al=al, rs=rs, ep=ep, hz=102400, hzs=horizon)
        os.system(command + ">> outputDataT1.txt")


def g(ins, al, min_seed, max_seed):
    for rs in range(min_seed, max_seed):
        print("random seed: ", rs)
        command = "python bandit.py --instance {ins} --algorithm {al} --randomSeed {rs} --epsilon {ep} " \
                  "--horizon {hz} --horizons {hzs}".format(ins=ins, al=al, rs=rs, ep=ep, hz=102400, hzs=horizon)
        os.system(command + ">> outputDataT2.txt")


def h(ins, al, min_seed, max_seed, ep):
    for rs in range(min_seed, max_seed):
        print("random seed: ", rs)
        command = "python bandit.py --instance {ins} --algorithm {al} --randomSeed {rs} --epsilon {ep} " \
                  "--horizon {hz} --horizons {hzs}".format(ins=ins, al=al, rs=rs, ep=ep, hz=102400, hzs=horizon)
        os.system(command + ">> outputDataT3.txt")


# for ins in banditInstances:
#     print("bandit instance: ", ins)
#     for al in algorithms1:
#         print("algorithm: ", al)
#         P1 = Process(target=f, args=(ins, al, 0, 6))
#         P2 = Process(target=f, args=(ins, al, 6, 12))
#         P3 = Process(target=f, args=(ins, al, 12, 18))
#         P4 = Process(target=f, args=(ins, al, 18, 25))
#         P5 = Process(target=f, args=(ins, al, 25, 31))
#         P6 = Process(target=f, args=(ins, al, 31, 37))
#         P7 = Process(target=f, args=(ins, al, 37, 43))
#         P8 = Process(target=f, args=(ins, al, 43, 50))
#
#         P1.start()
#         P2.start()
#         P3.start()
#         P4.start()
#         P5.start()
#         P6.start()
#         P7.start()
#         P8.start()
#
#         P1.join()
#         P2.join()
#         P3.join()
#         P4.join()
#
#         P5.join()
#         P6.join()
#         P7.join()
#         P8.join()

# for ins in banditInstances:
#     print("bandit instance: ", ins)
#     for al in algorithms2:
#         print("algorithm: ", al)
#         P1 = Process(target=g, args=(ins, al, 0, 6))
#         P2 = Process(target=g, args=(ins, al, 6, 12))
#         P3 = Process(target=g, args=(ins, al, 12, 18))
#         P4 = Process(target=g, args=(ins, al, 18, 25))
#         P5 = Process(target=g, args=(ins, al, 25, 31))
#         P6 = Process(target=g, args=(ins, al, 31, 37))
#         P7 = Process(target=g, args=(ins, al, 37, 43))
#         P8 = Process(target=g, args=(ins, al, 43, 50))
#
#         P1.start()
#         P2.start()
#         P3.start()
#         P4.start()
#         P5.start()
#         P6.start()
#         P7.start()
#         P8.start()
#
#         P1.join()
#         P2.join()
#         P3.join()
#         P4.join()
#
#         P5.join()
#         P6.join()
#         P7.join()
#         P8.join()

for ins in banditInstances:
    print("bandit instance: ", ins)
    al = "epsilon-greedy"
    epsilons = [0.0001, 0.02, 0.5]
    for ep in epsilons:
        print("algorithm: ", al)
        P1 = Process(target=h, args=(ins, al, 0, 6, ep))
        P2 = Process(target=h, args=(ins, al, 6, 12, ep))
        P3 = Process(target=h, args=(ins, al, 12, 18, ep))
        P4 = Process(target=h, args=(ins, al, 18, 25, ep))
        P5 = Process(target=h, args=(ins, al, 25, 31, ep))
        P6 = Process(target=h, args=(ins, al, 31, 37, ep))
        P7 = Process(target=h, args=(ins, al, 37, 43, ep))
        P8 = Process(target=h, args=(ins, al, 43, 50, ep))

        P1.start()
        P2.start()
        P3.start()
        P4.start()
        P5.start()
        P6.start()
        P7.start()
        P8.start()

        P1.join()
        P2.join()
        P3.join()
        P4.join()

        P5.join()
        P6.join()
        P7.join()
        P8.join()

