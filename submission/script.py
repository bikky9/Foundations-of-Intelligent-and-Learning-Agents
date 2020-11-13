import subprocess

cmd = 'python', 'MDP.py', '--agent', 'sarsa'
subprocess.check_output(cmd, universal_newlines=True)
cmd = 'python', 'MDP.py', '--agent', 'sarsa', '--king', 'True'
subprocess.check_output(cmd, universal_newlines=True)
cmd = 'python', 'MDP.py', '--agent', 'sarsa', '--king_moves', 'True', '--wind_stochasticity', 'True'
subprocess.check_output(cmd, universal_newlines=True)
cmd = 'python', 'MDP.py', '--agent', 'expected_sarsa'
subprocess.check_output(cmd, universal_newlines=True)
cmd = 'python', 'MDP.py', '--agent', 'q_learning'
subprocess.check_output(cmd, universal_newlines=True)
