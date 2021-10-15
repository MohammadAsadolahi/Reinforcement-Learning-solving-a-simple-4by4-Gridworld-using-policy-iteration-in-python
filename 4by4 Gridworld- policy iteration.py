# Reinforcement Learning solving a simple 4*4 Gridworld using policy iteration method
# WRITTEN BY MOHAMMAD ASADOLAHI
# Mohammad.E.Asadolahi@gmail.com
# https://github.com/mohammadAsadolahi

import numpy as np

class GridWorld:
    def __init__(self):
        # S O O O
        # O O O *
        # O * O O
        # O * 0 T
        self.currentState = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'D', 'R'),
            (0, 2): ('L', 'D', 'R'),
            (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'),
            (1, 1): ('U', 'L', 'D', 'R'),
            (1, 2): ('U', 'L', 'D', 'R'),
            (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'),
            (2, 1): ('U', 'L', 'D', 'R'),
            (2, 2): ('U', 'L', 'D', 'R'),
            (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'),
            (3, 1): ('U', 'L', 'R'),
            (3, 2): ('U', 'L', 'R')
        }
        self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -10}
        self.explored = 0
        self.exploited = 0

    def getRandomPolicy(self):
        policy = {}
        for state in self.actions:
            policy[state] = np.random.choice(self.actions[state])
        return policy

    def getCurrentState(self):
        if not self.currentState:
            self.currentState = (0, 0)
        return self.currentState

    def is_terminal(self, s):
        return s not in self.actions

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def greedyChoose(self, state, values):
        actions = self.actions[state]
        stateValues = []
        for item in actions:
            i, j = zip(state)
            row = int(i[0])
            column = int(j[0])
            if item == 'U':
                row -= 1
            elif item == 'D':
                row += 1
            elif item == 'L':
                column -= 1
            elif item == 'R':
                column += 1
            if (row, column) in values:
                stateValues.append(values[(row, column)])
        return actions[np.argmax(stateValues)]

    def getActionReward(self, state, action):
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return self.rewards[(row, column)]
        else:
            return 0

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return (row, column),self.rewards[(row, column)]
        return (row, column), 0

    def printVaues(self,values):
        line = ""
        counter = 0
        for item in values:
            line += f" | {values[item]} | "
            counter += 1
            if counter > 3:
                print(line)
                print("--------------------------------")
                counter = 0
                line = ""
        print(line)
        print("----------------------------")
        
    def printPolicy(self, policy):
        line = ""
        counter = 0
        for item in policy:
            line += f" | {policy[item]} | "
            counter += 1
            if counter > 3:
                print(line)
                print("----------------------------")
                counter = 0
                line = ""
        print(line)
        print("----------------------------")
        
        
 
enviroment = GridWorld()
policy = enviroment.getRandomPolicy()
#example optimal policy = {(0, 0): 'R', (0, 1): 'R', (0, 2): 'D', (0, 3): 'D', (1, 0): 'R', (1, 1): 'D', (1, 2): 'D', (1, 3): 'D',
#           (2, 0): 'R', (2, 1): 'D', (2, 2): 'R', (2, 3): 'D', (3, 0): 'R', (3, 1): 'R', (3, 2): 'R'}

enviroment.printPolicy(policy)
state = enviroment.getCurrentState()
values = {}

for j in range(10):
    values.clear()
    for state in policy:
        values[state] = 0
    values[(3, 3)] = 5
    for i in range(10):
        for item in enviroment.actions.keys():
            nextState, reward = enviroment.move(item, policy, exploreRate=0.2)
            values[item] = reward + 0.1 * values[nextState]
        for item in policy:
             policy[item] = enviroment.greedyChoose(item, values)
    if (j%1)==0:
      print(f"\n\n\n step:{j}")
      enviroment.printVaues(values)
      enviroment.printPolicy(policy)
print(f"exploited:{enviroment.exploited}  explored:{enviroment.explored}")
