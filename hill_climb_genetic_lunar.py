import random
import gym
import numpy as np
import keras
class candidate:
#member of populous
    def __init__(self, weight):
        self.env = gym.make('LunarLander-v2')
        self.weights = weight
        self.observation = self.env.reset()
    def action(self,state):
        a = np.dot(state,self.weights)
        return np.argmax(a)
    def worth(self):
    #utility function is reward
        done = False
        r = 0
        while not done:
            self.observation, reward, done, info = self.env.step(self.action(self.observation))
            r += reward
        self.env.reset()
        return r
class population:
    #population made up of candidates
    def __init__(self):
    #initialize to random pop
        self.population = [candidate(np.random.rand(8,4)) for i in range(10)]
    def mutate(self,child):
    #epsilon-greedy policy
        child.weights = child.weights+ np.random.normal(0,np.std(child.weights),(8,4))
    def crossover(self,parents):
    #cross the best performers of each generation
        child_weights = np.random.rand(8,4)
        for i in range(len(parents)):
            for j in range(len(parents)):
                child_weights[:4][:2] = parents[i].weights[:4][:2]
                child_weights[4:][2:] = parents[j].weights[4:][2:]
                child = candidate(child_weights)
                if len(self.population)> 20:
                #drop the oldest candidate, possible to make this worst performers instead
                    self.population = self.population[1:]
                prob = np.random.randint(10)
                if (prob == 1):
                    self.mutate(child)
                self.population.append(child)
#main training loop
#trains well, solves the game in ~400 steps, does not work for more complex environments
p = population()
for i in range(2000):
    worths = np.array([i.worth() for i in p.population])
    if i %100 == 0: print(worths.sum(0)/ len(worths))
    max_2 = np.argpartition(worths,-6)[-6:]
    parents = []
    for i in max_2:
        parents.append(p.population[i])
    p.crossover(parents)
best_agent = np.argmax(worths)
agent = p.population[best_agent]
mean = 0
for i in range(20):
    done = False
    r = 0
    while not done:
        agent.observation, reward, done, info = agent.env.step(agent.action(agent.observation))
        r += reward
    agent.env.reset()
    print(r)
    mean += r
print(f"Mean: {mean/100}")
while True:
    agent.env.render()
    agent.observation, reward, done, info = agent.env.step(agent.action(agent.observation))
    if done:
        agent.env.reset()


    


    

