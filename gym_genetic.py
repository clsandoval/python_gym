import random
import gym
import numpy as np
import keras
class candidate:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.weights = np.random.rand(4,2)
        self.observation = self.env.reset()
    def action(self,state):
        a = np.dot(state,self.weights)
        return np.argmax(a)
    def worth(self):
        done = False
        while not done:
            self.env.render()
            self.observation, reward, done, info = self.env.step(self.action(self.observation))
class population:
    def __init__(self):
        self.population = [candidate() for i in range(4)]
p = population()
p.population[0].run()



    

