import random
import gym
import numpy as np
#classic hill climb algorithm, not very effective
class MyAgent:
    def __init__(self):
    #initialize to random weights w/ noise
        self.env = gym.make('CartPole-v0')
        self.state_space = self.env.observation_space.shape
        self.action_size = self.env.action_space
        self.weights = (1e-4)*np.random.rand(4,2)
        self.best = -np.Inf
        self.best_weights = np.copy(self.weights)
        self.noise_scale = 1e-2
    def action(self,state):
        p = np.dot(state,self.weights)
        action = np.argmax(p)
        return self.action_size.sample()
    def update(self,reward):
    #go in direction of better reward
        if reward >= self.best:
            self.best = reward
            self.best_weights = np.copy(self.weights) 
            self.noise_scale /= 2
        else:
            self.noise_scale *= 2
        self.weights = self.best_weights + self.noise_scale * np.random.rand(4, 2)
    def run(self):
    #main loop achieves ~70 average in 1000 steps
        agent = MyAgent()
        observation = agent.env.reset()
        rewards = 0
        while True:
            agent.env.render()
            observation, reward, done, info =  agent.env.step(agent.action(observation))
            rewards += reward
            agent.update(reward)
            if done:
                print(rewards)
                rewards = 0
                agent.env.reset()
agent = MyAgent()
agent.run()