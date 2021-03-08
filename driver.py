from collections import deque
import gym 
import random
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import numpy as np 
model = load_model('ckpt')
env = gym.make("CartPole-v0")
state = env.reset().reshape(1,4)
r = 0
while True:
    env.render()
    action = np.argmax(model.predict(state))
    state,reward,done,_ = env.step(action)
    state = state.reshape(1,4)
    r += reward
    if done:
        env.reset()
        print(r)
        r = 0

