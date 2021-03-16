from collections import deque
import gym 
import random
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import numpy as np 
"""
Framework described in https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning
for some reason, runs better than torch version, albeit around 20-30x slower, coverges properly.
"""
tensorboard = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=True, update_freq='epoch', profile_batch=0,
    embeddings_freq=0, embeddings_metadata=None
)
class DQN:
    def __init__(self,env):
    """
    Sets the environment and other hyperparameters
    """
        self.env = env
        self.memory = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.target_model =  self.create_model()
    def create_model(self):  
    """
    keras.models.sequential is used for architecture, lower dimensions is better for CartPole
    """
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(12,input_dim=state_shape[0],activation = 'relu'))    
        model.add(Dense(24,activation = 'relu'))    
        model.add(Dense(12,activation = 'relu'))    
        model.add(Dense(self.env.action_space.n))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        model.summary()
        return model    
    def remember(self,state,action,reward,new_state,done):
    """
    append memory to deque
    """
        self.memory.append([state,action,reward,new_state,done])
    def replay(self):
    """
    experience replay
    """
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory,batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            """
            target values
            """
            target = self.target_model.predict(state)
            if done:
                """
                done states have no future
                """
                target[0][action] = reward
            else:
                """
                Q'(s',a') = max of target_model (next state)  * discount + reward
                """
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
        """
        loss/backward step both incorporated in .fit()
        """
        self.model.fit(state,target,epochs=1,verbose=0, callbacks=[tensorboard])
    def target_train(self):
        self.target_model.set_weights( self.model.get_weights())
    def act(self, state):
    """
    epsilon-greedy
    """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min,self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        z = np.argmax(self.model.predict(state)[0])
        return z
env = gym.make("CartPole-v0")
gamma = 0.9
epsilon = 0.95
agent = DQN(env)
for i in range(1001):
    current_state = env.reset().reshape(1,4)
    r = 0
    for step in range(500):
        action = agent.act(current_state)
        new_state,reward,done,_ = env.step(action)
        new_state = new_state.reshape(1,4)
        agent.remember(current_state,action,reward,new_state,done)
        agent.replay()
        agent.target_train()
        current_state = new_state
        r += reward
        if done:
            break
    print(i,r)
    if i%10 == 0:
        print(f"saving model at episode {i}")
        agent.model.save('ckpt')