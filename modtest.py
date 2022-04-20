import gym
import ArmEnv
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC

env = gym.make('PointToPoint-v0',gui=True,record=False)
model = SAC('MlpPolicy',env,verbose=1,device='cuda')
model.load('model2.zip')
obs = env.reset()
print('Observation:',obs)
dones = False
rews = []

count = 0
while(True):
    count += 1
    action, _states = model.predict(obs,deterministic=False)
    print(action)
    obs, rewards, dones, info = env.step(action)
    rews.append(rewards)
    if(dones):
        break
    #print(count,end='\r')

env.close()
plt.plot(rews)
plt.savefig('rews.png')
