import gym
import ArmEnv
import time
import matplotlib.pyplot as plt

env = gym.make('PointToPoint-v0',gui=True,mode='T')
env.reset()

rews = []

print(env.observation_space)
print(env.action_space)

for i in range(100):
    action = env.action_space.sample()
    _, rew, _, _ = env.step(action)
    rews.append(rew)
    time.sleep(0.25)
    print(action,rew)

