import gym
import ArmEnv
import time

env = gym.make('PointToPoint-v0')
env.reset()

print(env.observation_space)
print(env.action_space)

for i in range(100):
    action = env.action_space.sample()
    _, rew, _, _ = env.step(action)
    time.sleep(0.5)
    print(rew)

