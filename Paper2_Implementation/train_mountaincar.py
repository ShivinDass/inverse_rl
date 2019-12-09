import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('MountainCar-v0')

# model = DQN(MlpPolicy, env, verbose=1)
# model.save("deepq_mc")
#
# del model # remove to demonstrate saving and loading

model = DQN.load("deepq_mc",env=env)
# model.learn(total_timesteps=50000)
# model.save("deepq_mc")

obs = env.reset()
dones=True
while dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    print(dones)
print("done")