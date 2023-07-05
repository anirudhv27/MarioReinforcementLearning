import gymnasium as gym
from dqn import DeepQNet, ReplayBuffer

env = gym.make("Breakout-v4", render_mode="human")

INPUT_SHAPE = (4, 84, 84)
OUT_ACTIONS = 3
M = 10_000_000
N = int(1e5)
BATCH_SIZE = 32

buffer = ReplayBuffer(N, batch_size=BATCH_SIZE)
q = DeepQNet(input_shape=INPUT_SHAPE, out_actions=OUT_ACTIONS)

observation, info = env.reset(seed=42)
for i in range(M):
    img = env.render()
    print(img.shape)

    action = env.action_space.sample()  # this is where you would insert your policy

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
