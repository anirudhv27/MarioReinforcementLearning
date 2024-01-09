# Mario Reinforcement Learning (project in progress)

Implementing reinforcement learning concepts from scratch to play Mario.

Any reinforcement learning (RL) project requires the implementation of an environment, a model architecture, and an agent (which receives observations from the environment and sends an action to the environment)

The agent's "model" is what is used to actually translate this.

Deep Q-Learning: The "Q-function" is a function that determines the expected cumulative reward than an agent receives when taking action a at current state s. The Q network takes in the state s as an input, and assigns a score to each possible action (which ranges from 4 to 16 in the Atari games.)

Training algorithm:
1. Initialize replay memory D to capacity N
2. Initialize action-value function Q with random weights (deep learning model)
3. For each episode from 1 to M
    1. Initialize sequence, preprocessed state
    2. For each time 1 to T:
        1. Select random action with prob epsilon, else select action as action with maximum Q value from neural network
        2. execute the action selected, observe reward and image at that time step
        3. Store current image, action, reward, and next image in replay memory D
        4. Sample random minibatch of transitions from D
        5. y = r if terminal, r + decay rate*previous max Q if non-terminal
        6. gradient descent step on squared error with the Q value






