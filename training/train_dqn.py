#!/usr/bin/env python3
"""
Deep Q-Network training script for the EUAV environment.
"""
import argparse
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from euav_env import EUAVEnv


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        epsilon=0.9,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        batch_size=32,
        discount_factor=0.9,
        max_memory=10000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.memory = deque(maxlen=max_memory)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(256, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        outputs = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (s, a, r, s_next, done) in enumerate(minibatch):
            states[i] = s
            target = self.model.predict(s, verbose=0)[0]
            if done:
                target[a] = r
            else:
                t_next = self.target_model.predict(s_next, verbose=0)[0]
                target[a] = r + self.discount_factor * np.amax(t_next)
            targets[i] = target

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save_weights(path)


def train(args):
    env = EUAVEnv()
    state = env.reset()
    state_size = state.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size,
        action_size,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        discount_factor=args.discount_factor,
    )

    rewards = []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()
        agent.update_target_model()
        rewards.append(total_reward)

        if ep % args.log_interval == 0:
            print(f"Episode {ep}/{args.episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Save final weights
    agent.save(args.model_path)

    # Plot rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.savefig(args.plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN on EUAVEnv')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.9)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='models/dqn_weights.h5')
    parser.add_argument('--plot_path', type=str, default='models/reward_plot.png')
    args = parser.parse_args()

    train(args)
