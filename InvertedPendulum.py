import numpy as np
import matplotlib.pyplot as plt

class InvertedPendulumEnv:
    def __init__(self):
        # Constants from the assignment
        self.g = -9.8  # Gravity (m/s^2)
        self.mc = 1.0  # Mass of the cart (kg)
        self.m = 0.1  # Mass of the pole (kg)
        self.l = 0.5  # Half-length of the pole (m)
        self.force_mag = 10  # Magnitude of the force applied to the cart (Newtons)

        # Time parameters
        self.dt = 0.02  # Time interval (s)

        # State space: [cart position, cart velocity, pole angle, pole angular velocity]
        # Action space: [-force_mag, force_mag] (left or right force)
        self.state = np.array([0, 0, 0, 0])  # Initial state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action * self.force_mag  # Apply either +10 or -10 Newtons

        # Implementing the dynamics
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        total_mass = self.m + self.mc
        temp = (force + self.m * self.l * theta_dot ** 2 * sin_theta) / total_mass
        thetaacc = (self.g * sin_theta - cos_theta * temp) / (self.l * (4/3 - self.m * cos_theta ** 2 / total_mass))
        xacc = temp - self.m * self.l * thetaacc * cos_theta / total_mass

        # Update the state - Euler's method
        x += self.dt * x_dot
        x_dot += self.dt * xacc
        theta += self.dt * theta_dot
        theta_dot += self.dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])

        # Check if the state is terminal
        done = self.is_terminal_state(self.state)

        # Calculate reward
        if done:
            # Large negative reward for ending the episode
            reward = -100
        else:
            # Negative reward that increases as the pole angle gets closer to 12 degrees
            angle_degrees = np.abs(np.degrees(theta))  # Convert pole angle to degrees
            reward = max(1 - (angle_degrees / 12.0), 0)  # Decrease reward as angle approaches 12 degrees

        return self.state, reward, done

    def is_terminal_state(self, state):
        cart_pos, _, pole_angle, _ = state
        if abs(cart_pos) > 2.4 or abs(pole_angle) > (12 * np.pi / 180):
            return True
        return False

    def reset(self):
        self.state = np.array([0, 0, 0, 0])
        return self.state

def discretize_state(state, num_bins=3):
    upper_bounds = [2.4, 1, 12 * np.pi / 180, 1]
    lower_bounds = [-2.4, -1, -12 * np.pi / 180, -1]

    discretized_state = []
    for i in range(len(state)):
        # Scale state[i] to be within [0, num_bins-1]
        scaling = (state[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])
        discretized_index = int(round(scaling * (num_bins - 1)))

        # Ensure the discretized index is within bounds
        discretized_index = max(0, min(discretized_index, num_bins - 1))

        discretized_state.append(discretized_index)

    return tuple(discretized_state)

def q_learning(env, episodes=1000, alpha=0.01, gamma=0.99, epsilon=0.01, num_bins=3):
    num_actions = 2  # -10 or +10 force
    q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, num_actions))

    for episode in range(episodes):
        state = discretize_state(env.reset())
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice([-env.force_mag, env.force_mag])
            else:
                action = -env.force_mag if np.argmax(q_table[state]) == 0 else env.force_mag

            next_state, reward, done = env.step(action)
            next_state = discretize_state(next_state)

            # Calculate action index for updating Q-table
            action_index = 0 if action == -env.force_mag else 1

            # Update Q-table
            q_table[state][action_index] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action_index])
            state = next_state

    return q_table

def simulate(env, q_table, num_bins=3, max_steps=15500):
    state = env.reset()
    done = False
    steps = 0

    # For plotting
    x_positions = []
    angles = []

    while not done and steps < max_steps:
        discretized_state = discretize_state(state, num_bins)
        action_index = np.argmax(q_table[discretized_state])
        action = -env.force_mag if action_index == 0 else env.force_mag

        state, _, done = env.step(action)

        # For plotting
        x_positions.append(state[0])
        angles.append(state[2])
        steps += 1

    return x_positions, angles

# Create the environment and train using Q-learning
env = InvertedPendulumEnv()
q_table = q_learning(env)

# Simulate the environment using the learned Q-table
x_positions, angles = simulate(env, q_table)

# Convert angles from radians to degrees for plotting
angles_degrees = np.degrees(angles)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(x_positions)
plt.title('Cart Position Over Time')
plt.ylabel('Position (meters)')

plt.subplot(2, 1, 2)
plt.plot(angles_degrees)
plt.title('Pole Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time Steps')

plt.tight_layout()
plt.show()
