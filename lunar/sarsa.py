import gym as gym
import torch as torch

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Global Parameters
GAMMA = 0.9  # discount factor
Alpha = 1.0  # this like learning-rate
Epsilon = 1.0
EpsilonDecay = 0.9999
NumEvalGames = 1


###############################################
# Env Wrapper
###############################################
class QuantizedWrapper(gym.Wrapper):
    def __init__(self, env, action_intervals=3, space_digits=3):
        super(QuantizedWrapper, self).__init__(env)
        assert action_intervals > 1 and space_digits >= 1
        self.space_digits = space_digits

        self.num_to_action = []
        for a1 in np.linspace(-1, 1, action_intervals):
            for a2 in np.linspace(-1, 1, action_intervals):
                self.num_to_action.append(np.array([a1, a2]))

    def action_converter(self, action):
        return self.num_to_action[action]

    def observation_quantizer(self, observation):
        return tuple(np.round(observation, self.space_digits))

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action_converter(action))
        return self.observation_quantizer(observation), reward, done, info

    def reset(self):
        return self.observation_quantizer(self.env.reset())


def sampling(x):
    index = 0
    r = random.random()
    s = x[index]
    while s <= r:
        index += 1
        s += x[index]
    return index


###################################################
#                  Agent
###################################################
class Agent:
    def __init__(self, env):
        assert isinstance(env, gym.Env)
        self.env = env
        self.q_table = {}
        self.epsilon = Epsilon

    # choose an action according to policy (epsilon-greedy)
    def next_action(self, state):
        value_per_action = self.q_table.get(state, None)
        if value_per_action is None:
            value_per_action = np.zeros(len(self.env.num_to_action))
            self.q_table[state] = value_per_action

        # exploitation:
        if random.random() > self.epsilon:
            action = self.greedy_policy(state)
        # exploration:
        else:
            value_per_action = np.exp(value_per_action)
            value_per_action /= value_per_action.sum()
            action = sampling(value_per_action)
        return action

    def greedy_policy(self, state):
        value_per_action = self.q_table.get(state, None)
        if value_per_action is None:
            value_per_action = np.zeros(len(self.env.num_to_action))
            action = random.randint(0, len(value_per_action) - 1)
            return action

        value_per_action = np.exp(value_per_action)
        value_per_action /= value_per_action.sum()
        action = sampling(value_per_action)
        return action

    def play_a_game(self, animation=False):
        cum_reward = 0.
        state = env.reset()
        done = False
        while not done:
            if animation:
                env.render(mode="rgb_array")
            action = self.greedy_policy(state)
            next_state, reward, done, _ = env.step(action)
            cum_reward += reward
            state = next_state
        return cum_reward

    def eval_agent(self, num_games=100):
        rewards = []
        for _ in range(num_games):
            rewards.append(self.play_a_game())
        return np.array(rewards)

    def train_agent(self):
        eval_rewards = [-1000]
        episode = 0
        scores_window = deque(maxlen=100)
        while eval_rewards[-1] < 200 and episode < 2000:
            episode += 1
            # received initial observation:
            state = self.env.reset()
            score = 0.0
            done = False
            while not done:
                # act one step in the game:
                action = self.next_action(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                # update q_table:
                next_action = self.next_action(next_state)
                if not done:
                    self.q_table[state][action] += Alpha * (
                                reward + GAMMA * self.q_table[next_state][next_action] - self.q_table[state][action])
                else:
                    self.q_table[state][action] += Alpha * (reward - self.q_table[state][action])
                # update for next step:
                state = next_state

            # update epsilon:
            self.epsilon = max(self.epsilon * EpsilonDecay, 0.1)
            # evaluation
            scores_window.append(score)
            # eval_rewards.append(self.eval_agent(num_games=NumEvalGames).mean())
            eval_rewards.append(np.array(scores_window).mean())
            if (len(eval_rewards) % 100) == 0:
                print(
                    f'Episode [{episode}]: Mean rewards: {round(eval_rewards[-1], 3)} | Table Size = {len(self.q_table)}')

        print(f"Solved the game after {episode} episodes!")
        return eval_rewards


# ----------------------- MAIN ---------------------------- #
env = QuantizedWrapper(gym.make("LunarLanderContinuous-v2"), 3, 1)
agent = Agent(env=env)
rewards = agent.train_agent()

# plot the scores
rewards = np.array(rewards)
fig = plt.figure(None, (10, 10))
ax = fig.add_subplot(111)
y = np.array(rewards)
x = np.arange(len(y)) + 1
plt.plot(x, y)
plt.plot(x, np.zeros_like(y) + 200, label='Goal: 200 score')
plt.legend()
plt.grid(which='both', axis='both')
plt.xlim(x[0], x[-1])
plt.ylabel('Average Last 100 Rewards')
plt.xlabel('# Episodes')
plt.show()
