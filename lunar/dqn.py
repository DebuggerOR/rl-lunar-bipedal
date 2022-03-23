model_type = 'Dueling'  # {DQN, Dueling}

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 1
EpsilonDecay = 0.99
Uncertainty = False

seed = 124 if model_type == 'DQN' else 125
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

###############################################
#                 Helpers
###############################################
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


###############################################
#               Env Wrapper
###############################################
class UncertaintyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(UncertaintyWrapper, self).__init__(env)

    def noisy_observation(self, observation):
        observation[0] += np.random.normal(0, 0.05)
        observation[1] += np.random.normal(0, 0.05)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.noisy_observation(observation), reward, done, info

    def reset(self):
        return self.noisy_observation(self.env.reset())


class EnvDiscreteActionsWrapper(gym.Wrapper):
    def __init__(self, env, num_discrete_actions, space_digits=None):
        super(EnvDiscreteActionsWrapper, self).__init__(env)
        self.space_digits = space_digits
        self.action_space = []
        for a1 in np.linspace(-1, 1, num_discrete_actions):
            for a2 in np.linspace(-1, 1, num_discrete_actions):
                self.action_space.append(np.array([a1, a2]))

    def action(self, action):
        return self.action_space[action]

    def observation(self, observation):
        if self.space_digits is None:
            return observation
        else:
            return np.round(observation, self.space_digits)

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), reward, done, info

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)


###############################################
#                  Networks
###############################################
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super().__init__()

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.out(x)
        return action


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super().__init__()

        self.num_actions = action_size
        fc3_1_size = fc3_2_size = 32

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3_1 = nn.Linear(fc2_size, fc3_1_size)
        self.fc4_1 = nn.Linear(fc3_1_size, 1)
        self.fc3_2 = nn.Linear(fc2_size, fc3_2_size)
        self.fc4_2 = nn.Linear(fc3_2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        val = F.relu(self.fc3_1(x))
        val = self.fc4_1(val)

        adv = F.relu(self.fc3_2(x))
        adv = self.fc4_2(adv)

        action = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.num_actions)
        return action


###############################################
#               Replay Buffer
###############################################
class ReplayBuffer:
    def __init__(self, env, action_size, buffer_size, batch_size, fill_at_init=True):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # fill buffer
        if fill_at_init:
            print(f"start fill replay-buffer with {buffer_size} samples...")
            freq_update = buffer_size // 100
            state = env.reset()  # initial state
            while len(self.memory) < buffer_size:
                action = random.randint(0, len(env.action_space) - 1)
                next_state, reward, done, _ = env.step(action)
                self.add(state, action, reward, next_state, done)
                if done:
                    state = env.reset()
                else:
                    state = next_state

                if (len(self.memory) % freq_update) == 0:
                    print('\rReplay-Buffer filled %d%%...' % (int(100 * (len(self.memory) / buffer_size))), end="")
        print('\n')


    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


###################################################
#                    Agent
###################################################
MODEL_ARCHITECTURE = QNetwork if model_type == 'DQN' else DuelingQNetwork
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


class Agent:
    def __init__(self, env, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = MODEL_ARCHITECTURE(state_size, action_size).to(device)
        self.qnetwork_target = MODEL_ARCHITECTURE(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(env, action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn_DDQN(experiences, GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn_DDQN(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # Get index of maximum value for next state from Q_expected
        Q_argmax = self.qnetwork_local(next_states).detach()
        _, a_prime = Q_argmax.max(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, a_prime.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


# ----------------------- MAIN ---------------------------- #
tic = time.time()
env = gym.make('LunarLanderContinuous-v2')
env.seed(seed)
env.action_space.seed(seed)
env = EnvDiscreteActionsWrapper(env=env, num_discrete_actions=3, space_digits=None)
if Uncertainty:
    env = UncertaintyWrapper(env)
agent = Agent(env=env, state_size=env.observation_space.shape[0], action_size=len(env.action_space))

def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=EpsilonDecay):
    scores = []  # list containing scores from each episode
    average_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        average_scores.append(np.mean(scores_window))
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('Episode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)))
        if np.mean(scores_window) >= 200:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break

    return scores, average_scores


scores, average_scores = train()
scores, average_scores = np.array(scores), np.array(average_scores)

# --- time --- #
toc = time.time()
duration = toc - tic
H = int(duration / 3600)
M = int((duration - (H * 3600)) / 60)
S = int(duration - (H * 3600) - (M * 60))
print(f"\n------------------\nTraining Time: Time -> {H}h : {M}m : {S}s\n------------------\n")

# --- plot training graph --- #
rewards = np.array(average_scores)
fig = plt.figure(None, (10, 10))
ax = fig.add_subplot(111)
y = np.array(rewards)
x = np.arange(len(y)) + 1
plt.plot(x, y)
plt.plot(x, np.zeros_like(y)+200, label='Goal: 200 score')
plt.legend()
plt.grid(which='both', axis='both')
plt.xlim(x[0], x[-1])
plt.ylabel('Average Last 100 Rewards')
plt.xlabel('# Episodes')
plt.show()