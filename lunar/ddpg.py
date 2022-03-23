# Global Parameters
ReplayBufferSize = int(1e5)
GAMMA = 0.99
LR_Actor = 1e-4
LR_Critic = 1e-3
Weight_Decay_Critic = 1e-2
Epsilon = 1.0
EpsilonDecay = 0.99
NumEvalGames = 5
BatchSize = 256
Uncertainty = True
UpdatesPerStep = 2

seed = 124 if Uncertainty else 125
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


###############################################
#                Env Wrapper
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


###############################################
#                 Helpers
###############################################
# soft target update:
def soft_target_update(target_model, trained_model, tau=0.001):
    for target_p, trained_p in zip(target_model.parameters(), trained_model.parameters()):
        target_p.data.copy_(tau * trained_p.data + (1 - tau) * target_p.data)


# from numpy to tensor
def numpy_to_tensor(x, device=None):
    x = torch.from_numpy(x)
    if len(x.shape) < 2:
        x = torch.unsqueeze(x, dim=0)
    if device:
        x = x.to(device)
    return x


# from tensor to numpy
def tensor_to_numpy(x):
    x = x.cpu().detach().numpy()
    x = np.squeeze(x)
    return x


###############################################
#         Networks - Actor & Critic
###############################################
class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden1_dims=400, hidden2_dims=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dims, hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, action_dims)

        self._reset_parameters()

    def forward(self, states):
        x = torch.relu(self.fc1(states))
        x = torch.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))
        return actions

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name: continue
            # last layer has unique initialization:
            if 'fc3' in name:
                torch.nn.init.uniform_(param, -3e-3, 3e-3)
            elif 'fc1' in name or 'fc2' in name:
                fan_in = param.size(1)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(param, -bound, bound)


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden1_dims=400, hidden2_dims=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dims, hidden1_dims)
        self.fc2 = nn.Linear(hidden1_dims + action_dims, hidden2_dims)
        self.fc3 = nn.Linear(hidden2_dims, 1)

        self._reset_parameters()

    def forward(self, states, actions):
        x = torch.relu(self.fc1(states))
        # add actions to x:
        x = torch.cat([x, actions], dim=-1)
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name: continue
            # last layer has unique initialization:
            if 'fc3' in name:
                torch.nn.init.uniform_(param, -3e-4, 3e-4)
            elif 'fc1' in name or 'fc2' in name:
                fan_in = param.size(1)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(param, -bound, bound)


###############################################
#               Replay Buffer
###############################################
class ReplayBuffer:
    def __init__(self, env, buffer_size=ReplayBufferSize, batch_size=BatchSize, fill_at_init=True):
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.indices = np.arange(self.buffer_size)

        # fill buffer
        if fill_at_init:
            print(f'start fill replay-buffer with {self.buffer_size} samples')
            freq_update = self.buffer_size // 100
            state = env.reset()  # initial state
            while len(self.buffer) < self.buffer_size:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                self.buffer.append(
                    (numpy_to_tensor(state), numpy_to_tensor(action), reward, numpy_to_tensor(next_state), done))
                if done:
                    state = env.reset()
                else:
                    state = next_state

                if (len(self.buffer) % freq_update) == 0:
                    print('\rReplay-Buffer filled %d%%...' % (int(100 * (len(self.buffer) / self.buffer_size))), end="")
            print('\nDone.')

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, device=None):
        states = []
        actions = []
        rewards = torch.zeros(self.batch_size)
        next_states = []
        done = torch.zeros(self.batch_size, dtype=torch.bool)

        sampled_indices = np.random.choice(self.indices, self.batch_size, replace=False)
        for j, i in enumerate(sampled_indices):
            s, a, r, nxt_s, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards[j] = r
            next_states.append(nxt_s)
            done[j] = d

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        next_states = torch.cat(next_states, dim=0)
        done = done.type(torch.float32)
        if device:
            states = states.to(device)
            actions = actions.to(device)
            rewards = torch.unsqueeze(rewards, dim=1).to(device)
            next_states = next_states.to(device)
            done = torch.unsqueeze(done, dim=1).to(device)
        return states, actions, rewards, next_states, done


###################################################
#    Exploration Noise - Ornstein-Uhlenbeck
###################################################
class OrnsteinUnlenbeck:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2, dt=1e-2, device=None):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = torch.zeros(self.size)
        self.xt = self.x0.clone()
        self.device = device

    def reset(self):
        self.xt = self.x0.clone()

    def __call__(self):
        dxt = self.theta * (self.mu - self.xt) * self.dt + self.sigma * math.sqrt(self.dt) * torch.normal(mean=0, std=1,
                                                                                                          size=(
                                                                                                          self.size,))
        self.xt = self.xt + dxt
        if self.device:
            return torch.unsqueeze(self.xt.clone(), dim=0).to(self.device)
        return torch.unsqueeze(self.xt.clone(), dim=0)


###################################################
#                  DDPG - Agent
###################################################
class DDPGAgent:
    def __init__(self, env, max_episodes=500, max_steps=1000, use_gpu=False):
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
        print("Using device: ", self.device)

        assert isinstance(env, gym.Env)
        self.env = env
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]

        # initial networks:
        self.actor = Actor(self.state_dims, self.action_dims).to(self.device)
        self.critic = Critic(self.state_dims, self.action_dims).to(self.device)
        # initial target networks:
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

    @torch.no_grad()
    def play_a_game(self, env=None):
        if env is None:
            env = self.env
        self.actor.eval()
        cum_reward = 0.
        state = numpy_to_tensor(env.reset(), self.device)
        for t in range(self.max_steps):
            action = self.actor(state)
            next_state, reward, done, _ = env.step(tensor_to_numpy(action))
            cum_reward += reward
            state = numpy_to_tensor(next_state, self.device)
            if done:
                break
        self.actor.train()
        return cum_reward

    @torch.no_grad()
    def play_a_game_with_animation(self, outputPath=None):
        frames = []
        self.actor.eval()
        cum_reward = 0.
        state = numpy_to_tensor(self.env.reset(), self.device)
        for t in range(self.max_steps):
            frames.append(env.render(mode="rgb_array"))
            action = self.actor(state)
            next_state, reward, done, _ = self.env.step(tensor_to_numpy(action))
            cum_reward += reward
            state = numpy_to_tensor(next_state, self.device)
            if done:
                break
        self.actor.train()
        return cum_reward

    def eval_agent(self, num_games=100):
        rewards = []
        for _ in range(num_games):
            rewards.append(self.play_a_game())
        return np.array(rewards)

    # def save_agent(self):
    #     if self.checkpoint is None:
    #         return
    #     states = {'actor': self.actor,
    #               'critic': self.critic}
    #     torch.save(states, str(self.checkpoint / 'best_agent.pth'))

    # def load_pretrained_agent(self, file_path):
    #     states = torch.load(str(file_path), map_location=self.device)
    #     # load actor:
    #     self.actor = states.pop('actor')
    #     # load critic:
    #     self.critic = states.pop('critic')

    def train_agent(self):
        # initialized optimizers:
        actor_optimizer = Adam(self.actor.parameters(), lr=LR_Actor)
        critic_optimizer = Adam(self.critic.parameters(), lr=LR_Critic, weight_decay=Weight_Decay_Critic)

        eval_rewards = []
        La_list = []
        Lc_list = []

        # replay buffer:
        replay_buffer = ReplayBuffer(env)
        # exploration noise:
        noise = OrnsteinUnlenbeck(self.action_dims, dt=1, device=self.device)

        scores_window = deque(maxlen=100)
        epsilon = Epsilon
        for episode in range(self.max_episodes):
            La_per_episode = []
            Lc_per_episode = []
            # exploration noise:
            noise.reset()
            # received initial observation:
            state = self.env.reset()
            score = 0.0
            for t in range(self.max_steps):
                # act one step in game:
                self.actor.eval()
                with torch.no_grad():
                    action = self.actor(numpy_to_tensor(state, self.device)) + noise() * epsilon
                    action = torch.clamp(action, min=-1, max=1)
                next_state, reward, done, _ = self.env.step(tensor_to_numpy(action))
                self.actor.train()
                score += reward

                # add transition to replay-buffer:
                replay_buffer.add(numpy_to_tensor(state), action.detach().cpu(), reward, numpy_to_tensor(next_state),
                                  done)

                for i in range(UpdatesPerStep):
                    # sample a random mini-batch from replay-buffer:
                    s, a, r, nxt_s, d = replay_buffer.sample_batch(device=self.device)

                    ################
                    # Critic Loss
                    ################
                    # calculate predicted q-value:
                    Q_predicted = self.critic(s, a)
                    # calculate target q-value:
                    nxt_a = self.actor_target(nxt_s)
                    Q_tag = self.critic_target(nxt_s, nxt_a)
                    # if terminate state -> Q_tag needs to be zero:
                    Q_tag = Q_tag * (1 - d)
                    y = r + GAMMA * Q_tag
                    Lc = F.mse_loss(Q_predicted, y)
                    critic_optimizer.zero_grad()
                    clip_grad_norm_(self.critic.parameters(), 1.)
                    Lc.backward()
                    critic_optimizer.step()
                    Lc_per_episode.append(Lc.item())

                    ################
                    # Actor Loss
                    ################
                    # we we multiply by -1 because we want higher values.
                    La = -1 * self.critic(s, self.actor(s))
                    La = La.mean()
                    actor_optimizer.zero_grad()
                    clip_grad_norm_(self.actor.parameters(), 1.)
                    La.backward()
                    actor_optimizer.step()
                    La_per_episode.append(La.item())

                    # soft target updating:
                    soft_target_update(self.actor_target, self.actor)
                    soft_target_update(self.critic_target, self.critic)

                # if the episode ends before max_steps:
                if done:
                    break
                else:
                    state = next_state

            # update epsilon:
            epsilon = max(epsilon * EpsilonDecay, 0.1)
            # update La & Lc lists:
            La_list.append(np.array(La_per_episode).mean())
            Lc_list.append(np.array(Lc_per_episode).mean())
            # evaluation
            scores_window.append(score)
            # eval_rewards.append(self.eval_agent(num_games=NumEvalGames).mean())
            eval_rewards.append(np.array(scores_window).mean())
            print(
                f'Episode [{episode + 1}]: La = {round(La_list[-1], 3)} | Lc = {round(Lc_list[-1], 3)} | Score: {round(score, 3)} | Mean rewards: {round(eval_rewards[-1], 3)}')

            # save best episode:
            if eval_rewards[-1] >= 200:
                print(f"Solve the game after {episode} episodes")
                return eval_rewards


# ----------------------- MAIN ---------------------------- #
tic = time.time()
env = gym.make("LunarLanderContinuous-v2")
env.seed(seed)
env.action_space.seed(seed)
if Uncertainty:
    env = UncertaintyWrapper(env)
# env = EnvQuantStateWrapper(env, 2)
agent = DDPGAgent(env=env, use_gpu=True, max_episodes=1000)
rewards = agent.train_agent()

toc = time.time()
duration = toc - tic
H = int(duration / 3600)
M = int((duration - (H * 3600)) / 60)
S = int(duration - (H * 3600) - (M * 60))
print(f"\n------------------\nTraining Time: Time -> {H}h : {M}m : {S}s\n------------------\n")

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
