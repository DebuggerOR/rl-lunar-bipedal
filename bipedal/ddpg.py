import wandb
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

from bipedal.utils import Critic, Actor, OUNoise


class DDPGAgent:
    def __init__(self, env, gamma, tau, buffer_maxlen, batch_size, critic_learning_rate, actor_learning_rate, seed):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hyper parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # initialize actor and critic networks
        self.critic = Critic(env.observation_space.shape[0], env.action_space.shape[0], seed).to(self.device)
        self.critic_target = Critic(env.observation_space.shape[0], env.action_space.shape[0], seed).to(self.device)

        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], seed).to(self.device)
        self.actor_target = Actor(env.observation_space.shape[0], env.action_space.shape[0], seed).to(self.device)

        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)

        self.buffer = ReplayBuffer(buffer_maxlen, batch_size, seed)
        self.noise = OUNoise(env.action_space.shape[0])

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()

        action = action.cpu().numpy()
        return action

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.buffer.add(state, action, reward, next_state, done)

        q_loss, policy_loss = None, None

        # If enough samples are available in buffer, get random subset and learn
        if len(self.buffer) >= self.batch_size:
            experiences = self.buffer.sample()
            q_loss, policy_loss = self.learn(experiences)
            q_loss = q_loss.detach().item()
            policy_loss = policy_loss.detach().item()

        return q_loss, policy_loss

    def learn(self, experiences):
        """Updating actor and critic parameters based on sampled experiences from replay buffer."""
        states, actions, rewards, next_states, dones = experiences

        curr_Q = self.critic(states, actions)
        next_actions = self.actor_target(next_states).detach()
        next_Q = self.critic_target(next_states, next_actions).detach()
        target_Q = rewards + self.gamma * next_Q * (1 - dones)

        # losses
        q_loss = F.mse_loss(curr_Q, target_Q)
        policy_loss = -self.critic(states, self.actor(states)).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update critic
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        return q_loss, policy_loss


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)


def run_episode_ddpg(Q, env):
    """ Runs an episode for an agent, returns the cumulative reward."""
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = Q.get_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    return return_


def ddpg_train(env, agent, max_episodes, std_dev, eps_start, eps_end, eps_decay, wandb_report):
    """ Training the ddpg agent. Applying eps-decay exploration."""
    all_training_returns = []
    all_actions = []
    all_actions_raw = []
    val_returns = []
    eps = eps_start
    q_losses, policy_losses = [], []
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        training_return = 0
        while True:
            actor_action = agent.get_action(state)
            action_raw = actor_action + agent.noise.sample(eps * std_dev)
            action = np.clip(action_raw, -1, 1)

            all_actions.append(action)
            all_actions_raw.append(action_raw)

            next_state, reward, done, _ = env.step(action)
            training_return += reward
            q_loss, policy_loss = agent.step(state, action, reward, next_state, done)
            q_losses.append(q_loss)
            policy_losses.append(policy_loss)

            if done:
                break

            state = next_state
        eps = max(eps_end, eps_decay * eps)
        all_training_returns.append(training_return)

        # Calculate return based on current policy
        if episode % 10 == 0:
            ddpg_return = run_episode_ddpg(agent, env)
            val_returns.append(ddpg_return)
            if wandb_report: wandb.log({'validation_return_DDPG': ddpg_return}, step=episode)
            print(f'episode {episode}, eps: {eps:.2f}, return: {training_return:.1f}, Val return = {ddpg_return:.1f}')

        if wandb_report: wandb.log({'training_return_DDPG': training_return}, step=episode)
        if episode % 250 == 0:
            torch.save(agent.actor.state_dict(), f'Checkpoint/ddpg_actor_{wandb.run.name}_episode_{episode}.pth')
            torch.save(agent.critic.state_dict(), f'Checkpoint/ddpg_critic_{wandb.run.name}_episode_{episode}.pth')

    all_actions = np.stack(all_actions)
    all_actions_raw = np.stack(all_actions_raw)

    return agent, all_training_returns, all_actions, val_returns, all_actions_raw, q_losses, policy_losses
