import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule

from experience_replay_buffer import ReplayBuffer

from environment import create_environment

from dqn_model import DQN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()


def epsilon_greedy(state, env, net, epsilon=0.0):
  if np.random.random() < epsilon:
    action = env.action_space.sample()
  else:
    state = torch.tensor(state).to(device)
    q_values = net(state)
    _,action = torch.max(q_values, dim=1)
    action = int(action.item())
  return action

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size=200):
    self.buffer = buffer
    self.sample_size = sample_size

  def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience


class DeepQLearning(LightningModule):
  def __init__(self, env_name, policy=epsilon_greedy, capacity=100_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma= 0.99,
               loss_fn = F.smooth_l1_loss, optim=AdamW, eps_start=1.0,
               eps_end=0.15, eps_last_episode=100, samples_per_epoch = 1_000,
               sync_rate=10, a_start=0.5, a_end = 0.0, a_last_episode=100,
               b_start=0.4, b_end=1.0, b_last_episode=100):
    
    super().__init__()
    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape
    n_actions = self.env.action_space.n

    self.q_net = DQN(hidden_size, obs_size, n_actions) # q network

    self.target_q_net = copy.deepcopy(self.q_net) #target q network

    self.policy = policy
    self.buffer = ReplayBuffer(capacity=capacity)

    self.save_hyperparameters() #saves hyperparameters

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling...")
      self.play_episode(epsilon=self.hparams.eps_start)

  @torch.no_grad()
  def play_episode(self, policy=None, epsilon=0.0):
    state = self.env.reset()
    done = False

    while not done:
      if policy:
        action = policy(state, self.env, self.q_net, epsilon=epsilon)
      else:
        action = self.env.action_space.sample()
      next_state, reward, done, info = self.env.step(action)
      exp = (state, action, reward, done, next_state)
      self.buffer.append(exp)
      state = next_state


      #forward
  def forward(self, x):
    return self.q_net(x)

  # configure optimizers
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer]

  #create dataloader
  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(
        dataset= dataset,
        batch_size= self.hparams.batch_size
    )
    return dataloader

   #training step
  def training_step(self, batch, batch_idx):
    indices, weights, states, actions, rewards, dones, next_states = batch
    weights = weights.unsqueeze(1)
    actions = actions.unsqueeze(1) # creates multiple row in 1 column with unsqueeze(1) (rows = n number of actions, column = 1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    
    '''Q = self.q_net(state),  Q = tensor([[0.3,0.4,0.5],[0.1,0.2,0.3]]
    actions = actions.unqueese(1)  actions = tensor([[1],[2]])
    state_action_values = self.q_net(states).gather(1,action)
    state_acton_values = tensor([[0.4],[0.3]])'''
    state_action_values = self.q_net(states).gather(1, actions)

    with torch.no_grad():
      _, next_actions = self.q_net(next_states).max(dim=1, keepdim=True) #array(highest value, location of highest value)
      next_action_values = self.target_q_net(next_states).gather(1,next_actions)
      '''ex: dones = [[false, true, false, true, true]] then the values will only change
      when done = True which is 0.0 it allows for the agent to not take any actions onces
      it reaches the end of the goal'''
      next_action_values[dones] = 0.0 

    expected_state_action_values = rewards + self.hparams.gamma*next_action_values

    # compute the priorities and update 
    td_errors = (state_action_values - expected_state_action_values).abs().detach()

    for idx, e in zip(indices, td_errors):
      self.buffer.update(idx, e.cpu().item())

    #compute the weighted loss function
    loss = weights * self.hparams.loss_fn(state_action_values, expected_state_action_values, reduction='none')
    loss = loss.mean()

    self.log("episode/Q-Error", loss)
    return loss

  #training epoch end
  def training_epoch_end(self, training_step_outputs):

    epsilon = max(
        self.hparams.eps_end, 
        self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
    )
    alpha = max(
        self.hparams.a_end, 
        self.hparams.a_start - self.current_epoch / self.hparams.a_last_episode
    )
    beta = min(
        self.hparams.b_end, 
        self.hparams.b_start + self.current_epoch / self.hparams.b_last_episode
    )

    self.buffer.alpha = alpha
    self.buffer.beta = beta

    self.play_episode(policy=self.policy, epsilon=epsilon)
    self.log("episode/Return", self.env.game_state.score()) #last episode play by agent

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())
