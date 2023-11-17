from trainings import DeepQLearning
import torch
from pytorch_lightning import Trainer
from display import display_video

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

model = DeepQLearning(
    "FlappyBird-v0",
    lr=5e-4,
    hidden_size = 512,
    eps_end=0.01,
    eps_last_episode=1_000,
    capacity=10_000,
    gamma=0.9
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs=3_000,
    log_every_n_steps = 1
)

trainer.fit(model)


env = model.env
policy = model.policy
q_net = model.q_net.to(device)
frames = []

for episode in range(10):
  done = False
  obs = env.reset()
  while not done:
    frames.append(env.render(mode="rgb_array"))
    action = policy(obs,env, q_net)
    obs, _,done, _= env.step(action)
    
display_video(frames)