# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import gym
from PPO import PPO
from PIL import Image
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
import numpy as np


# %%
env_name = "LunarLander-v2"
# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 4
device = "cpu"


# %%
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    
ppo = ActorCritic(state_dim, action_dim, 64)


# %%
ppo.load_state_dict(torch.load("PPO_LunarLander-v2.pth"))


# %%
## This is only for testing if it works
state = env.reset()
while True:
    action = ppo.act(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    time.sleep(0.01)
    if done: break
env.close()

# %% [markdown]
# #### Start of behaviour Cloning

# %%
class Storage:
    def __init__(self):
        self.states = []
        self.actions = []

    def reset(self):
        del self.states[:]
        del self.actions[:]

    def return_onehot(self, data, size = action_dim):
        final = [0]*size
        final[data] += 1
        return torch.tensor(final).float().reshape(1,-1).to(device)

    def stack(self, data):
        return torch.cat(data, dim=0).detach()

    def sample(self):
        return self.stack(self.states), self.stack(self.actions)

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


# %%
max_len = 200
initial_episode = 20
lr = 0.0003             
betas = (0.9, 0.999)
batch_size = 128
training_epoch = 1000
initial_collection_visizliztion = False


# %%
## Data Collection 
storage = Storage()
for k in range(initial_episode):
    state = env.reset()
    for i in range(max_len):
        action = ppo.act(state)
        if initial_collection_visizliztion: env.render()
        storage.states.append(torch.tensor(state).reshape(1,-1))
        storage.actions.append(storage.return_onehot(action))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        time.sleep(0.01)
        if done: break
    print("\rCurrent Iteration {} ".format(k), end = " ")
    if initial_collection_visizliztion: env.close()


# %%
states, actions = storage.sample()
behavior_cloniing = ActorCritic(state_dim, action_dim, 64).to(device)
optimizer = torch.optim.Adam(behavior_cloniing.parameters(), lr=lr, betas=betas)
# optimizer = torch.optim.SGD(behavior_cloniing.parameters(), lr = 0.01)
MseLoss = nn.MSELoss()


# %%
## Training Network for supervised network
for i in range(training_epoch):
    loss_sample = []
    for sampled_set in random_sample(np.arange(states.size()[0]), batch_size):
        y_pred = behavior_cloniing.action_layer(states[sampled_set])
        y_actual = actions[sampled_set]
        optimizer.zero_grad()
        loss = MseLoss(y_pred, y_actual)
        loss_sample.append(loss.data)
        loss.backward()
        optimizer.step()
    if i%10 == 0:
        print("Iteration {} with Current Loss :{}".format(i, sum(loss_sample)*(1/len(loss_sample))))


# %%
## This is only for testing if it works
state = env.reset()
while True:
    action = behavior_cloniing.act(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    time.sleep(0.01)
    if done: break
env.close()
