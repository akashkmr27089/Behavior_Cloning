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
from collections import deque


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
ppo.load_state_dict(torch.load("./PPO_LunarLander-v2.pth",map_location=torch.device('cpu')))


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


# %%
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def play(model):
    state = env.reset()
    score = 0
    for _ in range(max_len):
        action = model.act(state)
        env.render()
        state, reward, done , _ = env.step(action)
        score += reward
        time.sleep(0.1) 
        if done:break
    env.close()
    return score


# %%
class Storage:
    def __init__(self):
        self.states = deque(maxlen=10000)
        self.actions = deque(maxlen=10000)

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
        return self.stack(list(self.states)), self.stack(list(self.actions))

    def append(self,states, actions):
        self.actions += actions
        self.states += states

    def return_full_data(self):
        return (self.states, self.actions)


# %%
max_len = 200
initial_episode = 5
lr = 0.0003             
betas = (0.9, 0.999)
batch_size = 128
training_epoch = 200
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
        if done: break
    print("\rCurrent Iteration {} ".format(k), end = " ")
    if initial_collection_visizliztion: env.close()


# %%



# %%
def training():
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
        if i%50 == 0:
            print("Iteration {} with Current Loss :{}".format(i, sum(loss_sample)*(1/len(loss_sample))))


# %%
## Functions is used to do step 2 and 3 of Dagger Algo
## this function is used to play using behavioural Network and 
## rectifing actions inaccordance with perfect network
def play_imporve_and_return_storage():
    secondary_storage = Storage()
    state = env.reset()
    while True:
        action = behavior_cloniing.act(state)
        secondary_storage.states.append(state.reshape(1,-1))
        state, _ , done, _ = env.step(action)
        action_selected_by_master = ppo.act(state)
        secondary_storage.actions.append(secondary_storage.return_onehot(action_selected_by_master))
        if done: break
    return secondary_storage


# %%
states, actions = storage.sample()
behavior_cloniing = ActorCritic(state_dim, action_dim, 64).to(device)
optimizer = torch.optim.Adam(behavior_cloniing.parameters(), lr=lr, betas=betas)
# optimizer = torch.optim.SGD(behavior_cloniing.parameters(), lr = 0.01)
MseLoss = nn.MSELoss()
print(storage.states.__len__())


# %%
def DaGGER(iterations = 10):
    print("Initial Training with Master Network")
    training()
    for i in range(iterations):
        print("Game {} :".format(i))
        secondary_storage = play_imporve_and_return_storage()
        new_states, new_actions = secondary_storage.return_full_data()
        storage.append(new_states, new_actions)
        print("Size of new Storage {}".format(storage.states.__len__()))
        training()


# %%
DaGGER()


# %%
play(behavior_cloniing)


# %%


