import torch
import torch.nn as nn
import torch.nn.functional as F 

def getreward(x):
    return 0

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1) 
        
        
training_steps = 10000
mlp = MLP(100, 100, 200)
sample_size = 5
input = torch.zeros(100)
rewards = []
probs= []
eps = 1e-8
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-8)
for _ in range(training_steps):
    for _ in range(sample_size):
        gen = mlp(sample_size)
        reward = getreward(gen) 
        rewards.append(reward)
        probs.append(gen)
    
    rewards = torch.tensor(rewards)
    probs = torch.tensor(probs) 
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)
    loss = -(rewards * probs).mean()
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()