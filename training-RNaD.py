import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy 

def getreward(x, y): # b gets reward if diff, a gets  
    
    dist_A = torch.distributions.Categorical(x)
    sample_A = dist_A.sample() 
    dist_B =  torch.distributions.Categorical(y)
    sample_B = dist_B.sample()
    
    if(sample_A == sample_B):
        return 1, -1, dist_A.log_prob(sample_A), dist_B.log_prob(sample_B)
    else:
        return -1, 1, dist_A.log_prob(sample_A), dist_B.log_prob(sample_B)
        
class MLP(nn.Module): # smh, couldve jsut trained some parameters wtv tho
    def __init__(self, in_dim, out_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1) 
        
        
training_steps = 30000
mlp_A = MLP(2, 2, 200) # probability of sampling heads, probability of sampling tails
mlp_B = MLP(2, 2, 200)
mlp_A_ref = copy.deepcopy(mlp_A)
mlp_B_ref = copy.deepcopy(mlp_B)

sample_size = 5
input = torch.zeros(2) # dummy, lowkey constant, not needed 

eps = 1e-8
eta = 10 # idk hyperparameter for KL updates
history_A = []
history_B = []
history_exploitable_A = []
history_exploitable_B = []
history_lyuponov_A = []
history_lyuponov_B = []
# remember --> if same, player A wins and if different, player B wins 

optimizer_A = torch.optim.Adam(mlp_A.parameters(), lr=1e-3)
optimizer_B = torch.optim.Adam(mlp_B.parameters(), lr=1e-3)
for i in range(training_steps):
    rewards_A = []
    rewards_B = []
    probs_A = []
    probs_B = []
    kl_A = []
    kl_B = []
    for _ in range(sample_size):
        gen_A = mlp_A(input)
        gen_B = mlp_B(input)
        gen_A_ref = mlp_A_ref(input).detach()
        gen_B_ref = mlp_B_ref(input).detach()
        kl_A.append(gen_A * (torch.log(gen_A) - torch.log(gen_A_ref)))
        kl_B.append(gen_B * (torch.log(gen_B) - torch.log(gen_B_ref)))
        reward_A, reward_B, prob_A, prob_B= getreward(gen_A, gen_B)
        rewards_A.append(torch.tensor(reward_A, dtype=torch.float32))
        rewards_B.append(torch.tensor(reward_B, dtype=torch.float32))
        probs_A.append(prob_A)
        probs_B.append(prob_B)

    with torch.no_grad():
        pA = mlp_A(input)[0].item()
        pB = mlp_B(input)[0].item()
        exploit_A = max(2*pB - 1, 1 - 2*pB) - (pA * (2*pB - 1) + (1-pA) * (1 - 2*pB)) # max expected value playing heads vs tails vs current strategy
        exploit_B = max(2*pA - 1, 1 - 2*pA) - (pB * (2*pA - 1) + (1-pB) * (1 - 2*pA))
        history_exploitable_A.append(exploit_A)
        history_exploitable_B.append(exploit_B)
        history_lyuponov_A.append((0.5 * (torch.log(torch.tensor(0.5, dtype=torch.float32) ) - torch.log(gen_A))).sum())
        history_lyuponov_B.append((0.5 * (torch.log(torch.tensor(0.5, dtype=torch.float32) ) - torch.log(gen_B))).sum())
    
            
    rewards_A = torch.stack(rewards_A).detach() # detach graph
    rewards_B =  torch.stack(rewards_B).detach()
    
    probs_A = torch.stack(probs_A)  # preserve gradient graph,ldike so grad can floww
    probs_B = torch.stack(probs_B) 
    kl_A = torch.stack(kl_A)
    kl_B = torch.stack(kl_B)
    
    
    rewards_A = (rewards_A - rewards_A.mean())/(rewards_A.std() + eps)
    rewards_B = (rewards_B - rewards_B.mean())/(rewards_B.std() + eps)
    loss_A = -(rewards_A * probs_A).mean() + eta * kl_A.sum(dim=-1).mean() # each kl is initially sample_size * 2 tensor
    loss_B = -(rewards_B * probs_B).mean() + eta * kl_B.sum(dim=-1).mean()
    
    if((i+1) % 10 == 0):
        print(f"loss A: {loss_A} loss_B: {loss_B}")
        print(f"Probs A: {prob_A} | Probs B: {prob_B}")
    with torch.no_grad():
        history_A.append(mlp_A(input)[0].item())
        history_B.append(mlp_B(input)[0].item())
    optimizer_A.zero_grad()
    optimizer_B.zero_grad()
    loss_A.backward()
    loss_B.backward()
    optimizer_A.step()
    optimizer_B.step()
    
    if (i+1) % 500 == 0:
        mlp_A_ref = copy.deepcopy(mlp_A)
        mlp_B_ref = copy.deepcopy(mlp_B)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

ax1.plot(history_A, label="Player A P(Heads)", alpha=0.7)
ax1.plot(history_B, label="Player B P(Heads)", alpha=0.7)
ax1.axhline(y=0.5, color='r', linestyle='--', label="Nash Equilibrium")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("P(Heads)")
ax1.set_title("Policy Convergence")
ax1.legend()

nashconv = [a + b for a, b in zip(history_exploitable_A, history_exploitable_B)]
ax2.plot(nashconv, label="NashConv", alpha=0.7, color='purple')
ax2.axhline(y=0.0, color='r', linestyle='--', label="Nash (0)")
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Exploitability")
ax2.set_title("NashConv (Exploitability)")
ax2.legend()

lyapunov = [a + b for a, b in zip(history_lyuponov_A, history_lyuponov_B)]
ax3.plot(lyapunov, label="Lyapunov Potential", alpha=0.7, color='green')
ax3.axhline(y=0.0, color='r', linestyle='--', label="Nash (0)")
ax3.set_xlabel("Training Step")
ax3.set_ylabel("KL(pi || uniform)")
ax3.set_title("Lyapunov Potential Decay")
ax3.legend()

plt.tight_layout()
plt.savefig("convergence_rnad.png")
plt.show()