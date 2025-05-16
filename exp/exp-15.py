# 96 CPUS
# 1 replicates
# each replicate 96 CPUs 
# for 10 weight decays from -1.0 to 1.0
# 6 different # of params
# 1 alphas

# 9 different sparsity

# fix possible n 
# fix batch size

import sys
import torch
from torch import nn
import math
from adamw import AdamW

# define models and function
class FeatureRecovery(nn.Module):
    def __init__(self, n,m):
        super(FeatureRecovery, self).__init__()
        # n is number of features
        # m is number of hidden dimensions
        self.W = nn.Parameter(torch.randn(n, m) / math.sqrt(m))
        self.b = nn.Parameter(torch.randn(n))
        self.relu = nn.ReLU()
    def forward(self, x):
        # x [batch_size, n]
        return self.relu(x @ self.W @ self.W.T + self.b)

def get_lr(step, lr, n_steps, warmup_steps=2000):
    step = step + 1
    min_lr = 0.05 * lr
    if warmup_steps < n_steps:
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            # cosine decay
            return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))) + min_lr
    else:
        return (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (step) / (n_steps))) + min_lr

def per_sample_loss(x,y):
    # x [batch_size, n]
    # y [batch_size, n]
    # MSE loss for each sample
    # output [batch_size]
    return ((x - y) ** 2).mean(dim=-1)
    

# Grab the argument that is passed in
# This is the index into fnames for this process
task_id = int(sys.argv[1]) # from 0 to 89
num_tasks = int(sys.argv[2]) # should be 90

wd_ran = torch.linspace(-1, 1, 10)
wd = wd_ran[task_id % 10]

sparsity_ran = torch.linspace(0, 1, 9)
sparsity_level = sparsity_ran[task_id // 10]

alpha = 0.0

m_ran = torch.floor(torch.exp(torch.linspace(math.log(10), math.log(100), 6))).long()

n = 1000 # number of features fixed
prob = torch.tensor([1 / i ** (1 + alpha) for i in range(1, n+1)])
prob = prob / prob.sum()
prob = prob * (1.0 + sparsity_level* (1 / prob[0] - 1.0))
#D = int(1.1 / prob[-1])
#D = int(n ** (1+alpha) / alpha)
batch_size = 2048

# data generated
# x = (torch.rand(D, n) < prob).float() * torch.rand(D, n) * 2

criteria = nn.MSELoss()
n_steps = 20000
results = torch.zeros(len(m_ran), n_steps)
W_matrices = {}

#log_steps = 100
#per_feature_loss = torch.zeros(len(m_ran), int(n_steps/log_steps), n)

for ii in range(len(m_ran)):
    m = m_ran[ii].item()
    model = FeatureRecovery(n, m)
    weight_decay = wd
    parameter_groups = [{'params': model.W, 'weight_decay': weight_decay}, {'params': model.b, 'weight_decay': 0.0}]
    optimizer = AdamW(parameter_groups)
    
    for step in range(n_steps):
        # generate data
        x = (torch.rand(batch_size, n) < prob).float() * torch.rand(batch_size, n) * 2
        # update learning rate
        lr = get_lr(step, 1e-2, n_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # training
        optimizer.zero_grad()
        y = model(x)
        loss = criteria(y, x)
        loss.backward()
        optimizer.step()
        results[ii, step] = loss.item()

        #if (step+1) % log_steps == 0:
            #with torch.no_grad():
                #prediction = model(torch.eye(n))
                #per_feature_loss[ii, step // log_steps] = per_sample_loss(prediction, torch.eye(n)).detach().clone()

    W_matrices[ii] = model.W.detach().clone()
    print(f"sparsity: {task_id // 10}, weight_decay: {task_id % 10}, m_idx: {ii}")

torch.save(results, f"../outputs/exp-15-{task_id}.pt")
torch.save(W_matrices, f"../outputs/exp-15-W-{task_id}.pt")
#torch.save(per_feature_loss, f"../outputs/exp-9-pf-{task_id}.pt")