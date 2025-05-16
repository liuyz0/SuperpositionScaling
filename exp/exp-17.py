# large toy model
# train with V100 GPU and FP16
# seems with FP16, results are different from CPUs? use FP32

import torch
from torch import nn
import math
import argparse
from adamw import AdamW
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10240, help="output dimension")
parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=-1.0, help="weight decay")
parser.add_argument("--batch_size", type=int, default=8192, help="batch size")
parser.add_argument("--n_steps", type=int, default=40000, help="number of steps")
parser.add_argument("--dist", type=str, default="power", help="distribution of features")

args = parser.parse_args()

def probability(name: str, n = args.n):
    names = ['exponential', 'power', 'linear', 'exponential1']
    assert name in names, f"Distribution {name} not found"
    if name == 'exponential':
        prob = torch.exp( - torch.arange(n) / 1000)
        prob = prob / prob.sum()
    elif name == 'power':
        prob = torch.tensor([1.0 / i ** 1.2 for i in range(1,n+1)])
        prob = prob / prob.sum()
    elif name == 'linear':
        prob = torch.tensor([(n-i) / n for i in range(n)])
        prob = prob / prob.sum()
    elif name == 'exponential1':
        prob = torch.exp( - torch.arange(n) / 400)
        prob = prob / prob.sum()
    return prob

m_ran = 2 ** torch.arange(3, 11)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prob = probability(args.dist).to(device)
#scaler = torch.GradScaler()
criteria = nn.MSELoss()

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

def get_lr(step, n_steps, warmup_ratio=.1):
    assert warmup_ratio <= 1
    warmup_steps = int(n_steps * warmup_ratio)
    step = step + 1
    min_lr = 0.05
    if step < warmup_steps:
        return step / warmup_steps
    else:
        # cosine decay
        return (1.0 - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))) + min_lr

results = {}
Ws = {}
results['m_ran'] = m_ran
losses = torch.zeros(len(m_ran), args.n_steps)

for m_i, m in enumerate(m_ran):
    t0 = time.perf_counter()
    model = torch.compile(FeatureRecovery(args.n, m)).to(device)
    print(f"compile time: {time.perf_counter() - t0:.2f}s")
    parameter_groups = [{'params': model.W, 'weight_decay': args.weight_decay, 'lr': args.lr * (8 / m) ** 0.25},
                        {'params': model.b, 'weight_decay': 0.0, 'lr': 2.0 / m}]
    optimizer = AdamW(parameter_groups)
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']

    t0 = time.perf_counter()
    for step in range(args.n_steps):
        optimizer.zero_grad(set_to_none=True)
        # generate data
        x = (torch.rand(args.batch_size, args.n, device=device) < prob) * torch.rand(args.batch_size, args.n, device=device) * 2
        # update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group["init_lr"] * get_lr(step, args.n_steps,warmup_ratio=.05)
        # training
        #with torch.autocast(device_type=device.type):
        y = model(x)
        loss = criteria(y, x) * 100
        losses[m_i, step] = loss.item() / 100
        loss.backward()
        optimizer.step()
        '''
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        '''

    Ws[m_i] = model.W.detach().cpu()
    print(f"m: {m}, Loss: {losses[m_i,-1]:.2e}, Run time: {time.perf_counter() - t0:.2f}s")

results['losses'] = losses
results['W'] = Ws

torch.save(results, f"../outputs/exp-17-{args.dist}_{args.weight_decay:.2f}.pt")
