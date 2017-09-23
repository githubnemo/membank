import torch
import torch.nn as nn
from torch.autograd import Variable



class Latch(nn.Module):

    def __init__(self, size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_train = Variable(torch.zeros(size), requires_grad=True)
        self.q_valid = Variable(torch.zeros(size))
        self.act = nn.Hardtanh(min_val=0, max_val=1)
        self.consent_threshold = 0.75

    def t(self, s):
        # all inputs in s must say 'store' or otherwise
        # there is no consent to store an value.
        return (self.act(s).mean(-1) >= self.consent_threshold).float()

    def q(self):
        q = self.q_train if self.training else self.q_valid
        return Variable(q.data)

    def forward(self, s, v):
        t = self.t(s).unsqueeze(-1)
        new_q = v * t + self.q() * (1 - t)

        if self.training:
            self.q_train = new_q
        else:
            self.q_valid = new_q

        return new_q


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_units = output_dim

        self.latch_size = self.num_units
        self.latch = Latch(self.latch_size)

        self.units = nn.Linear(input_dim, output_dim)
        self.memory = nn.Linear(self.latch_size, self.num_units)
        self.set = nn.Parameter(torch.zeros(self.num_units))
        self.value = nn.Linear(self.num_units, self.latch_size)

        self.act = nn.Tanh()

    def forward(self, x):
        # x.shape = (b, t, u)
        t = x.size(1)
        acts = []
        qs = []
        for ti in range(t):
            v = self.units(x[:,ti]) + self.memory(self.latch.q())
            q = self.latch(self.set * v, self.value(v))
            acts += [self.act(v)]
            qs += [q]
        return torch.stack(acts, dim=1), torch.stack(qs, dim=0)


class BlockLayer(nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        self.units_per_block = output_dim // num_blocks
        self.blocks = [Block(input_dim=input_dim,
                             output_dim=self.units_per_block)
                       for _ in range(num_blocks)]
        for i in range(len(self.blocks)):
            self.add_module('block_{}'.format(i), self.blocks[i])

    def forward(self, x):
        acts = [block(x)[0] for block in self.blocks]
        return torch.cat(acts, dim=-1)

"""

Latch:
 v = [012]

Units:
 (a) (b) (c) (d)

v[0] = w_{a}{v[0]} * a + w_{b}{v[0]} * b + w_{c}{v[0]} * c + w_{d}{v[0]} * d
v[1] = w_{a}{v[1]} * a + w_{b}{v[1]} * b + w_{c}{v[1]} * c + w_{d}{v[1]} * d
...
v[n] = w_{a}{v[n]} * a + w_{b}{v[n]} * b + w_{c}{v[n]} * c + w_{d}{v[n]} * d

==> self.value(self.units(x))

"""
