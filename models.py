import skorch
import torch
import torch.nn as nn

from torch.autograd import Variable


""" Models incapable of input-driven update rates """

class ClockingCWRNN(nn.Module):
    """Simple clocking CWRNN using free parameters for sine wave period/shift
    """
    def __init__(self, input_dim, output_dim, num_modules, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modules = num_modules

        self.input_mod = nn.Linear(input_dim, output_dim)
        self.hidden_mod = nn.Linear(output_dim, output_dim, bias=False)

        self.module_periods = nn.Parameter(torch.zeros(num_modules) + 1)
        self.module_shifts = nn.Parameter(torch.zeros(num_modules))

        self.f_mod = nn.Tanh()

    def step(self, ti, xi, h):
        module_size = self.output_dim // self.num_modules

        acts = self.f_mod(self.input_mod(xi) + self.hidden_mod(h))
        module_acts = acts.view(-1, self.num_modules, module_size)

        # y=(sin(x)+1)/2 so we y is in [0;1]
        gate = (torch.sin(ti * self.module_periods + self.module_shifts) + 1) / 2
        gate = gate.view(1, -1, 1).expand_as(module_acts).contiguous()
        gate = gate.view(-1, self.output_dim)

        y = (1 - gate) * acts + gate * h

        return y, y

    def init_hidden(self, use_cuda):
        var = Variable(torch.zeros(self.output_dim))
        return var.cuda() if use_cuda else var

    def forward(self, x):
        t = x.size(1)
        ys = []
        h = self.init_hidden(x.is_cuda)
        for ti in range(t):
            xi = x[:, ti]
            yi, h = self.step(ti, xi, h)
            ys.append(yi)
        return torch.stack(ys, dim=1), h


""" Models capable of input-driven update rates"""

class InputDrivenCWRNN(nn.Module):
    """simply learning sine parameters from var(sister module)"""
    def __init__(self, input_dim, output_dim, num_modules, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modules = num_modules

        self.input_mod = nn.Linear(input_dim, output_dim)
        self.hidden_mod = nn.Linear(output_dim, output_dim, bias=False)

        self.input_period = nn.Linear(input_dim, output_dim)
        self.hidden_period = nn.Linear(output_dim, output_dim, bias=False)

        self.module_shifts = nn.Parameter(torch.zeros(num_modules))

        self.f_mod = nn.Tanh()

    def step(self, ti, xi, h):
        module_size = self.output_dim // self.num_modules

        acts = self.f_mod(self.input_mod(xi) + self.hidden_mod(h))
        module_acts = acts.view(-1, self.num_modules, module_size)

        acts_period = self.f_mod(self.input_period(xi) + self.hidden_period(h))
        module_acts_period = acts.view(-1, self.num_modules, module_size)

        # use variance as indicator for surprisal and, hence, update rate
        suprisal = module_acts_period.var(-1, keepdim=True)
        module_periods = surprisal

        # y=(sin(x)+1)/2 so we y is in [0;1]
        gate = (torch.sin(ti * module_periods + self.module_shifts) + 1) / 2
        #gate = gate.view(1, -1, 1).expand_as(module_acts).contiguous()
        gate = gate.view(-1, self.output_dim)

        y = (1 - gate) * acts + gate * h

        return y, y, module_periods

    def init_hidden(self):
        return Variable(torch.zeros(self.output_dim))

    def forward(self, x):
        t = x.size(1)
        ys = []
        h = self.init_hidden()
        ps = []
        for ti in range(t):
            xi = x[:, ti]
            yi, h, p = self.step(ti, xi, h)
            ys.append(yi)
            ps.append(p.squeeze(-1))
        return torch.stack(ys, dim=1), h, torch.stack(ps, dim=1)



class SurprisalCWRNN(nn.Module):
    """Surprisal through reconstruction and measuring resulting entropy.
    Inspired by Rocki (2016)
    """
    def __init__(self, input_dim, output_dim, num_modules, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modules = num_modules

        self.input_mod = nn.Linear(input_dim, output_dim)
        self.hidden_mod = nn.Linear(output_dim, output_dim, bias=False)

        self.input_rec = nn.Linear(input_dim, input_dim * num_modules)
        self.hidden_rec = nn.Linear(input_dim * num_modules, input_dim * num_modules, bias=False)

        self.module_periods = nn.Parameter(torch.zeros(num_modules) + 1)
        self.module_shifts = nn.Parameter(torch.zeros(num_modules))

        self.f_mod = nn.Tanh()

        self.log_softmax = nn.LogSoftmax()

    def step(self, ti, xi, h, x_pred):
        module_size = self.output_dim // self.num_modules

        acts = self.f_mod(self.input_mod(xi) + self.hidden_mod(h))
        module_acts = acts.view(-1, self.num_modules, module_size)

        # predict x_{t+1}, store for next step
        acts_rec = self.f_mod(self.input_rec(xi) + self.hidden_rec(x_pred))
        module_acts_rec = acts_rec.view(-1, self.num_modules, self.input_dim)

        # compare last prediction with current x, compute surprisal from that
        module_x_pred = x_pred.view(-1, self.num_modules, self.input_dim)
        surprisal = self.log_softmax(module_x_pred) * xi.unsqueeze(1).expand_as(module_acts_rec)
        # compute the mean surprisal for each module
        mean_surprisal = surprisal.mean(-1)
        # use mean surprisal to drive the periodicity of the module's gate
        # weight down surprisal for specific modules
        module_periods = mean_surprisal * self.module_periods

        # y=(sin(x)+1)/2 so we y is in [0;1]
        gate = (torch.sin(ti * module_periods + self.module_shifts) + 1) / 2
        gate = gate.unsqueeze(-1).expand_as(module_acts).contiguous()
        gate = gate.view(-1, self.output_dim)

        y = (1 - gate) * acts + gate * h

        return y, y, acts_rec, module_periods

    def init_hidden(self, use_cuda):
        var = Variable(torch.zeros(self.output_dim))
        return var.cuda() if use_cuda else var

    def forward(self, x):
        t = x.size(1)
        ys = []
        h = self.init_hidden(x.is_cuda)
        x_pred = Variable(torch.zeros(self.input_dim * self.num_modules))
        x_pred = x_pred.cuda() if x.is_cuda else x_pred
        ps = []
        for ti in range(t):
            xi = x[:, ti]
            yi, h, x_pred, p = self.step(ti, xi, h, x_pred)
            ys.append(yi)
            ps.append(p.squeeze(-1))
        return torch.stack(ys, dim=1), h, torch.stack(ps, dim=1)
