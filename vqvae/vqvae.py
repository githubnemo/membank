

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        self.l0 = nn.Linear(input_size, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        l0 = self.l0(x)
        l0 = self.act(l0)
        l1 = self.l1(l0)
        l1 = self.act(l1)
        return l1


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        self.l0 = nn.Linear(hidden_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        l0 = self.l0(x)
        l0 = self.act(l0)
        l1 = self.l1(l0)
        l1 = self.act(l1)
        return l1


class Discretizer:
    def __init__(self, num_latents, hidden_dim):
        self.num_latents = num_latents
        self.hidden_dim = hidden_dim
        self.e = nn.Parameter(torch.zeros((num_latents, hidden_dim)))

    def distance(self, a, b):
        return ((a - b)**2).sum(dim=-1).sqrt()

    def forward(self, z_e):
        # compute min distance between z_e and self.e for each row in z_e,
        # return matrix with (_, e_{best match for each row})

        # to find the embedding row of self.e that matches best with the input
        # for each row in the input z_e, we first have to align both tensors.
        z_e_exp = z_e.repeat(1, self.e.size(0)).view(z_e.size(0) * self.e.size(0), -1)
        e_exp = self.e.repeat(z_e.size(0), 1)
        d = self.distance(z_e_exp, e_exp)

        d = d.view(z_e.size(0), self.e.size(0))

        _min_dists, idx = d.min(-1)
        print(_min_dists)

        return self.e[idx]



class VQVAE(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_latents):
            self.enc = Encoder(input_dim, hidden_dim)
            self.dis = Discretizer(num_latents, hidden_dim)
            self.dec = Decoder(hidden_dim, input_size)

        def forward(x):
            z_e = self.enc(x)
            e_k = self.dis(z_e)
            z_q = self.dec(e_k)
            return z_q, z_e, e_k



class Trainer(skorch.NeuralNet):
        def __init__(self, beta=0.25, criterion='MSELoss'):
            self.beta = beta

            super().__init__(criterion=criterion)

        def train_step(self, X, y, optimizer):
            optimizer.zero_grad()

            z_q, z_e, e_k = self.infer(X)

            def l2(x):
                return (x**2).sum(dim=-1).sqrt()

            L = self.criterion(z_q, y)
            L += l2(Variable(z_e) - e_k)
            L += self.beta * l2(z_e - Variable(e_k))

            L.backward(retain_graph=True)

            # TODO: is this really what they meant?
            z_e.grad = e_k.grad

            optimizer.step()
            return L



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)



ef = Trainer(
    model=VQVAE,
    model__input_dim=28 * 28,
    model__hidden_dim=128,
    model__num_latents=10,

    iterator_train=train_loader,
    train_split=None,
)

ef.fit(np.zeros(1), None)
