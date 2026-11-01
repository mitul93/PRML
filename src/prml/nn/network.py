import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        learning_rate=0.1,
    ):
        super().__init__()

        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sequential.html
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        ## This is possible as well. use model[:2] to get hidden layer output
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, output_dim),
        # )

        self.output = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

    def forward(self, x):
        return self.output(self.hidden(x))

    def hidden_outputs(self, x):
        self.eval()

        with torch.no_grad():
            return self.hidden(x)

    def loss(self, x, t):
        return self.loss_fn(self(x), t)

    def fit(self, x, t, n_iter=1000):
        self.train()
        losses = []

        for _ in range(n_iter):
            self.optimizer.zero_grad()

            loss = self.loss(x, t)
            loss.backward()

            self.optimizer.step()
            losses.append(loss.item())

        return losses

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            return self(x)
