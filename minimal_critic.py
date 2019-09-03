# -*- coding: utf-8 -*-

"""
Minimal example of a basic classifier with a tiny critic and multiplexed output
"""

import torch
from torch.nn.functional import binary_cross_entropy


def mplex(out, mplex):
    # mplex == 1 => keep output value
    # mplex == 0 => invert output value
    return mplex * out + (1 - mplex) * (1 - out)


class SimpleMultiplex(torch.nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        # Basic classifier network
        # (hidden layers are superfluous for this simple data)
        self.h1 = torch.nn.Linear(data_dim, 10)
        self.h2 = torch.nn.Linear(10, 5)
        self.out = torch.nn.Linear(5, 1)
        # Multiplexed critic
        self.mplex = torch.nn.Linear(5, 1)

    def forward(self, x):
        """basic feedforward network (w/o critic network)"""
        h2 = torch.relu(self.h2(torch.relu(self.h1(x))))
        return torch.sigmoid(self.out(h2))

    def forward_multiplex_value(self, x):
        """feedforward to embedding to multiplex bit"""
        h2 = torch.relu(self.h2(torch.relu(self.h1(x)))).detach()
        return torch.sigmoid(self.mplex(h2))

    def forward_multiplex_out(self, x):
        """apply the multiplexing bit to reverse (or not) the classifier output"""
        mplex_val = self.forward_multiplex_value(x)
        h2 = torch.relu(self.h2(torch.relu(self.h1(x))))
        out = torch.sigmoid(self.out(h2)).detach()
        return mplex(out, mplex_val)


def gen_features(n_samples, data_dim, separation):
    """helper function to generate features as data_dim sized multivariate gaussian mixture"""
    half = round(n_samples / 2)
    class1 = torch.randn([half, data_dim]) - (separation / 2) * torch.ones(data_dim)
    class2 = torch.randn([half, data_dim]) + (separation / 2) * torch.ones(data_dim)
    features = torch.cat([class1, class2])
    return features


def gen_data(n_samples, data_dim, separation):
    """generate initial training data"""
    features = gen_features(n_samples, data_dim, separation)
    half = round(n_samples / 2)
    labels = torch.cat([torch.zeros(half), torch.ones(half)])
    data_combined = torch.cat([features, labels.reshape([n_samples, 1])], dim=1)
    data_loader = torch.utils.data.DataLoader(data_combined, batch_size=64)
    return features, labels, data_loader


def gen_crit_data(n_samples, data_dim, separation, crit_gen=torch.zeros):
    """generate data to train multiplexed critic network"""
    features = gen_features(n_samples, data_dim, separation)
    labels = crit_gen(n_samples)  # zeros = flip values, ones = keep values
    data_combined = torch.cat([features, labels.reshape([n_samples, 1])], dim=1)
    data_loader = torch.utils.data.DataLoader(data_combined, batch_size=64)
    return features, labels, data_loader


def train(epoch, model, optimizer, data_loader, device, data_dim):
    """train the classifier"""
    loss = 0
    for _, data in enumerate(data_loader):
        data = data.to(device)
        features_batch = data[:, 0:data_dim]
        labels_batch = data[:, data_dim]
        optimizer.zero_grad()
        pred = model(features_batch)
        loss = binary_cross_entropy(pred.squeeze(), labels_batch)
        loss.backward()
        optimizer.step()
    return loss


def train_critic(epoch, model, optimizer, data_loader, device, data_dim):
    """train the critic"""
    loss = 0
    for _, data in enumerate(data_loader):
        data = data.to(device)
        features_batch = data[:, 0:data_dim]
        mplex_batch = data[:, data_dim]
        optimizer.zero_grad()
        mplex = model.forward_multiplex_value(features_batch)
        loss = binary_cross_entropy(mplex.squeeze(), mplex_batch)
        loss.backward()
        optimizer.step()
    return loss


def report(model, data_dim):
    """report model outputs"""
    print("\nClassifier")
    print("class 0 region: %s" % model(-torch.ones(data_dim)))
    print("boundary: %s" % model(torch.zeros(data_dim)))
    print("class 1 region: %s" % model(torch.ones(data_dim)))
    print("\nCritic")
    print("class 0 region: %s" % model.forward_multiplex_out(-torch.ones(data_dim)))
    print("boundary: %s" % model.forward_multiplex_out(torch.zeros(data_dim)))
    print("class 1 region: %s" % model.forward_multiplex_out(torch.ones(data_dim)))


def main():

    data_dim = 20
    separation = 1

    # optimizer state
    model = SimpleMultiplex(data_dim)
    model.train()
    device = "cpu"

    print("\nTest ========")

    # train core model
    print("\nModel Training ----\n")
    _, _, data_loader = gen_data(3000, data_dim, separation)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(30):
        loss = train(epoch, model, optimizer, data_loader, device, data_dim)
    print("Loss: %s" % loss)
    report(model, data_dim)

    # train critic - all observations wrong
    print("\nModel Criticism (all wrong) ----\n")
    _, _, data_loader_crit = gen_crit_data(
        3000, data_dim, separation, crit_gen=torch.zeros
    )
    # note fresh optimizer prevents .step() from perturbing state of core network
    # in spite of .zero_grad() being called prior to .step()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(30):
        loss = train_critic(epoch, model, optimizer, data_loader_crit, device, data_dim)
    print("Loss: %s" % loss)
    report(model, data_dim)

    # train critic - all observations correct
    _, _, data_loader_crit = gen_crit_data(
        3000, data_dim, separation, crit_gen=torch.ones
    )
    print("\nModel Criticism (all correct) ----\n")
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(30):
        loss = train_critic(epoch, model, optimizer, data_loader_crit, device, data_dim)
    print("Loss: %s" % loss)
    report(model, data_dim)

    print("\nDone")


if __name__ == "__main__":
    main()
