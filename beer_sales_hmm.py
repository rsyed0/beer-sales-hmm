import torch
from torch import nn

import pandas as pd
import numpy as np

import sys

# TODO add month as a feature
# TODO implement >1 stride lengths
# TODO model the residuals and subtract assumed growth

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mn_norm_data_val, mx_norm_data_val = 0, 1
n_bins = 10
norm_data_range = mx_norm_data_val - mn_norm_data_val
bin_width = norm_data_range / n_bins

# Can try dataset that doesn't need to be normalized
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, id_col, target_col, window_size, batch_size):
        self.data = pd.read_csv(csv_file)

        # TODO decide how this can/should be used...
        self.id = id_col

        self.target = target_col
        self.window_size = window_size
        self.batch_size = batch_size

        # normalize input data to [0,1]
        target_data = torch.from_numpy(self.data[self.target].values)
        mn_target_data, mx_target_data = torch.aminmax(target_data)
        self.norm_target_data = np.array([(x - mn_target_data) / (mx_target_data - mn_target_data) for x in target_data])
        self.mn_target_data, self.mx_target_data = mn_target_data, mx_target_data


    def __len__(self):
        return (len(self.data[self.target])-self.window_size) // self.batch_size


    def __getitem__(self, idx):
        # shape: [batch_size, window_size]
        max_i = len(self.norm_target_data)-self.window_size
        window = torch.Tensor([self.norm_target_data[i:i+self.window_size] for i in range(idx*self.batch_size, min(max_i, (idx+1)*self.batch_size))])

        # shape: [batch_size]
        # TODO remove this, not predicting directly
        label = torch.Tensor([self.norm_target_data[i+self.window_size] for i in range(idx*self.batch_size, (idx+1)*self.batch_size)])

        #print(idx, window.shape)
        return window, label


class HMM(torch.nn.Module):
    def __init__(self, width, length, pr_i=None):
        super().__init__()

        self.width, self.length = width, length
        assert self.width >= 1 and self.length >= 1

        # dimensions: [width, n_bins]
        if pr_i is None:
            pr_i = torch.randn(width, n_bins)
        self.input_distros = nn.Parameter(pr_i, requires_grad=True)

        # dimensions: [length-1, width, width]
        dense_layer_weights = torch.randn(length-1, width, width)
        self.dense_layer_weights = nn.Parameter(dense_layer_weights, requires_grad=True)

        # dimensions: [width]
        final_layer_weights = torch.randn(width)
        self.final_layer_weights = nn.Parameter(final_layer_weights, requires_grad=True)


    def forward(self, data):
        batch_size, window_size = data.shape
        assert window_size == self.length

        # using log-domain
        log_leafs = nn.functional.log_softmax(self.input_distros, dim=-1)

        # convert normalized data into bin indices
        data.apply_(xtb)

        # get output of first product layer (first set of leaf nodes * [1,1,1,...])
        y = torch.transpose(log_leafs[:, data[:,0].long()], 0, 1)

        # working w 1d values instead of scalars
        for layer_i in range(self.length-1):
            
            # shape: [width, width]
            log_dense_weights = nn.functional.log_softmax(self.dense_layer_weights[layer_i], dim=-1)

            # use broadcasting to compute dense layer weighted sums
            # compute output of sum layer
            sm_y = torch.reshape(y, (batch_size, 1, self.width)) + log_dense_weights
            y = torch.logsumexp(sm_y, dim=-1)

            # get output of leaf/distribution nodes
            # compute output of product layer
            y = y + torch.transpose(log_leafs[:, data[:,layer_i+1].long()], 0, 1)

        log_dense_weights = nn.functional.log_softmax(self.final_layer_weights, dim=0)
        sm_y = y + log_dense_weights
        y = torch.logsumexp(sm_y, dim=1)

        return y # / self.width

def xtb(x):
    return int(min(n_bins-1, x // bin_width))


def main():
    window_size = 2
    learning_rate = 0.05
    batch_size = 5
    width = 5
    n_epochs = 15

    alcohol_ds = CustomDataset("Alcohol_Sales.csv", 'DATE', 'S4248SM144NCEN', window_size, batch_size)

    model = HMM(width, window_size) #Test()
    model.to(device)

    lls = []
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_i in range(n_epochs):
        epoch_lls = []

        # run/train fixed context window over one batch at a time
        # TODO switch to using DataLoader
        for batch, _ in alcohol_ds:
            optim.zero_grad()

            batch = batch.to(device)
            batch_lls = model(batch)

            # compute loss as sum(log(Pr(...))) for each batch
            loss = -torch.sum(batch_lls, dim=-1)
            loss.retain_grad()

            loss.backward()
            optim.step()

            epoch_lls.extend(torch.flatten(batch_lls).tolist())

        lls.append(epoch_lls)

    # TODO fix this
    lls = torch.Tensor(lls)
    print(lls)
    model.to('cpu')
    ll = torch.mean(lls, dim=-1)
    print(ll)
    print(torch.max(lls))

    # sanity check
    # iterate over all values for a small window size, see if sum to 1
    assert window_size == 2

    data = torch.Tensor([[[a/20,b/20] for a in range(1, 21, 2)] for b in range(1, 21, 2)])
    data = torch.flatten(data, end_dim=-2)
    print(data.shape)

    a = torch.Tensor([model(data[i*batch_size:(i+1)*batch_size]).tolist() for i in range(len(data) // batch_size)])

    sm_prob = torch.sum(torch.exp(a))
    print(sm_prob) # should be 1

if __name__ == "__main__":
    main()