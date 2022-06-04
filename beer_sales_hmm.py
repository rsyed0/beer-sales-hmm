import torch
from torch import nn

import pandas as pd
import numpy as np

# TODO add month as a feature
# TODO implement >1 stride lengths

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO decide where to put these
mn_norm_data_val, mx_norm_data_val = 0, 1
n_bins = 10
norm_data_range = mx_norm_data_val - mn_norm_data_val
bin_width = norm_data_range / n_bins

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, id_col, target_col, window_size, batch_size):
        self.data = pd.read_csv(csv_file)

        # TODO decide how this can/should be used...
        self.id = id_col

        self.target = target_col
        self.window_size = window_size
        self.batch_size = batch_size

        # normalize input data
        # TODO integrate mn/mx_norm_data_val into this

        target_data = torch.from_numpy(self.data[self.target].values)
        mn_target_data, mx_target_data = torch.aminmax(target_data)
        self.norm_target_data = np.array([(x - mn_target_data) / (mx_target_data - mn_target_data) for x in target_data])


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

        # TODO implement biases for each layer in addition to weights

        # TODO decide how to handle input distribution nodes
        # use normal distro by default?

        # dimensions: [width, n_bins]
        if pr_i is None:
            pr_i = torch.randn(width, n_bins)
            pr_i = nn.functional.softmax(pr_i, dim=1)

        self.input_distros = nn.Parameter(pr_i, requires_grad=True)

        # dimensions: [length-1, width, width]
        # TODO use nn.Linear layer instead?
        # TODO use softmax, ensure all weights per-node sum to 1
        dense_layer_weights = torch.randn(length-1, width, width)
        self.dense_layer_weights = nn.Parameter(dense_layer_weights, requires_grad=True)


    def forward(self, data):
        batch_size, window_size = data.shape
        print("Data: "+str(data))
        #print(type(data))

        assert window_size == self.length

        # using log-domain
        log_leafs = nn.functional.log_softmax(self.input_distros, dim=-1)

        y = torch.zeros(batch_size, 1).to(data.device)

        for window_i in range(batch_size):
            window = data[window_i] #.cpu().detach().numpy()

            x = torch.Tensor([log_leafs[distr_i][min(n_bins-1, int(window[0] // bin_width))] for distr_i in range(self.width)])

            for layer_i in range(self.length-1):
                
                # shape: [width, width]
                log_dense_weights = nn.functional.log_softmax(self.dense_layer_weights[layer_i], dim=-1)

                # apply dense weights
                # feed through sum layer

                # multiply by log(dense_weights)
                x = x + log_dense_weights

                x = torch.logsumexp(x, dim=1)

                # feed through product layer
                log_pr_x = torch.Tensor([log_leafs[distr_i][min(n_bins-1, int(window[layer_i+1] // bin_width))] for distr_i in range(self.width)])
                x = x + log_pr_x

            print("X: "+str(x))
            y[window_i] = torch.logsumexp(x, dim=0)

        y = torch.exp(y)
        print("Y: "+str(y))
        return y


def main():
    window_size = 12
    learning_rate = 0.1
    batch_size = 1
    width = 5
    n_epochs = 5

    alcohol_ds = CustomDataset("Alcohol_Sales.csv", 'DATE', 'S4248SM144NCEN', window_size, batch_size)

    # TODO decide how/if to initialize pr_i
    model = HMM(width, window_size)
    model.to(device)

    lls = []
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch_i in range(n_epochs):
        epoch_lls = []

        # run/train fixed context window over one batch
        for batch, _ in alcohol_ds:
            optim.zero_grad()

            batch = batch.to(device)
            batch_lls = model(batch)

            print("Batch LLs: "+str(batch_lls))

            loss = -torch.sum(batch_lls, dim=-1)
            loss.requires_grad = True

            # TODO debug this, gives same output/loss on each epoch/batch
            loss.backward()
            optim.step()

            print(batch_lls.tolist()[0])
            epoch_lls.extend(torch.flatten(batch_lls).tolist())

        lls.append(epoch_lls)

    lls = torch.Tensor(lls)
    print(lls)
    model.to('cpu')
    ll = torch.mean(lls)
    print(ll)

if __name__ == "__main__":
    main()