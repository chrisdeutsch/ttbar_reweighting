#!/usr/bin/env python
import argparse
import logging
import os
import sys
from glob import glob

from ttbar_reweighting import get_dataframe, remove_duplicates, apply_selection, is_hf

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("reweight_nn.py")

parser = argparse.ArgumentParser()
parser.add_argument("ntuple_dir")
args = parser.parse_args()


# Load ntuples
dfs = {}
for fn in glob(os.path.join(args.ntuple_dir, "*.root")):
    sample = os.path.basename(fn).replace(".root", "")
    log.info("Found {} in ntuple directory".format(sample))
    dfs[sample] = get_dataframe(fn)

# Remove duplicates before applying selection
for sample in dfs:
    df, ndupes = remove_duplicates(dfs[sample])
    log.info("Removing {} duplicates in {}".format(ndupes, sample))
    dfs[sample] = df

# Apply selection
for sample in dfs:
    dfs[sample] = apply_selection(dfs[sample])

# Z+HF scale factor
for sample in ["Zee", "Zmumu", "Ztautau"]:
    df = dfs[sample]
    df.loc[is_hf(df), "weight"] *= 1.3

# Remove tau scale factor
for sample in dfs:
    if sample == "data":
        continue

    df = dfs[sample]
    df.weight /= df.tauSF

# Finalize dataframes
df_data = dfs["data"]
df_ttbar = dfs["ttbarIncl"]
df_non_ttbar = pd.concat([dfs[sample] for sample in dfs if sample !="data" and sample != "ttbarIncl"])

# Want to estimate the density ratio: f_0(x) / f_1(x) = f_(data - non_ttbar)(x) / f_(ttbar)(x)
# Therefore need to build pseudodataset from data - non_ttbar
# First augment with additional info for training
df_ttbar_nn = df_ttbar.copy()
df_data_nn = df_data.copy()
df_non_ttbar_nn = df_non_ttbar.copy()
df_non_ttbar_nn.weight *= -1.0 # Need to subtract non ttbar from data

for df in [df_data_nn, df_non_ttbar_nn, df_ttbar_nn]:
    df["njets_trafo"] = (np.clip(df.n_jets, 2, 10) - 2) / 8.0

    q_lo, med, q_hi = df_ttbar_nn.HT.quantile([0.25, 0.5, 0.75])
    df["HT_trafo"] = (df.HT - med) / (q_hi - q_lo)

    q_lo, med, q_hi = df_ttbar_nn.lead_jet_pt.quantile([0.25, 0.5, 0.75])
    df["lead_jet_pt_trafo"] = (df.lead_jet_pt - med) / (q_hi - q_lo)

# Pseudo-datasets
df_p0 = pd.concat([df_data_nn, df_non_ttbar_nn])
df_p1 = df_ttbar_nn

# Definitions for training
invars = ["njets_trafo", "HT_trafo", "lead_jet_pt_trafo"]
epochs = 2
batch_size = 256


# Some pytorch action!
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader

from ttbar_reweighting import ReweightingNet

# Reproducibility
torch.manual_seed(0)

# Input variables
X0 = torch.Tensor(df_p0[invars].values).float()
X1 = torch.Tensor(df_p1[invars].values).float()

# Weights
W0 = torch.Tensor(df_p0.weight.values[:, np.newaxis]).float()
W1 = torch.Tensor(df_p1.weight.values[:, np.newaxis]).float()

# Batches will be sampled by probability given by abs(weight)
# Therefore only the sign of the weight needs to be kept
dataset0 = TensorDataset(X0, torch.sign(W0))
dataset1 = TensorDataset(X1, torch.sign(W1))

min_dataset_len = min(len(dataset0), len(dataset1))

# Replacement=false?
sampler0 = WeightedRandomSampler(torch.abs(W0.view(-1)), min_dataset_len)
sampler1 = WeightedRandomSampler(torch.abs(W1.view(-1)), min_dataset_len)

loader0 = DataLoader(dataset0, batch_size=batch_size, sampler=sampler0, num_workers=1)
loader1 = DataLoader(dataset1, batch_size=batch_size, sampler=sampler1, num_workers=1)

net = ReweightingNet(len(invars))
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda i: 0.95)

for epoch in range(epochs):
    running_loss = 0.0

    for i, (data0, data1) in enumerate(zip(loader0, loader1)):
        if i % 1000 == 999:
            log.info("[{} {}]: {:.5f}".format(epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

        x0, w0 = data0
        x1, w1 = data1

        if torch.sum(w0) <= 0 or torch.sum(w1) <= 0:
            log.warning("Skipping bad batch...")
            log.warning("sum(w0) = {}".format(torch.sum(w0).item()))
            log.warning("sum(w1) = {}".format(torch.sum(w1).item()))
            log.warning("This should not occur often. If it does it will likely introduce a bias.")
            continue

        optimizer.zero_grad()

        pred0, pred1 = net(x0), net(x1)

        loss = torch.sum(w0 / torch.sqrt(torch.exp(pred0))) / torch.sum(w0) \
             + torch.sum(w1 * torch.sqrt(torch.exp(pred1))) / torch.sum(w1)

        if torch.isnan(loss) or torch.isinf(loss):
            log.error("Loss is nan or inf. Aborting...")
            sys.exit(1)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Print out learning rate to get a feeling for the scheduler
    params, = optimizer.param_groups
    log.info("Epoch finished. Current LR: {}".format(params["lr"]))


# Add scale factor to ttbar dataframe
X = torch.tensor(df_ttbar_nn[invars].values, dtype=torch.float)
pred = net(X).detach().numpy()
df_ttbar["sf_nn"] = np.exp(pred) * (torch.sum(W0) / torch.sum(W1)).item()

# Plot this puppy
import matplotlib.pyplot as plt
from ttbar_reweighting import Plotter, default_plots

plotter = Plotter(df_data, df_ttbar, df_non_ttbar)

for args in default_plots:
    fig = plotter.plot(*args[1:], "sf_nn")
    fig.savefig("{}_sf_nn.pdf".format(args[0]))
    plt.close(fig)
