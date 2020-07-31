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
parser.add_argument("-o", "--outfile", default=None,
                    help="File to store the trained model in [suffix: .pt or .pth]")
parser.add_argument("--load-model", default=None,
                    help="Load model instead of training")

parser.add_argument("--invars", nargs="+", default=["n_jets", "HT", "lead_jet_pt"])
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs. The epoch is defined as a single pass through the smallest dataset.")
parser.add_argument("--clip-grad-value", default=None, type=float)

# Sampling options
parser.add_argument("--batch-size", default=256, type=int,
                    help="Size of the batches sampled from each sample "
                    "(i.e. a single batch in training is twice this size)")

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
    df["HT_tau"] = df.HT + df.tau_pt

# Pseudo-datasets
df_p0 = pd.concat([df_data_nn, df_non_ttbar_nn])
df_p1 = df_ttbar_nn

invars = args.invars

# To scale inputs by (x - median) / IQR
offset = df_p1[invars].quantile(0.5).values.astype(np.float32)
scale = (df_p1[invars].quantile(0.75) - df_p1[invars].quantile(0.25)).values.astype(np.float32)

# Some pytorch action!
import torch
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader

from ttbar_reweighting import ReweightingNet, train

# Reproducibility
torch.manual_seed(0)

# Input variables
X0 = torch.Tensor(df_p0[invars].values).float()
X1 = torch.Tensor(df_p1[invars].values).float()

# Transform inputs
X0 = (X0 - offset) / scale
X1 = (X1 - offset) / scale

# Weights
W0 = torch.Tensor(df_p0.weight.values[:, np.newaxis]).float()
W1 = torch.Tensor(df_p1.weight.values[:, np.newaxis]).float()

# Batches will be sampled by probability given by abs(weight)
# Therefore only the sign of the weight needs to be kept
dataset0 = TensorDataset(X0, torch.sign(W0))
dataset1 = TensorDataset(X1, torch.sign(W1))

min_dataset_len = min(len(dataset0), len(dataset1))

sampler0 = WeightedRandomSampler(torch.abs(W0.view(-1)), min_dataset_len, replacement=True)
sampler1 = WeightedRandomSampler(torch.abs(W1.view(-1)), min_dataset_len, replacement=True)

loader0 = DataLoader(dataset0, batch_size=args.batch_size, sampler=sampler0, num_workers=1)
loader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1, num_workers=1)

net = ReweightingNet(len(invars), leak=0.01)

if not args.load_model:
    train(net, loader0, loader1, epochs=args.epochs, clip_grad_value=args.clip_grad_value)

    # Saving the model for python
    if args.outfile:
        torch.save(net.state_dict(), args.outfile)
else:
    net.load_state_dict(torch.load(args.load_model))

# Turn on evaluation mode
net.eval()

# Add scale factor to ttbar dataframe
X = torch.tensor((df_ttbar_nn[invars].values - offset) / scale, dtype=torch.float)
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

# Plot only ID region
# Have to add tauSF back in (removed previously)
df_ttbar.weight *= df_ttbar.tauSF
df_non_ttbar.weight *= df_non_ttbar.tauSF

plotter = Plotter(df_data, df_ttbar, df_non_ttbar, sel=lambda df: df.tau_loose)
for args in default_plots:
    fig = plotter.plot(*args[1:], "sf_nn")
    fig.savefig("{}_sf_nn_{}.pdf".format(args[0], "idregion"))
    plt.close(fig)

del plotter
