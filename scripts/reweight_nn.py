#!/usr/bin/env python
import argparse
import logging
import os
import sys
from glob import glob

from ttbar_reweighting import get_dataframe, remove_duplicates, apply_selection, is_hf
from ttbar_reweighting import Plotter, default_plots, plot_loss

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("ntuple_dir")
parser.add_argument("--outfile-model", default=None,
                    help="File to store the trained model in [extension: .pt or .pth]")
parser.add_argument("--outfile-preprocessing", default=None,
                    help="File to store the preprocessing factors [extension: .npz]")
parser.add_argument("--load-preprocessing", default=None,
                    help="Load preprocessing instead of deriving")
parser.add_argument("--load-model", default=None,
                    help="Load model instead of training")

parser.add_argument("--invars", nargs="+", default=["n_jets", "HT", "lead_jet_pt"])
parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs. The epoch is defined as a single pass through the smallest dataset.")
parser.add_argument("--clip-grad-value", default=None, type=float)
parser.add_argument("--weight-decay", default=0, type=float)

# Sampling options
parser.add_argument("--batch-size", default=256, type=int,
                    help="Size of the batches sampled from each sample "
                    "(i.e. a single batch in training is twice this size)")

parser.add_argument("--debug", action="store_true")

# Fold options
parser.add_argument("--n-folds", default=None, type=int,
                    help="Number of folds to use for training / testing.")
parser.add_argument("--fold", default=0, type=int,
                    help="Fold to use for testing (training fold: event_number % n_folds != fold)")

# Network options
parser.add_argument("--layers", nargs="+", default=[32, 32, 32, 32, 32], type=int,
                    help="Number of nodes in the hidden layers of the network")

args = parser.parse_args()


level = logging.INFO
if args.debug:
    level = logging.DEBUG

logging.basicConfig(level=level)
log = logging.getLogger("reweight_nn.py")


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

# Add new variables
for df in [df_data, df_ttbar, df_non_ttbar]:
    df["HT_tau"] = df.HT + df.tau_pt

# Want to estimate the density ratio: f_0(x) / f_1(x) = f_(data - non_ttbar)(x) / f_(ttbar)(x)
# Therefore need to build pseudodataset from data - non_ttbar
# First augment with additional info for training
df_ttbar_nn = df_ttbar.copy()
df_data_nn = df_data.copy()
df_non_ttbar_nn = df_non_ttbar.copy()
df_non_ttbar_nn.weight *= -1.0 # Need to subtract non ttbar from data

# Pseudo-datasets
df_p0 = pd.concat([df_data_nn, df_non_ttbar_nn])
df_p1 = df_ttbar_nn

invars = args.invars

# Preprocessing
# To scale inputs by (x - median) / IQR and apply the top cutoff (given by denominator of density ratio)
if args.load_preprocessing:
    log.info("Loading preprocessing from " + repr(args.load_preprocessing))
    f = np.load(args.load_preprocessing)
    offset = f["offset"]
    scale = f["scale"]
    cutoff = f["cutoff"]
    norm_factor = f["norm_factor"]
else:
    offset = df_p1[invars].quantile(0.5).values.astype(np.float32)
    scale = (df_p1[invars].quantile(0.75) - df_p1[invars].quantile(0.25)).values.astype(np.float32)
    cutoff = df_p1[invars].quantile(0.99).values.astype(np.float32)
    norm_factor = df_p0.weight.sum() / df_p1.weight.sum()

# Save preprocessing
if args.outfile_preprocessing:
    log.info("Saving preprocessing in " + repr(args.outfile_preprocessing))
    with open(args.outfile_preprocessing, "wb") as f:
        np.savez(f, offset=offset, scale=scale, cutoff=cutoff, norm_factor=norm_factor)

# Some pytorch action!
import torch
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader

import ttbar_reweighting.nn
from ttbar_reweighting import ReweightingNet, train

# To find weird things happening
if args.debug:
    log.info("Setting autograd anomaly detection")
    torch.autograd.set_detect_anomaly(True)
    ttbar_reweighting.nn.log.setLevel(logging.DEBUG)

# Reproducibility
torch.manual_seed(0)

# Fold selection
sel_train0 = None
sel_train1 = None

sel_test0 = None
sel_test1 = None

if args.n_folds:
    assert args.fold < args.n_folds

    log.info("Training with cross validation. Selecting fold {} out of {}".format(args.fold, args.n_folds))
    sel_train0 = df_p0.event_number % args.n_folds != args.fold
    sel_train1 = df_p1.event_number % args.n_folds != args.fold
else:
    log.info("Training without cross validation")
    sel_train0 = np.ones(len(df_p0), dtype=np.bool)
    sel_train1 = np.ones(len(df_p1), dtype=np.bool)

sel_test0 = ~sel_train0
sel_test1 = ~sel_train1


# Input variables
X0_train = torch.Tensor(df_p0.loc[sel_train0, invars].values).float()
X1_train = torch.Tensor(df_p1.loc[sel_train1, invars].values).float()

X0_test = torch.Tensor(df_p0.loc[sel_test0, invars].values).float()
X1_test = torch.Tensor(df_p1.loc[sel_test1, invars].values).float()

# Transform inputs
X0_train = torch.min(X0_train, torch.Tensor(cutoff))
X0_train = (X0_train - offset) / scale

X1_train = torch.min(X1_train, torch.Tensor(cutoff))
X1_train = (X1_train - offset) / scale

if args.n_folds:
    X0_test = torch.min(X0_test, torch.Tensor(cutoff))
    X0_test = (X0_test - offset) / scale

    X1_test = torch.min(X1_test, torch.Tensor(cutoff))
    X1_test = (X1_test - offset) / scale

# Weights
W0_train = torch.Tensor(df_p0.loc[sel_train0, "weight"].values[:, np.newaxis]).float()
W1_train = torch.Tensor(df_p1.loc[sel_train1, "weight"].values[:, np.newaxis]).float()

W0_test = torch.Tensor(df_p0.loc[sel_test0, "weight"].values[:, np.newaxis]).float()
W1_test = torch.Tensor(df_p1.loc[sel_test1, "weight"].values[:, np.newaxis]).float()

# Batches will be sampled by probability given by abs(weight)
# Therefore only the sign of the weight needs to be kept
dataset0 = TensorDataset(X0_train, torch.sign(W0_train))
dataset1 = TensorDataset(X1_train, torch.sign(W1_train))

min_dataset_len = min(len(dataset0), len(dataset1))

sampler0 = WeightedRandomSampler(torch.abs(W0_train.view(-1)), min_dataset_len, replacement=True)
sampler1 = WeightedRandomSampler(torch.abs(W1_train.view(-1)), min_dataset_len, replacement=True)

loader0 = DataLoader(dataset0, batch_size=args.batch_size, sampler=sampler0, num_workers=1)
loader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1, num_workers=1)

log.info("Setting up network for {} input variables".format(len(invars)))
log.info("Hidden layers: " + repr(args.layers))

net = ReweightingNet(len(invars),
                     hidden_layers=args.layers,
                     leak=0.1)

if not args.load_model:
    test_monitor = None
    if args.n_folds:
        test_monitor = (X0_test, W0_test, X1_test, W1_test)

    loss_train, loss_test = \
        train(net, loader0, loader1,
              epochs=args.epochs,
              clip_grad_value=args.clip_grad_value,
              weight_decay=args.weight_decay,
              train_monitor=(X0_train, W0_train, X1_train, W1_train),
              test_monitor=test_monitor
        )

    plot_loss(loss_train, loss_test)
    log.info("Training loss: " + repr(loss_train))
    log.info("Testing loss: " + repr(loss_test))

    # Saving the model for python
    if args.outfile_model:
        log.info("Saving model in " + repr(args.outfile_model))
        torch.save(net.state_dict(), args.outfile_model)
else:
    log.info("Loading model from " + repr(args.load_model))
    net.load_state_dict(torch.load(args.load_model))

# Turn on evaluation mode
net.eval()

with torch.no_grad():
    # Add NN to dataframes
    for df in [df_data, df_ttbar, df_non_ttbar, df_p0, df_p1]:
        X = torch.Tensor(df[invars].values).float()
        X = torch.min(X, torch.Tensor(cutoff))
        X = (X - offset) / scale
        pred = net(X).detach().numpy()
        df["nn"] = pred

# Add scale factor to ttbar dataframe
df_ttbar["sf_nn"] = norm_factor * np.exp(df_ttbar.nn)

log.info("Normalization factor: {:.5f}".format(norm_factor))

# Plot this puppy
import matplotlib.pyplot as plt


plotter = Plotter(df_data, df_ttbar, df_non_ttbar)

for args in default_plots:
    fig = plotter.plot(*args[1:], "sf_nn")
    fig.savefig("{}_sf_nn.pdf".format(args[0]))
    plt.close(fig)

del plotter

# Split in 1-prong / 3-prong
for prong in [1, 3]:
    plotter = Plotter(df_data, df_ttbar, df_non_ttbar, sel=lambda df: df.tau_prong == prong)
    for args in default_plots:
        if args[0] != "taupt":
            continue

        fig = plotter.plot(*args[1:], "sf_nn")
        fig.savefig("{}_{}p_sf_nn.pdf".format(args[0], prong))

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
