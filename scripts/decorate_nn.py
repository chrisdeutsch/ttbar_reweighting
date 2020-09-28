#!/usr/bin/env python
import argparse
import logging
import sys
import os

import numpy as np
import uproot
import torch

from ttbar_reweighting import ReweightingNet


parser = argparse.ArgumentParser()
parser.add_argument("ntuple")
parser.add_argument("outfile")

parser.add_argument("--treename", default="rw_tree")

parser.add_argument("--label", nargs="+", required=True)
parser.add_argument("--model", nargs="+", required=True)
parser.add_argument("--preprocessing", nargs="+", required=True)

parser.add_argument("--variables", nargs="+", required=True)
parser.add_argument("--layers", nargs="+", default=[32, 32, 32])

parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


level = logging.INFO
if args.debug:
    level = logging.DEBUG

logging.basicConfig(level=level)
log = logging.getLogger("decorate_nn.py")


if len(args.label) != len(args.model) or len(args.label) != len(args.preprocessing):
    log.error("Lengths of label / model / preprocessing do not match")
    sys.exit(1)

log.info("Loading tree from: " + repr(args.ntuple))
with uproot.open(args.ntuple) as f:
    to_load = [var for var in args.variables if var != "HT_tau" and var != "HT_tau_met"]
    if "HT" not in to_load:
        to_load.append("HT")
    if "tau_pt" not in to_load:
        to_load.append("tau_pt")
    if "MET" not in to_load:
        to_load.append("MET")

    df = f["Nominal"].pandas.df(to_load)
    df["HT_tau"] = df.HT + df.tau_pt
    df["HT_tau_met"] = df.HT_tau + df.MET

for label, model, preprocessing in zip(args.label, args.model, args.preprocessing):
    if not os.path.exists(model) or not os.path.exists(preprocessing):
        log.warning("Skipping " + label)
        continue

    log.info("Loading preprocessing from: " + repr(preprocessing))

    f = np.load(preprocessing)

    offset = f["offset"]
    scale = f["scale"]
    top_cutoff = f["top_cutoff"]
    bot_cutoff = f["bot_cutoff"]
    norm_factor = f["norm_factor"]

    log.info("Loading model from: " + repr(model))
    net = ReweightingNet(len(args.variables), hidden_layers=args.layers, leak=0.1)
    net.load_state_dict(torch.load(model))
    net.eval()

    with torch.no_grad():
        log.info("Evaluating network")
        X = torch.Tensor(df[args.variables].values).float()
        X = torch.min(X, torch.Tensor(top_cutoff))
        X = torch.max(X, torch.Tensor(bot_cutoff))
        X = (X - offset) / scale

        pred = net(X).detach().numpy()
        df[label] = norm_factor * np.exp(pred)

log.info("Saving to: " + repr(args.outfile))
with uproot.recreate(args.outfile) as f:
    branch_dict = {l: "float32" for l in args.label if l in df}
    f[args.treename] = uproot.newtree(branch_dict)

    contents = {l: np.array(df[l].values) for l in args.label if l in df}
    f[args.treename].extend(contents)
