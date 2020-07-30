#!/usr/bin/env python
import argparse
import json

import torch
from ttbar_reweighting import ReweightingNet


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
parser.add_argument("--num-inputs", default=3, type=int)
args = parser.parse_args()


import ROOT as R

model = ReweightingNet(args.num_inputs)
model.load_state_dict(torch.load(args.infile))

print("Converting model:")
print(model)

f = R.TFile.Open(args.outfile, "RECREATE")
for layername in ["fc1", "fc2", "fc3", "fc4", "fc5", "fc6"]:
    layer = getattr(model, layername)

    weight = layer.weight.detach().numpy()
    bias = layer.bias.detach().numpy()

    weight = R.TMatrixF(*weight.shape, weight)
    bias = R.TVectorF(*bias.shape, bias)

    weight.Write(layername + "_weight")
    bias.Write(layername + "_bias")

f.Close()
