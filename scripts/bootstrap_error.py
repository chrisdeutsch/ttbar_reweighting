#!/usr/bin/env python
import argparse
import re

from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()


import ROOT as R

f_in = R.TFile.Open(args.infile)
f_out = R.TFile.Open(args.outfile, "RECREATE")

bootstraps = {}
pattern = re.compile(r"h_(\w+)RW(\d+)_(\w+)")
for key in f_in.GetListOfKeys():
    name = key.GetName()

    m = pattern.match(name)
    if m:
        sample, bootstrap, variable = m.groups()
        bootstraps.setdefault((sample, variable), []).append(key)


for sample, variable in bootstraps:
    histname_nom = "h_{sample}RW_{variable}".format(sample=sample, variable=variable)
    h_nom = f_in.Get(histname_nom)

    h_std = h_nom.Clone(histname_nom + "_std")
    h_std.Reset()

    n = 0
    for key in bootstraps[sample, variable]:
        n += 1

        h = key.ReadObj()
        h.Add(h_nom, -1.0)

        for i in range(h.GetNbinsX() + 2):
            h_std.SetBinContent(i, h_std.GetBinContent(i) + h.GetBinContent(i)**2)

    for i in range(h_std.GetNbinsX() + 2):
        h_std.SetBinContent(i, sqrt(h_std.GetBinContent(i) / n))

    f_out.cd()
    h_std.Write()
