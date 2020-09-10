#!/usr/bin/env python
import argparse
import logging
import os
from array import array
from glob import glob
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument("ntuple_dir")
parser.add_argument("dupe_dir")
parser.add_argument("deco_tree")

parser.add_argument("--sel", default=None)
parser.add_argument("--same-sign", action="store_true")
parser.add_argument("-o", "--outfile", default="plots.root")

parser.add_argument("--baseline-selection", default=None)
parser.add_argument("--plot-weight", default="weight / tauSF")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("plot_df.py")


import ROOT as R
R.gROOT.SetBatch(True)


# Load trees
files = {}
trees = {}
for fn in glob(os.path.join(args.ntuple_dir, "*.root")):
    sample = os.path.basename(fn).replace(".root", "")
    log.info("Found {} in ntuple directory".format(sample))

    files[sample] = R.TFile.Open(fn, "READ")
    t = files[sample].Get("Nominal")
    t.AddFriend("dupe_tree", os.path.join(args.dupe_dir, sample + ".root"))
    trees[sample] = t


# Weights for ttbar
trees["ttbarIncl"].AddFriend("rw_tree", args.deco_tree)

ttbar_weights = []
for br in trees["ttbarIncl"].GetFriend("rw_tree").GetListOfBranches():
    ttbar_weights.append(br.GetName())

# Build dataframes for quicker plotting
dfs = {}
for key, tree in trees.items():
    dfs[key] = R.RDataFrame(tree)

# Dataframes after selection
filtered_dfs = {}
for key, df in dfs.items():
    filtered = df.Filter("!is_dupe && n_btag == 2")

    if not args.baseline_selection:
        filtered = df.Filter("mBB > 150000.0 && mTW > 40000.0")
    else:
        filtered = df.Filter(args.baseline_selection)

    if args.same_sign:
        filtered = filtered.Filter("!OS")
    else:
        filtered = filtered.Filter("OS")

    if args.sel:
        filtered = filtered.Filter(args.sel)

    filtered_dfs[key] = filtered

filtered_dfs["ttbar"] = filtered_dfs["ttbarIncl"].Filter("!is_fake")
filtered_dfs["ttbarFake"] = filtered_dfs["ttbarIncl"].Filter("is_fake")

# Add definitions
for key, df in filtered_dfs.items():
    # Define weights
    df = df.Define("weight_nosf", args.plot_weight)
    if "ttbar" in key:
        df = df.Define("weight_nosf_rw", "weight_nosf * rw_nominal")
        for i in range(1, 51):
            df = df.Define("weight_nosf_rw{}".format(i), "weight_nosf * rw_bs{}".format(i))

    # Define variables to plot
    df = df.Define("tauptfr", "tau_pt / 1000.0")
    df = df.Define("taupt", "tau_pt / 1000.0")
    df = df.Define("njets", "n_jets")
    df = df.Define("ht", "HT / 1000.0")
    df = df.Define("mbb", "mBB / 1000.0")
    df = df.Define("mbblo", "mBB / 1000.0")
    df = df.Define("mmc", "mMMC / 1000.0")
    df = df.Define("drtaulep", "dRTauLep")
    df = df.Define("drbb", "dRBB")
    df = df.Define("mtw", "mTW / 1000.0")
    df = df.Define("mtwhi", "mTW / 1000.0")
    df = df.Define("met", "MET / 1000.0")
    df = df.Define("leppt", "lep_pt / 1000.0")
    df = df.Define("mhh", "mHH / 1000.0")
    df = df.Define("b0pt", "b0_pt / 1000.0")
    df = df.Define("b1pt", "b1_pt / 1000.0")
    df = df.Define("rnnscore", "tau_rnn")

    filtered_dfs[key] = df


pt_bins = array("f", [20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 90, 120, 160, 250, 500, 1000])
njets_bins = array("f", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

plots = [
    ("tauptfr", R.RDF.TH1DModel("h_tauptfr_proto", "", len(pt_bins) - 1, pt_bins)),
    ("taupt", R.RDF.TH1DModel("h_taupt_proto", "", 36, 20, 200)),
    ("njets", R.RDF.TH1DModel("h_njets_proto", "", len(njets_bins) - 1, njets_bins)),
    ("ht", R.RDF.TH1DModel("h_ht_proto", "", 40, 0, 2000)),
    ("mbb", R.RDF.TH1DModel("h_mbb_proto", "", 25, 150, 650)),
    ("mbblo", R.RDF.TH1DModel("h_mbblo_proto", "", 30, 0, 150)),
    ("mmc", R.RDF.TH1DModel("h_mmc_proto", "", 30, 60, 660)),
    ("drtaulep", R.RDF.TH1DModel("h_drtaulep_proto", "", 36, 0, 6)),
    ("drbb", R.RDF.TH1DModel("h_drbb_proto", "", 36, 0, 6)),
    ("mtw", R.RDF.TH1DModel("h_mtw_proto", "", 25, 0, 250)),
    ("mtwhi", R.RDF.TH1DModel("h_mtwhi_proto", "", 25, 150, 400)),
    ("met", R.RDF.TH1DModel("h_met_proto", "", 20, 0, 400)),
    ("leppt", R.RDF.TH1DModel("h_leppt_proto", "", 36, 20, 200)),
    ("mhh", R.RDF.TH1DModel("h_mhh_proto", "", 36, 200, 2000)),
    ("b0pt", R.RDF.TH1DModel("h_b0pt_proto", "", 41, 45, 250)),
    ("b1pt", R.RDF.TH1DModel("h_b1pt_proto", "", 36, 20, 200)),
    ("rnnscore", R.RDF.TH1DModel("h_rnnscore_proto", "", 50, 0., 1.)),
]

all_hists = []
for (name, hist), (sample, df) in product(plots, filtered_dfs.items()):
    histname = "h_{sample}_{variable}".format(sample=sample, variable=name)
    hist.fName = histname

    all_hists.append(
        df.Histo1D(hist, name, "weight_nosf")
    )

    if "ttbar" in sample:
        histname = "h_{sample}RW_{variable}".format(sample=sample, variable=name)
        hist.fName = histname

        all_hists.append(
            df.Histo1D(hist, name, "weight_nosf_rw")
        )

        for i in range(1, 51):
            histname = "h_{sample}RW{num}_{variable}".format(sample=sample, variable=name, num=i)
            hist.fName = histname

            all_hists.append(
                df.Histo1D(hist, name, "weight_nosf_rw{}".format(i))
            )

# Do the plotting
f_out = R.TFile.Open(args.outfile, "RECREATE")

for hist in all_hists:
    h = hist.GetValue()
    h.Write()

f_out.Close()
for _, f in files.items():
    f.Close()
