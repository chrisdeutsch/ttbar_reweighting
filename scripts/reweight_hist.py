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
log = logging.getLogger("reweight_hist.py")

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


# Correction via histogram ratios
def get_correction(f_var, f_weight, bins):
    # Event counts
    data, _ = np.histogram(f_var(df_data), weights=df_data.weight, bins=bins)
    ttbar, _ = np.histogram(f_var(df_ttbar), weights=f_weight(df_ttbar), bins=bins)
    non_ttbar, _ = np.histogram(f_var(df_non_ttbar), weights=df_non_ttbar.weight, bins=bins)

    # Variance
    data_var, _ = np.histogram(f_var(df_data), weights=df_data.weight**2, bins=bins)
    ttbar_var, _ = np.histogram(f_var(df_ttbar), weights=f_weight(df_ttbar)**2, bins=bins)
    non_ttbar_var, _ = np.histogram(f_var(df_non_ttbar), weights=df_non_ttbar.weight**2, bins=bins)

    corr = (data - non_ttbar) / ttbar
    corr_var = data_var / ttbar**2 + non_ttbar_var / ttbar**2 + ((data - non_ttbar) / ttbar**2)**2 * ttbar_var

    return corr, np.sqrt(corr_var)


def add_correction_to_df(df, f_var, corr, bins, name):
    bin_idx = np.digitize(f_var(df), bins)

    if bin_idx.min() == 0:
        log.error("There are events in the underflow bin")
        sys.exit(1)

    if bin_idx.max() == len(bins):
        log.error("There are events in the overflow bin")
        sys.exit(1)

    df[name] = corr[bin_idx - 1]


# nJets correction
bins = np.array([2, 3, 4, 5, 6, 7, 8, 9])
f_var = lambda df: np.clip(df.n_jets, 2, 8)
f_weight = lambda df: df.weight
corr_njets, dcorr_njets = get_correction(f_var, f_weight, bins)

print("NJets correction factors: " + repr(corr_njets))
print("NJets correction uncertainty: " + repr(dcorr_njets))

add_correction_to_df(df_ttbar, f_var, corr_njets, bins, "sf_njets")

# Tau Pt correction
bins = np.array([20, 25, 30, 35, 40, 45, 55, 70, 100, 200])
f_var = lambda df: np.clip(df.tau_pt / 1000.0, 20, 150 - 1e-6)
f_weight = lambda df: df.weight
corr_taupt, dcorr_taupt = get_correction(f_var, f_weight, bins)

print("Taupt correction factors: " + repr(corr_taupt))
print("Taupt correction uncertainty: " + repr(dcorr_taupt))

add_correction_to_df(df_ttbar, f_var, corr_taupt, bins, "sf_taupt")

# Sequential tau pt correction (after njets)
f_weight = lambda df: df.weight * df.sf_njets
corr_seq_taupt, dcorr_seq_taupt = get_correction(f_var, f_weight, bins)

print("Sequential taupt correction factors: " + repr(corr_seq_taupt))
print("Sequential taupt correction uncertainty: " + repr(dcorr_seq_taupt))

add_correction_to_df(df_ttbar, f_var, corr_seq_taupt, bins, "sf_seq_taupt")
df_ttbar["sf_njets_taupt"] = df_ttbar.sf_njets * df_ttbar.sf_seq_taupt

# Lepton pt correction
#bins = np.array([25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 130, 250])
bins = np.array([25, 30, 32.5, 35, 37.5, 40, 45, 50, 60, 70, 80, 100, 130, 250])
f_var = lambda df: np.clip(df.lep_pt / 1000.0, 25, 250 - 1e-3)
f_weight = lambda df: df.weight
corr_leppt, dcorr_leppt = get_correction(f_var, f_weight, bins)
add_correction_to_df(df_ttbar, f_var, corr_leppt, bins, "sf_leppt")

print("Leppt correction factors: " + repr(corr_leppt))
print("Leppt correction uncertainty (rel.): " + repr(dcorr_leppt / corr_leppt))

# Sequential lepton pt correction
f_weight = lambda df: df.sf_njets * df.weight
corr_leppt, dcorr_leppt = get_correction(f_var, f_weight, bins)
add_correction_to_df(df_ttbar, f_var, corr_leppt, bins, "sf_seq_leppt")
df_ttbar["sf_njets_leppt"] = df_ttbar.sf_njets * df_ttbar.sf_seq_leppt

# HT correction
bins = np.array([  96.72029114,  197.22161682,  219.42976868,  235.9423761 ,
                   249.99807739,  262.73795776,  274.88577637,  286.69605469,
                   298.36404907,  310.11971802,  322.06361084,  334.41631104,
                   347.32684326,  360.86949951,  375.28020874,  390.82820435,
                   407.81650757,  426.67416016,  447.84242798,  472.19020752,
                   500.97312622,  536.76955322,  583.44659912,  650.9482666 ,
                   772.47488037, 5426.69433594])

f_var = lambda df: np.clip(df.HT / 1000.0, 100, 5000)
f_weight = lambda df: df.weight
corr_ht, dcorr_ht = get_correction(f_var, f_weight, bins)
add_correction_to_df(df_ttbar, f_var, corr_ht, bins, "sf_ht")

print("HT correction factors: " + repr(corr_ht))
print("HT correction uncertainty: " + repr(dcorr_ht))

# Sequential HT correction
f_weight = lambda df: df.sf_njets * df.weight
corr_ht, dcorr_ht = get_correction(f_var, f_weight, bins)
add_correction_to_df(df_ttbar, f_var, corr_ht, bins, "sf_seq_ht")
df_ttbar["sf_njets_ht"] = df_ttbar.sf_njets * df_ttbar.sf_seq_ht


import matplotlib.pyplot as plt
from ttbar_reweighting import Plotter, default_plots

scale_factors = ["sf_taupt", "sf_njets", "sf_njets_taupt", "sf_leppt", "sf_njets_leppt", "sf_ht", "sf_njets_ht"]

plotter = Plotter(df_data, df_ttbar, df_non_ttbar)
for sf in scale_factors:
    for args in default_plots:
        fig = plotter.plot(*args[1:], sf)
        fig.savefig("{}_{}.pdf".format(args[0], sf))
        plt.close(fig)

del plotter

for name, sel in [("1p", lambda df: df.tau_prong == 1), ("3p", lambda df: df.tau_prong == 3)]:
    plotter = Plotter(df_data, df_ttbar, df_non_ttbar, sel)
    for sf in scale_factors:
        for args in default_plots:
            fig = plotter.plot(*args[1:], sf)
            fig.savefig("{}_{}_{}.pdf".format(args[0], sf, name))
            plt.close(fig)

    del plotter

# TODO: Plot ID region
