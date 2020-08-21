import logging

import uproot


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ttbar_reweighting.utils")

def get_dataframe(filename):
    f = uproot.open(filename)

    variables = [
        "weight", "tauSF",
        "n_btag", "n_jets", "OS", "is_fake", "event_flavour",
        "run_number", "event_number",
        "tau_pt", "tau_prong", "tau_loose",
        "lep_pt", "b0_pt", "b1_pt", "lead_jet_pt",
        "mBB", "mTW", "MET", "HT", "mHH", "dRTauLep", "dRBB",
        "pTBB", "pTHH", "pTTauTau", "pTTauLep", "minDRbl", "minDRbtau"
    ]

    df = f["Nominal"].pandas.df(variables)

    return df


def remove_duplicates(df):
    dupe_mask = df.duplicated(["run_number", "event_number"], keep=False)
    return df.loc[~(dupe_mask & ~df.tau_loose)], (dupe_mask & ~df.tau_loose).sum()


def apply_selection(df, region="FR"):
    sel = df.OS & (df.n_btag == 2)

    if region == "FR":
        sel = sel & (df.mBB > 150000.0)
        sel = sel & (df.mTW > 40000.0)
    elif region == "FR_Bowen":
        sel = sel & (df.mBB > 50000.0) & ((df.mBB > 150000.0) | (df.mBB < 100000.0))
        sel = sel & (df.mTW > 60000.0)
    elif region == "VR1":
        sel = sel & (df.mBB <= 150000.0)
        sel = sel & (df.mTW > 60000.0)
    elif region == "VR2":
        sel = sel & (df.mBB > 150000.0)
        sel = sel & (df.mTW > 150000.0)
    else:
        raise RuntimeError("Unknown selection")

    return df.loc[sel].copy()


def is_hf(df):
    flavour = df.event_flavour
    # Flavour: bb = 1, bc = 2, bl = 3, cc = 4, cl = 5, l = 6
    return (flavour == 1) | (flavour == 2) | (flavour == 4)
