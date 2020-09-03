#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import uproot


parser = argparse.ArgumentParser()
parser.add_argument("ntuple")
parser.add_argument("outfile")
args = parser.parse_args()


with uproot.open(args.ntuple) as f:
    df = f["Nominal"].pandas.df(["run_number", "event_number", "tau_loose"])
    mask = df.duplicated(["run_number", "event_number"], keep=False)
    is_dupe = (mask & ~df.tau_loose)

with uproot.recreate(args.outfile) as f:
    branch_dict = {"is_dupe": "bool"}
    f["dupe_tree"] = uproot.newtree(branch_dict)
    f["dupe_tree"].extend({"is_dupe": is_dupe})
