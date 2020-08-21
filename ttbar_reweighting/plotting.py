import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ttbar_reweighting.plotting")


def plot_loss(train_loss, test_loss, outfile="loss.pdf"):
    fig, ax = plt.subplots()

    x_train = 1 + np.arange(len(train_loss))
    x_test = 1 + np.arange(len(test_loss))
    y_train = np.array(train_loss)
    y_test = np.array(test_loss)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.plot(x_train, y_train, c="r", label="Training")
    ax.plot(x_test, y_test, c="b", label="Testing")
    ax.legend()

    fig.savefig(outfile)
    plt.close(fig)


# See also Plotter.plot
default_plots = [
    ("taupt", lambda df: df.tau_pt / 1000.0, dict(bins=36, range=(20, 200)), "Tau candidate $p_{T}$ [GeV]"),
    ("leppt", lambda df: df.lep_pt / 1000.0, dict(bins=36, range=(20, 200)), "Lepton $p_{T}$ [GeV]"),
    ("mtw", lambda df: df.mTW / 1000.0, dict(bins=25, range=(0, 250)), "$m_{T,W}$ [GeV]"),
    ("met", lambda df: df.MET / 1000.0, dict(bins=20, range=(0, 400)), "MET [GeV]"),
    ("njets", lambda df: np.clip(df.n_jets, 0, 12), dict(bins=np.arange(2, 13)), "$N_{jets}$"),
    ("ht", lambda df: df.HT / 1000.0, dict(bins=40, range=(0, 2000)), "$H_{T}$ [GeV]"),
    ("b0pt", lambda df: df.b0_pt / 1000.0, dict(bins=41, range=(45, 250)), "B0 $p_{T}$ [GeV]"),
    ("b1pt", lambda df: df.b1_pt / 1000.0, dict(bins=36, range=(20, 200)), "B1 $p_{T}$ [GeV]"),
    ("jet0pt", lambda df: df.lead_jet_pt / 1000.0, dict(bins=71, range=(45, 400)), "Lead. jet $p_{T}$ [GeV]"),
    ("mbb", lambda df: df.mBB / 1000.0, dict(bins=25, range=(150, 650)), "$m_{bb}$ [GeV]"),
    ("mbblow", lambda df: df.mBB / 1000.0, dict(bins=30, range=(0, 150)), "$m_{bb}$ [GeV]"),
    ("mhh", lambda df: df.mHH / 1000.0, dict(bins=36, range=(200, 2000)), "$m_{HH}$ [GeV]"),
    ("drtaulep", lambda df: df.dRTauLep, dict(bins=36, range=(0, 6)), "$\Delta R(tau, lep)$"),
    ("drbb", lambda df: df.dRBB, dict(bins=36, range=(0, 6)), "$\Delta R(b, b)$"),
    ("mindrbl", lambda df: df.minDRbl, dict(bins=36, range=(0, 6)), "$min \Delta R(b, l)$"),
    ("mindrbtau", lambda df: df.minDRbtau, dict(bins=36, range=(0, 6)), "$min \Delta R(b, tau)$"),
    ("ptbb", lambda df: df.pTBB / 1000.0, dict(bins=40, range=(0, 400)), "$p_{T, bb}$ [GeV]"),
    ("pthh", lambda df: df.pTHH / 1000.0, dict(bins=50, range=(0, 500)), "$p_{T, hh}$ [GeV]"),
    ("pttautau", lambda df: df.pTTauTau / 1000.0, dict(bins=40, range=(0, 400)), "$p_{T, tautau}$ [GeV]"),
    ("pttaulep", lambda df: df.pTTauLep / 1000.0, dict(bins=40, range=(0, 400)), "$p_{T, taulep}$ [GeV]"),
]


class Plotter(object):
    def __init__(self, data, ttbar, non_ttbar, sel=None):
        self.data = data.copy()
        self.ttbar = ttbar.copy()
        self.non_ttbar = non_ttbar.copy()

        if sel:
            self.data = self.data.loc[sel(self.data)].copy()
            self.ttbar = self.ttbar.loc[sel(self.ttbar)].copy()
            self.non_ttbar = self.non_ttbar.loc[sel(self.non_ttbar)].copy()


    def plot(self, f_var, binning, label, sf):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})

        ax[0][0].set_title("Before correction")
        _, bins, _ = ax[0][0].hist(
            [f_var(self.non_ttbar), f_var(self.ttbar)],
            weights=[self.non_ttbar.weight, self.ttbar.weight],
            stacked=True, label=["non-ttbar", "ttbar"], **binning
        )

        ax[0][1].set_title("After correction")
        _, bins, _ = ax[0][1].hist(
            [f_var(self.non_ttbar), f_var(self.ttbar)],
            weights=[self.non_ttbar.weight, self.ttbar[sf] * self.ttbar.weight],
            stacked=True, label=["non-ttbar", "ttbar"], **binning
        )

        data, _ = np.histogram(f_var(self.data), bins=bins)

        # Total prediction to compare to data
        pred_prefit = np.histogram(f_var(self.ttbar),
                                   weights=self.ttbar.weight,
                                   **binning)[0]
        pred_prefit += np.histogram(f_var(self.non_ttbar),
                                    weights=self.non_ttbar.weight,
                                    **binning)[0]

        pred_postfit = np.histogram(f_var(self.ttbar),
                                    weights=self.ttbar[sf] * self.ttbar.weight,
                                    **binning)[0]
        pred_postfit += np.histogram(f_var(self.non_ttbar),
                                     weights=self.non_ttbar.weight,
                                     **binning)[0]

        # Variance of total prediction
        var_prefit = np.histogram(f_var(self.ttbar),
                                  weights=self.ttbar.weight**2,
                                  **binning)[0]
        var_prefit += np.histogram(f_var(self.non_ttbar),
                                   weights=self.non_ttbar.weight**2,
                                   **binning)[0]

        var_postfit = np.histogram(f_var(self.ttbar),
                                   weights=(self.ttbar[sf] * self.ttbar.weight)**2,
                                   **binning)[0]
        var_postfit += np.histogram(f_var(self.non_ttbar),
                                    weights=self.non_ttbar.weight**2,
                                    **binning)[0]

        rel_unc_prefit = np.sqrt(var_prefit) / pred_prefit
        rel_unc_postfit = np.sqrt(var_postfit) / pred_postfit

        # Add the last one twice for 'fill_between' (step='post')
        rel_unc_prefit = np.concatenate([rel_unc_prefit, [rel_unc_prefit[-1]]])
        rel_unc_postfit = np.concatenate([rel_unc_postfit, [rel_unc_postfit[-1]]])

        # Ratio plots
        ax[1][0].axhline(1, ls="--", c="k")
        ax[1][1].axhline(1, ls="--", c="k")

        ax[1][0].fill_between(bins, 1 + rel_unc_prefit, 1 - rel_unc_prefit, step="post")
        ax[1][1].fill_between(bins, 1 + rel_unc_postfit, 1 - rel_unc_prefit, step="post")

        ax[1][0].errorbar(0.5 * (bins[:-1] + bins[1:]), data / pred_prefit,
                          xerr=0.5 * np.diff(bins), yerr=np.sqrt(data) / pred_prefit, fmt="ko")
        ax[1][1].errorbar(0.5 * (bins[:-1] + bins[1:]), data / pred_postfit,
                          xerr=0.5 * np.diff(bins), yerr=np.sqrt(data) / pred_postfit, fmt="ko")

        for a in ax[0]:
            a.errorbar(0.5 * (bins[:-1] + bins[1:]), data, xerr=0.5 * np.diff(bins), yerr=np.sqrt(data), fmt="ko")
            a.set_xlim(bins[0], bins[-1])
            a.set_xlabel(label)
            a.set_ylabel("Events")
            a.legend()

        for a in ax[1]:
            a.set_xlim(bins[0], bins[-1])
            a.set_xlabel(label)
            a.set_ylabel("Data / Pred.")
            a.set_ylim(0.7, 1.3)

        return fig
