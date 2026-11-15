"""Make the report figures from results/results.csv.

    python -m eval.plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {
    "Phantom (RepairNet)": dict(color="#d62728", lw=2.6, marker="o"),
    "Phantom + NACK": dict(color="#9467bd", lw=2.2, marker="D"),
    "Phantom-Interp (framegen, +33 ms)": dict(color="#e377c2", lw=2.6, marker="h"),
    "Flow extrapolation": dict(color="#ff7f0e", lw=1.6, marker="^"),
    "Copy-tile (no retx)": dict(color="#7f7f7f", lw=1.6, marker="s"),
    "Freeze (no retx)": dict(color="#000000", lw=1.4, marker="x", ls="--"),
    "NACK + copy": dict(color="#1f77b4", lw=1.8, marker="v"),
    "FEC k=5 + copy": dict(color="#2ca02c", lw=1.8, marker="P"),
    "TCP": dict(color="#8c564b", lw=1.8, marker="*"),
}


def line(ax, df, metric, xlabel, ylabel, title, skip=()):
    for name, g in df.groupby("scheme"):
        if name in skip:
            continue
        g = g.groupby("x")[metric].mean().sort_index()
        ax.plot(g.index, g.values, label=name, **STYLE.get(name, {}))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/figs")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    loss = df[df.experiment == "loss"]
    if len(loss):
        fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
        line(axs[0], loss, "psnr_mean", "packet loss (%)", "PSNR (dB)", "Quality vs packet loss (RTT 50 ms)", skip=("TCP",))
        line(axs[1], loss, "psnr_p5", "packet loss (%)", "worst-5% PSNR (dB)", "Tail quality vs loss", skip=("TCP",))
        line(axs[2], loss, "latency_p95", "packet loss (%)", "P95 frame latency (ms)", "Latency vs loss")
        axs[2].set_yscale("log")
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "e1_loss.png", dpi=150)

        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        line(ax, loss, "overhead_pct", "packet loss (%)", "bytes on the wire beyond media (%)", "Bandwidth overhead")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "e1_overhead.png", dpi=150)

    burst = df[df.experiment == "burst"]
    if len(burst):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        line(ax, burst, "psnr_mean", "mean burst length (packets)", "PSNR (dB)", "Bursty loss (5% mean, Gilbert-Elliott)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "e2_burst.png", dpi=150)

    rtt = df[df.experiment == "rtt"]
    if len(rtt):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
        line(axs[0], rtt, "psnr_mean", "RTT (ms)", "PSNR (dB)", "Quality vs RTT (5% loss, 100 ms jitter buffer)", skip=("TCP",))
        line(axs[1], rtt, "latency_p95", "RTT (ms)", "P95 frame latency (ms)", "Latency vs RTT")
        axs[0].axvline(100, color="k", ls=":", lw=1)
        axs[0].text(101, axs[0].get_ylim()[0] + 0.3, "RTT = jitter buffer", fontsize=8)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "e3_rtt.png", dpi=150)

    ilv = df[df.experiment == "ilv"]
    if len(ilv):
        t = ilv.groupby("scheme")["psnr_mean"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.barh(t.index, t.values, color=["#d62728" if "RepairNet" in s else "#7f7f7f" for s in t.index])
        ax.set_xlim(t.min() - 2, t.max() + 0.5)
        ax.set_xlabel("PSNR (dB)")
        ax.set_title("Interleaving ablation (5% bursty loss, burst 4)")
        fig.tight_layout()
        fig.savefig(out / "e4_interleave.png", dpi=150)

    # per-content breakdown at 10 % loss: gain over copy-tile concealment
    if "content" in df.columns and len(loss) and loss["content"].nunique() > 1:
        L = loss[loss.x == 10]
        piv = L.groupby(["content", "scheme"])["psnr_mean"].mean().unstack()
        base = piv["Copy-tile (no retx)"]
        show = [s for s in ["Flow extrapolation", "FEC k=5 + copy", "NACK + copy", "Phantom (RepairNet)",
                            "Phantom + NACK", "Phantom-Interp (framegen, +33 ms)"] if s in piv]
        gain = piv[show].sub(base, axis=0)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        w = 0.8 / len(show)
        for j, s in enumerate(show):
            ax.bar([i + j * w for i in range(len(gain))], gain[s], w, label=s, color=STYLE[s]["color"])
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(gain))])
        ax.set_xticklabels(gain.index)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("PSNR gain over copy-tile (dB)")
        ax.set_title("Gain by content type (10% loss, RTT 50 ms)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "e5_content.png", dpi=150)
        gain.round(2).to_csv(out / "table_content_gain.csv")
        print(gain.round(2).to_string())

    # headline table at 5 % loss
    t = df[(df.experiment == "loss") & (df.x == 5)].groupby("scheme")[
        ["psnr_mean", "psnr_p5", "ssim_mean", "frozen_pct", "stall_ms", "latency_p95", "overhead_pct", "infer_ms_p95"]
    ].mean().round(2).sort_values("psnr_mean", ascending=False)
    if len(t):
        t.to_csv(out / "table_5pct_loss.csv")
        print(t.to_string())
    print("figures in", out)


if __name__ == "__main__":
    main()
