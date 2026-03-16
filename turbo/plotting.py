import numpy as np
import matplotlib.pyplot as plt
from turbo.config import CONV_EBN0_DB, TURBO_EBN0_DB, ITERATIONS, SHOW_PLOTS, SAVE_PLOTS, PLOT_PREFIX

def plot_turbo_results(conv_uncoded, conv_coded, turbo_results, llr_snapshot):
    floor = 1e-8

    plt.figure(figsize=(7, 4.8))
    plt.semilogy(CONV_EBN0_DB, np.clip(conv_uncoded, floor, None), "rx-", label="Uncoded")
    plt.semilogy(CONV_EBN0_DB, np.clip(conv_coded, floor, None), "bo-", label="Coded")
    plt.xlabel("EbNo(dB)")
    plt.ylabel("BER")
    plt.title("BPSK modulation, Convolutional Code, Soft Viterbi Decoding")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_conv.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()

    plt.figure(figsize=(7, 4.8))
    for it in ITERATIONS:
        plt.semilogy(TURBO_EBN0_DB, np.clip(turbo_results[it], floor, None), marker="x", label=f"Iteration {it}")
    plt.xlabel("EbNo(dB)")
    plt.ylabel("BER")
    plt.title("BPSK modulation, Turbo Code, max log MAP decoding")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_turbo.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()

    plt.figure(figsize=(8, 5.2))
    for it in ITERATIONS:
        if llr_snapshot[it] is not None:
            plt.scatter(range(len(llr_snapshot[it])), llr_snapshot[it], s=18, label=f"Iteration {it}")
    plt.xlabel("Bits")
    plt.ylabel("Soft Bits")
    plt.title("Turbo Decoder Iterations")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_llr.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()
