import numpy as np
import matplotlib.pyplot as plt
from ldpc.config import LDPC_EBN0_DB, ITERATIONS, SHOW_PLOTS, SAVE_PLOTS, PLOT_PREFIX

def plot_ldpc_results(ldpc_results, llr_snapshot):
    floor = 1e-8

    plt.figure(figsize=(7, 4.8))
    for it in ITERATIONS:
        plt.semilogy(LDPC_EBN0_DB, np.clip(ldpc_results[it], floor, None), marker="x", label=f"Iteration {it}")
    plt.xlabel("EbNo(dB)")
    plt.ylabel("BER")
    plt.title("BPSK modulation, LDPC Code, normalized min-sum decoding")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_ber.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()

    plt.figure(figsize=(8, 5.2))
    for it in ITERATIONS:
        if llr_snapshot[it] is not None:
            plt.scatter(range(len(llr_snapshot[it])), llr_snapshot[it], s=18, label=f"Iteration {it}")
    plt.xlabel("Bits")
    plt.ylabel("Soft Bits")
    plt.title("LDPC Decoder Iterations")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_llr.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()
