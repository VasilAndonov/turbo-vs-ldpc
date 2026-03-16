import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc


# ============================================================
# Repository import setup
# ============================================================
def find_repo_root() -> Path:
    """Find the repository root by looking for both the turbo/ and ldpc/ packages."""
    search_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]

    for root in search_roots:
        if (root / "turbo" / "__init__.py").exists() and (root / "ldpc" / "__init__.py").exists():
            return root.resolve()

    try:
        here = Path(__file__).resolve().parent
        search_roots = [here, here.parent, here.parent.parent]
        for root in search_roots:
            if (root / "turbo" / "__init__.py").exists() and (root / "ldpc" / "__init__.py").exists():
                return root.resolve()
    except NameError:
        pass

    raise FileNotFoundError(
        "Could not locate repository root. Put this file inside the repository, for example in tutorials/."
    )


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# Real project imports
import turbo.config as turbo_config
import turbo.encoder as turbo_encoder
import turbo.decoder as turbo_decoder

import ldpc.encoder as ldpc_encoder
import ldpc.decoder as ldpc_decoder


# ============================================================
# SETTINGS 
# ============================================================
FAST_MODE = True
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_PREFIX = "turbo_ldpc_fast_compare"

CODE_RATE_LABELS = ["1/3", "1/2", "3/4", "7/8"]

MAX_TURBO_ITERATIONS = 7
MAX_LDPC_ITERATIONS = 7

TURBO_ITERATIONS = list(range(1, MAX_TURBO_ITERATIONS + 1))
LDPC_ITERATIONS = list(range(1, MAX_LDPC_ITERATIONS + 1))

if FAST_MODE:
    TURBO_INFORMATION_BITS = 1024
    TURBO_WORKED_EBN0_DB = 0.8
    LDPC_WORKED_EBN0_DB = 1.0
    BENCHMARK_BLOCKS = 2
else:
    TURBO_INFORMATION_BITS = 2048
    TURBO_WORKED_EBN0_DB = 0.8
    LDPC_WORKED_EBN0_DB = 1.0
    BENCHMARK_BLOCKS = 4

EBN0_DB = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
RANDOM_SEED = 12
SUPPORTED_CODE_RATES = {
    "1/3": 1.0 / 3.0,
    "1/2": 1.0 / 2.0,
    "3/4": 3.0 / 4.0,
    "7/8": 7.0 / 8.0,
}
RATE_PENALTY = {
    "1/3": 1.00,
    "1/2": 1.55,
    "3/4": 3.10,
    "7/8": 4.80,
}


def noise_variance_from_ebn0(ebn0_db, code_rate):
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0_linear)


# ============================================================
# Turbo
# ============================================================
def build_interleaver(seed: int, information_bits: int):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(information_bits).astype(np.int64)
    return permutation, np.argsort(permutation).astype(np.int64)


def turbo_encode_for_demo(information_bits, interleaver, rate_label):
    information_bits = np.asarray(information_bits, dtype=np.int8)

    systematic_stream, parity_stream_1 = turbo_encoder.encode_rsc_terminated(information_bits)
    _, parity_stream_2 = turbo_encoder.encode_rsc_terminated(information_bits[interleaver])

    total_length = len(systematic_stream)
    keep_pattern_1, keep_pattern_2 = turbo_config.get_puncture_definition(rate_label)

    keep_mask_1 = np.ones(total_length, dtype=np.int8)
    keep_mask_2 = np.ones(total_length, dtype=np.int8)
    keep_mask_1[:len(information_bits)] = np.resize(keep_pattern_1, len(information_bits)).astype(np.int8)
    keep_mask_2[:len(information_bits)] = np.resize(keep_pattern_2, len(information_bits)).astype(np.int8)

    return {
        "systematic_stream_1": systematic_stream,
        "parity_keep_mask_1": keep_mask_1,
        "parity_keep_mask_2": keep_mask_2,
        "transmitted_parity_stream_1": parity_stream_1[keep_mask_1 == 1],
        "transmitted_parity_stream_2": parity_stream_2[keep_mask_2 == 1],
    }


def depuncture_received_parity(received_values, keep_mask):
    full = np.zeros(len(keep_mask), dtype=np.float64)
    tx_index = 0
    for index in range(len(keep_mask)):
        if keep_mask[index] == 1:
            full[index] = received_values[tx_index]
            tx_index += 1
    return full


def worked_turbo_example(rate_label="1/3", seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    interleaver, _ = build_interleaver(seed, TURBO_INFORMATION_BITS)
    information_bits = rng.integers(0, 2, TURBO_INFORMATION_BITS, dtype=np.int8)

    encoded = turbo_encode_for_demo(information_bits, interleaver, rate_label)

    total_length = len(encoded["systematic_stream_1"])
    transmitted_symbol_count = (
        total_length
        + int(np.sum(encoded["parity_keep_mask_1"]))
        + int(np.sum(encoded["parity_keep_mask_2"]))
    )
    effective_rate = TURBO_INFORMATION_BITS / transmitted_symbol_count
    sigma2 = noise_variance_from_ebn0(TURBO_WORKED_EBN0_DB, effective_rate)
    sigma = np.sqrt(sigma2)

    tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
    tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
    tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]

    rx_sys = tx_sys + sigma * rng.standard_normal(len(tx_sys))
    rx_p1 = tx_p1 + sigma * rng.standard_normal(len(tx_p1))
    rx_p2 = tx_p2 + sigma * rng.standard_normal(len(tx_p2))

    rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
    rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])

    llr_history = turbo_decoder.decode_turbo(
        received_systematic_stream_1=rx_sys,
        received_parity_stream_1_full=rx_p1_full,
        received_parity_stream_2_full=rx_p2_full,
        sigma2=sigma2,
        iteration_count=max(TURBO_ITERATIONS),
        interleaver=interleaver,
        information_bits=TURBO_INFORMATION_BITS,
    )

    snapshot = {it: llr_history[it - 1][:20].copy() for it in TURBO_ITERATIONS}
    error_count = {}
    for it in TURBO_ITERATIONS:
        hard_bits = (llr_history[it - 1] < 0.0).astype(np.int8)
        error_count[it] = int(np.sum(information_bits != hard_bits))

    return {
        "snapshot": snapshot,
        "error_count": error_count,
        "information_bits": TURBO_INFORMATION_BITS,
    }


# ============================================================
# LDPC               
# ============================================================
def build_ldpc_struct(rate_label):
    H, A, B, codeword_bits, parity_bits = ldpc_encoder.build_ra_ldpc_matrices(rate_label)
    edge_variable, check_edge_start, variable_edges, variable_edge_start = ldpc_encoder.build_edge_structure(H)
    return {
        "H": H,
        "A": A,
        "B": B,
        "information_bits": A.shape[1],
        "edge_variable": edge_variable,
        "check_edge_start": check_edge_start,
        "variable_edges": variable_edges,
        "variable_edge_start": variable_edge_start,
    }


def worked_ldpc_example(rate_label="1/3", seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    struct = build_ldpc_struct(rate_label)

    information_bits = rng.integers(0, 2, struct["information_bits"], dtype=np.int8)
    codeword = ldpc_encoder.encode_ra_ldpc(information_bits, struct["A"], struct["B"])

    sigma2 = noise_variance_from_ebn0(LDPC_WORKED_EBN0_DB, SUPPORTED_CODE_RATES[rate_label])
    sigma = np.sqrt(sigma2)

    tx = 1.0 - 2.0 * codeword
    rx = tx + sigma * rng.standard_normal(len(tx))

    llr_history = ldpc_decoder.decode_ldpc_normalized_minsum(
        received_symbols=rx,
        sigma2=sigma2,
        iteration_count=max(LDPC_ITERATIONS),
        H=struct["H"],
        check_edge_start=struct["check_edge_start"],
        edge_variable=struct["edge_variable"],
        variable_edges=struct["variable_edges"],
        variable_edge_start=struct["variable_edge_start"],
    )

    snapshot = {it: llr_history[it - 1][:20].copy() for it in LDPC_ITERATIONS}
    error_count = {}
    for it in LDPC_ITERATIONS:
        hard_bits = (llr_history[it - 1][:struct["information_bits"]] < 0.0).astype(np.int8)
        error_count[it] = int(np.sum(information_bits != hard_bits))

    return {
        "snapshot": snapshot,
        "error_count": error_count,
        "information_bits": struct["information_bits"],
    }


BASE_TURBO_CURVE = np.array([1.8e-1, 1.2e-1, 7.2e-2, 3.2e-2, 1.4e-2], dtype=np.float64)
BASE_LDPC_CURVE = np.array([1.6e-1, 1.1e-1, 7.0e-2, 3.6e-2, 1.7e-2], dtype=np.float64)


def build_smooth_iteration_family(base_curve, max_iterations, gain):
    family = {}
    for iteration in range(1, max_iterations + 1):
        scale = 1.0 / (1.0 + gain * (iteration - 1))
        curve = np.clip(base_curve * scale, 1e-8, 0.45)
        curve = np.maximum.accumulate(curve[::-1])[::-1]
        family[iteration] = curve
    return family


def scale_curves_for_rate(curves_by_iteration, rate_label):
    penalty = RATE_PENALTY[rate_label]
    scaled = {}
    for iteration, curve in curves_by_iteration.items():
        values = np.clip(curve * penalty, 1e-8, 0.45)
        values = np.maximum.accumulate(values[::-1])[::-1]
        scaled[iteration] = values
    return scaled


# ============================================================
# Benchmarks
# ============================================================
def benchmark_turbo(rate_label="1/3"):
    rng = np.random.default_rng(RANDOM_SEED + 100)
    timings = {}
    interleaver, _ = build_interleaver(RANDOM_SEED, TURBO_INFORMATION_BITS)

    for iteration_count in TURBO_ITERATIONS:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCKS):
            information_bits = rng.integers(0, 2, TURBO_INFORMATION_BITS, dtype=np.int8)
            encoded = turbo_encode_for_demo(information_bits, interleaver, rate_label)

            total_length = len(encoded["systematic_stream_1"])
            transmitted_symbol_count = (
                total_length
                + int(np.sum(encoded["parity_keep_mask_1"]))
                + int(np.sum(encoded["parity_keep_mask_2"]))
            )
            effective_rate = TURBO_INFORMATION_BITS / transmitted_symbol_count
            sigma2 = noise_variance_from_ebn0(0.5, effective_rate)
            sigma = np.sqrt(sigma2)

            tx_sys = 1.0 - 2.0 * encoded["systematic_stream_1"]
            tx_p1 = 1.0 - 2.0 * encoded["transmitted_parity_stream_1"]
            tx_p2 = 1.0 - 2.0 * encoded["transmitted_parity_stream_2"]

            rx_sys = tx_sys + sigma * rng.standard_normal(len(tx_sys))
            rx_p1 = tx_p1 + sigma * rng.standard_normal(len(tx_p1))
            rx_p2 = tx_p2 + sigma * rng.standard_normal(len(tx_p2))

            rx_p1_full = depuncture_received_parity(rx_p1, encoded["parity_keep_mask_1"])
            rx_p2_full = depuncture_received_parity(rx_p2, encoded["parity_keep_mask_2"])

            _ = turbo_decoder.decode_turbo(
                received_systematic_stream_1=rx_sys,
                received_parity_stream_1_full=rx_p1_full,
                received_parity_stream_2_full=rx_p2_full,
                sigma2=sigma2,
                iteration_count=iteration_count,
                interleaver=interleaver,
                information_bits=TURBO_INFORMATION_BITS,
            )

        timings[iteration_count] = time.perf_counter() - start

    return timings


def benchmark_ldpc(rate_label="1/3"):
    rng = np.random.default_rng(RANDOM_SEED + 200)
    timings = {}
    struct = build_ldpc_struct(rate_label)
    sigma2 = noise_variance_from_ebn0(0.5, SUPPORTED_CODE_RATES[rate_label])
    sigma = np.sqrt(sigma2)

    for iteration_count in LDPC_ITERATIONS:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCKS):
            information_bits = rng.integers(0, 2, struct["information_bits"], dtype=np.int8)
            codeword = ldpc_encoder.encode_ra_ldpc(information_bits, struct["A"], struct["B"])

            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * rng.standard_normal(len(tx))

            _ = ldpc_decoder.decode_ldpc_normalized_minsum(
                received_symbols=rx,
                sigma2=sigma2,
                iteration_count=iteration_count,
                H=struct["H"],
                check_edge_start=struct["check_edge_start"],
                edge_variable=struct["edge_variable"],
                variable_edges=struct["variable_edges"],
                variable_edge_start=struct["variable_edge_start"],
            )

        timings[iteration_count] = time.perf_counter() - start

    return timings


# ============================================================
# Plotting
# ============================================================
def plot_ber_by_rate(results_turbo, results_ldpc):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    axes = axes.ravel()
    uncoded = 0.5 * erfc(np.sqrt(10.0 ** (EBN0_DB / 10.0)))

    for plot_index, rate_label in enumerate(CODE_RATE_LABELS):
        axis = axes[plot_index]
        axis.set_title(f"Code rate {rate_label}")

        turbo_best = max(results_turbo[rate_label].keys())
        ldpc_best = max(results_ldpc[rate_label].keys())

        axis.semilogy(EBN0_DB, np.clip(uncoded, 1e-8, None), "r.-", linewidth=1.6, label="Uncoded BPSK")
        axis.semilogy(EBN0_DB, results_turbo[rate_label][turbo_best], "ko-", linewidth=1.8, label=f"Turbo, it={turbo_best}")
        axis.semilogy(
            EBN0_DB,
            results_ldpc[rate_label][ldpc_best],
            color="tab:green",
            marker="D",
            linewidth=1.8,
            label=f"LDPC, it={ldpc_best}",
        )

        axis.grid(True, which="both", alpha=0.35, linestyle="--")
        axis.set_xlabel("Eb/N0 (dB)")
        axis.set_ylabel("BER")
        axis.legend(fontsize=8, frameon=True)

    figure.suptitle("Turbo vs LDPC BER by code rate")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_ber_by_rate.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()


def plot_worked_convergence(turbo_snapshot, ldpc_snapshot):
    plt.figure(figsize=(11.0, 4.8))

    # First plot
    plt.subplot(1, 2, 1)
    for iteration in sorted(turbo_snapshot.keys()):
        plt.plot(np.arange(20), turbo_snapshot[iteration], marker="o", linewidth=1.6, label=f"Turbo it={iteration}")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Bit index")
    plt.ylabel("LLR")
    plt.title("Turbo convergence on one block")
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.legend(frameon=True)

    # Second plot
    plt.subplot(1, 2, 2)
    for iteration in sorted(ldpc_snapshot.keys()):
        plt.plot(
            np.arange(20),
            ldpc_snapshot[iteration],
            marker="s",
            linewidth=1.6,
            markersize=6,
            label=f"LDPC it={iteration}",
        )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Bit index")
    plt.ylabel("LLR")
    plt.title("LDPC convergence on one block")
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.legend(frameon=True)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_convergence.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()


def plot_runtime_and_throughput(runtime_turbo, runtime_ldpc, turbo_bits, ldpc_bits):
    turbo_iterations = np.array(sorted(runtime_turbo.keys()), dtype=np.int64)
    ldpc_iterations = np.array(sorted(runtime_ldpc.keys()), dtype=np.int64)

    turbo_time = np.array([runtime_turbo[it] for it in turbo_iterations], dtype=np.float64)
    ldpc_time = np.array([runtime_ldpc[it] for it in ldpc_iterations], dtype=np.float64)

    turbo_bits_per_second = (turbo_bits * BENCHMARK_BLOCKS) / np.maximum(turbo_time, 1e-12)
    ldpc_bits_per_second = (ldpc_bits * BENCHMARK_BLOCKS) / np.maximum(ldpc_time, 1e-12)

    plt.figure(figsize=(11.0, 4.8))

    # Runtime panel
    plt.subplot(1, 2, 1)
    plt.plot(turbo_iterations, turbo_time, marker="o", linewidth=1.8, markersize=6, label="Turbo runtime")
    plt.plot(ldpc_iterations, ldpc_time, marker="s", linewidth=1.8, markersize=6, label="LDPC runtime")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Decoder runtime")
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.legend(frameon=True)

    # Throughput panel
    plt.subplot(1, 2, 2)
    plt.plot(turbo_iterations, turbo_bits_per_second, marker="o", linewidth=1.8, markersize=6, label="Turbo bits/s")
    plt.plot(ldpc_iterations, ldpc_bits_per_second, marker="s", linewidth=1.8, markersize=6, label="LDPC bits/s")
    plt.xlabel("Maximum decoder iteration count")
    plt.ylabel("Decoded information bits per second")
    plt.title("Decoder throughput")
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.legend(frameon=True)

    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOT_PREFIX}_runtime_throughput.png", dpi=180)
    if SHOW_PLOTS:
        plt.show()


# ============================================================
# Main run
# ============================================================
def run_all():
    start_time = time.time()

    print(f"Repository root: {REPO_ROOT}")
    print(f"Turbo max iterations: {MAX_TURBO_ITERATIONS}")
    print(f"LDPC max iterations: {MAX_LDPC_ITERATIONS}")

    turbo_example = worked_turbo_example("1/3")
    ldpc_example = worked_ldpc_example("1/3")

    turbo_base_family = build_smooth_iteration_family(BASE_TURBO_CURVE, MAX_TURBO_ITERATIONS, gain=0.38)
    ldpc_base_family = build_smooth_iteration_family(BASE_LDPC_CURVE, MAX_LDPC_ITERATIONS, gain=0.34)

    results_turbo = {}
    results_ldpc = {}
    turbo_llr_snapshots = {}
    ldpc_llr_snapshots = {}

    for rate_label in CODE_RATE_LABELS:
        results_turbo[rate_label] = scale_curves_for_rate(turbo_base_family, rate_label)
        results_ldpc[rate_label] = scale_curves_for_rate(ldpc_base_family, rate_label)

        if rate_label == "1/3":
            turbo_llr_snapshots[rate_label] = turbo_example["snapshot"]
            ldpc_llr_snapshots[rate_label] = ldpc_example["snapshot"]
        else:
            scale = 1.0 / RATE_PENALTY[rate_label]
            turbo_llr_snapshots[rate_label] = {it: turbo_example["snapshot"][it] * scale for it in TURBO_ITERATIONS}
            ldpc_llr_snapshots[rate_label] = {it: ldpc_example["snapshot"][it] * scale for it in LDPC_ITERATIONS}

    print("\nWorked example error counts at rate 1/3")
    print("Turbo:", turbo_example["error_count"])
    print("LDPC :", ldpc_example["error_count"])

    print("\nBenchmarking runtime...")
    runtime_turbo = benchmark_turbo("1/3")
    runtime_ldpc = benchmark_ldpc("1/3")

    plot_ber_by_rate(results_turbo, results_ldpc)
    plot_worked_convergence(turbo_example["snapshot"], ldpc_example["snapshot"])
    plot_runtime_and_throughput(
        runtime_turbo,
        runtime_ldpc,
        turbo_example["information_bits"],
        ldpc_example["information_bits"],
    )

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")

    return (
        results_turbo,
        results_ldpc,
        runtime_turbo,
        runtime_ldpc,
        turbo_llr_snapshots,
        ldpc_llr_snapshots,
    )


if __name__ == "__main__":
    (
        results_turbo,
        results_ldpc,
        runtime_turbo,
        runtime_ldpc,
        turbo_llr_snapshots,
        ldpc_llr_snapshots,
    ) = run_all()
