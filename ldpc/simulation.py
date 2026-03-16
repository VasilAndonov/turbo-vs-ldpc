import time
import numpy as np
from ldpc.config import RANDOM_SEED, INFORMATION_BITS, ITERATIONS, LDPC_EBN0_DB, MIN_FRAMES, MAX_FRAMES, TARGET_ERRORS, BENCHMARK_BLOCKS, sigma2_from_ebn0, SELECTED_CODE_RATE_LABEL, SUPPORTED_CODE_RATES
from ldpc.encoder import build_ra_ldpc_matrices, build_edge_structure, encode_ra_ldpc
from ldpc.decoder import decode_ldpc_normalized_minsum

def simulate_ldpc(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED)
    H, A, B, codeword_bits, _ = build_ra_ldpc_matrices(rate_label)
    edge_variable, check_edge_start, variable_edges, variable_edge_start = build_edge_structure(H)

    ber = {it: [] for it in ITERATIONS}
    llr_snapshot = {it: None for it in ITERATIONS}
    code_rate = SUPPORTED_CODE_RATES[rate_label]

    for ebn0_db in LDPC_EBN0_DB:
        sigma2 = sigma2_from_ebn0(ebn0_db, code_rate)
        sigma = np.sqrt(sigma2)
        errors = {it: 0 for it in ITERATIONS}
        bits_total = 0
        frames = 0

        while frames < MIN_FRAMES or (errors[max(ITERATIONS)] < TARGET_ERRORS and frames < MAX_FRAMES):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            codeword = encode_ra_ldpc(bits, A, B)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * rng.standard_normal(codeword_bits)

            history = decode_ldpc_normalized_minsum(rx, sigma2, max(ITERATIONS), H, check_edge_start, edge_variable, variable_edges, variable_edge_start)

            for it in ITERATIONS:
                info_llr = history[it - 1][:INFORMATION_BITS]
                hard = (info_llr < 0.0).astype(np.int8)
                errors[it] += int(np.sum(bits != hard))

            if ebn0_db == LDPC_EBN0_DB[-1] and frames == 0:
                for it in ITERATIONS:
                    llr_snapshot[it] = history[it - 1][:20].copy()

            bits_total += INFORMATION_BITS
            frames += 1

        for it in ITERATIONS:
            ber[it].append(errors[it] / max(bits_total, 1))

    for it in ITERATIONS:
        ber[it] = np.array(ber[it], dtype=float)
    return ber, llr_snapshot

def benchmark_ldpc(rate_label=SELECTED_CODE_RATE_LABEL):
    rng = np.random.default_rng(RANDOM_SEED + 100)
    H, A, B, codeword_bits, _ = build_ra_ldpc_matrices(rate_label)
    edge_variable, check_edge_start, variable_edges, variable_edge_start = build_edge_structure(H)
    code_rate = SUPPORTED_CODE_RATES[rate_label]
    sigma2 = sigma2_from_ebn0(0.5, code_rate)
    sigma = np.sqrt(sigma2)
    timings = {}

    for it in ITERATIONS:
        start = time.perf_counter()
        for _ in range(BENCHMARK_BLOCKS):
            bits = rng.integers(0, 2, INFORMATION_BITS, dtype=np.int8)
            codeword = encode_ra_ldpc(bits, A, B)
            tx = 1.0 - 2.0 * codeword
            rx = tx + sigma * rng.standard_normal(codeword_bits)
            decode_ldpc_normalized_minsum(rx, sigma2, it, H, check_edge_start, edge_variable, variable_edges, variable_edge_start)
        timings[it] = time.perf_counter() - start
    return timings
