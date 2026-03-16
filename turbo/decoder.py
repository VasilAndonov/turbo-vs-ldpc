import numpy as np
from turbo.encoder import INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATE, NEXT_STATES
from turbo.config import TRACEBACK_DEPTH

def maxlogmap_decode(systematic_llr, parity_llr, apriori_llr):
    n = len(systematic_llr)
    state_count = INPUT_SIGN.shape[0]
    neg_inf = -1e15

    alpha = np.full((n + 1, state_count), neg_inf, dtype=float)
    beta = np.full((n + 1, state_count), neg_inf, dtype=float)
    gamma0 = np.zeros((n, state_count), dtype=float)
    gamma1 = np.zeros((n, state_count), dtype=float)

    alpha[0, 0] = 0.0
    beta[n, 0] = 0.0

    for k in range(n):
        for s in range(state_count):
            gamma0[k, s] = 0.5 * ((systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 0] + parity_llr[k] * PARITY_SIGN[s, 0])
            gamma1[k, s] = 0.5 * ((systematic_llr[k] + apriori_llr[k]) * INPUT_SIGN[s, 1] + parity_llr[k] * PARITY_SIGN[s, 1])

    for k in range(n):
        for ns in range(state_count):
            best = neg_inf
            for idx, prev_state in enumerate(PREV_STATES[ns][0]):
                input_bit = PREV_STATES[ns][1][idx]
                candidate = alpha[k, prev_state] + (gamma0[k, prev_state] if input_bit == 0 else gamma1[k, prev_state])
                if candidate > best:
                    best = candidate
            alpha[k + 1, ns] = best

    for k in range(n - 1, -1, -1):
        for state in range(state_count):
            best = neg_inf
            for idx, next_state in enumerate(NEXT_STATES[state][0]):
                input_bit = NEXT_STATES[state][1][idx]
                candidate = beta[k + 1, next_state] + (gamma0[k, state] if input_bit == 0 else gamma1[k, state])
                if candidate > best:
                    best = candidate
            beta[k, state] = best

    posterior = np.zeros(n, dtype=float)
    extrinsic = np.zeros(n, dtype=float)

    for k in range(n):
        best0 = neg_inf
        best1 = neg_inf
        for state in range(state_count):
            ns0 = NEXT_STATE[state, 0]
            ns1 = NEXT_STATE[state, 1]
            v0 = alpha[k, state] + gamma0[k, state] + beta[k + 1, ns0]
            v1 = alpha[k, state] + gamma1[k, state] + beta[k + 1, ns1]
            if v0 > best0:
                best0 = v0
            if v1 > best1:
                best1 = v1
        posterior[k] = best0 - best1
        extrinsic[k] = posterior[k] - systematic_llr[k] - apriori_llr[k]

    return posterior, extrinsic

def decode_turbo(received_systematic_stream_1, received_parity_stream_1_full, received_parity_stream_2_full, sigma2, iteration_count, interleaver, information_bits):
    total_len = len(received_systematic_stream_1)
    channel_reliability = 2.0 / sigma2

    sys1_llr = channel_reliability * received_systematic_stream_1
    par1_llr = channel_reliability * received_parity_stream_1_full
    par2_llr = channel_reliability * received_parity_stream_2_full

    apriori1 = np.zeros(total_len, dtype=float)
    llr_history = np.zeros((iteration_count, information_bits), dtype=float)
    extrinsic_scale = 0.75

    for iteration_index in range(iteration_count):
        _, ext1 = maxlogmap_decode(sys1_llr, par1_llr, apriori1)

        apriori2 = np.zeros(total_len, dtype=float)
        sys2_llr = np.zeros(total_len, dtype=float)
        for bit_index in range(information_bits):
            apriori2[bit_index] = extrinsic_scale * ext1[interleaver[bit_index]]
            sys2_llr[bit_index] = sys1_llr[interleaver[bit_index]]

        post2_int, ext2_int = maxlogmap_decode(sys2_llr, par2_llr, apriori2)

        ext_info = np.zeros(information_bits, dtype=float)
        for bit_index in range(information_bits):
            llr_history[iteration_index, interleaver[bit_index]] = post2_int[bit_index]
            ext_info[interleaver[bit_index]] = ext2_int[bit_index]

        for bit_index in range(information_bits):
            apriori1[bit_index] = extrinsic_scale * ext_info[bit_index]

    return llr_history

def viterbi_decode_75(received_symbols, information_bits):
    received_symbols = np.asarray(received_symbols, dtype=float)
    num_steps = len(received_symbols) // 2
    num_states = 4
    large = 1e30

    path_metric = np.full((num_steps + 1, num_states), large, dtype=float)
    prev_state = np.full((num_steps + 1, num_states), -1, dtype=int)
    prev_input = np.full((num_steps + 1, num_states), -1, dtype=int)
    path_metric[0, 0] = 0.0

    for step in range(num_steps):
        r0 = received_symbols[2 * step]
        r1 = received_symbols[2 * step + 1]
        for state in range(num_states):
            metric = path_metric[step, state]
            if metric > 1e20:
                continue
            s0 = (state >> 1) & 1
            s1 = state & 1
            for u in (0, 1):
                c0 = u ^ s0 ^ s1
                c1 = u ^ s1
                next_state = (u << 1) | s0
                e0 = 1.0 if c0 == 0 else -1.0
                e1 = 1.0 if c1 == 0 else -1.0
                branch_metric = (r0 - e0) ** 2 + (r1 - e1) ** 2
                candidate = metric + branch_metric
                if candidate < path_metric[step + 1, next_state]:
                    path_metric[step + 1, next_state] = candidate
                    prev_state[step + 1, next_state] = state
                    prev_input[step + 1, next_state] = u

    state = 0
    decoded = []
    for step in range(num_steps, 0, -1):
        decoded.append(prev_input[step, state])
        state = prev_state[step, state]
    decoded.reverse()
    return np.array(decoded[:information_bits], dtype=np.int8)
