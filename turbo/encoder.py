import numpy as np
from turbo.config import INFORMATION_BITS, RSC_TAIL_LENGTH, RSC_STATE_COUNT, get_puncture_definition

def build_rsc_tables():
    next_state = np.zeros((RSC_STATE_COUNT, 2), dtype=int)
    parity = np.zeros((RSC_STATE_COUNT, 2), dtype=np.int8)
    input_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    parity_sign = np.zeros((RSC_STATE_COUNT, 2), dtype=float)
    prev_states = [[[] for _ in range(2)] for _ in range(RSC_STATE_COUNT)]
    next_states = [[[] for _ in range(2)] for _ in range(RSC_STATE_COUNT)]

    for state in range(RSC_STATE_COUNT):
        s0 = (state >> 1) & 1
        s1 = state & 1
        for u in (0, 1):
            recursive_bit = u ^ s1
            parity_bit = recursive_bit ^ s0 ^ s1
            ns = (recursive_bit << 1) | s0

            next_state[state, u] = ns
            parity[state, u] = parity_bit
            input_sign[state, u] = 1.0 if u == 0 else -1.0
            parity_sign[state, u] = 1.0 if parity_bit == 0 else -1.0

            prev_states[ns][0].append(state)
            prev_states[ns][1].append(u)
            next_states[state][0].append(ns)
            next_states[state][1].append(u)

    return next_state, parity, input_sign, parity_sign, prev_states, next_states

NEXT_STATE, PARITY, INPUT_SIGN, PARITY_SIGN, PREV_STATES, NEXT_STATES = build_rsc_tables()

def build_interleaver(seed: int):
    rng = np.random.default_rng(seed)
    p = rng.permutation(INFORMATION_BITS).astype(int)
    return p, np.argsort(p).astype(int)

def zero_termination_tail_bits(state: int):
    bits = []
    st = state
    for _ in range(RSC_TAIL_LENGTH):
        u = st & 1
        bits.append(u)
        st = NEXT_STATE[st, u]
    return np.array(bits, dtype=np.int8)

def encode_rsc_terminated(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    k = len(info_bits)
    n = k + RSC_TAIL_LENGTH

    systematic = np.zeros(n, dtype=np.int8)
    parity = np.zeros(n, dtype=np.int8)
    state = 0

    for index, bit_value in enumerate(info_bits):
        bit_value = int(bit_value)
        systematic[index] = bit_value
        parity[index] = PARITY[state, bit_value]
        state = NEXT_STATE[state, bit_value]

    tail_bits = zero_termination_tail_bits(state)
    for offset, bit_value in enumerate(tail_bits):
        bit_value = int(bit_value)
        systematic[k + offset] = bit_value
        parity[k + offset] = PARITY[state, bit_value]
        state = NEXT_STATE[state, bit_value]

    return systematic, parity

def turbo_encode_transmitted_symbols(info_bits, interleaver, rate_label):
    info_bits = np.asarray(info_bits, dtype=np.int8)

    systematic_stream, parity_stream_1 = encode_rsc_terminated(info_bits)
    _, parity_stream_2 = encode_rsc_terminated(info_bits[interleaver])

    total_len = len(systematic_stream)
    keep_pattern_1, keep_pattern_2 = get_puncture_definition(rate_label)

    keep_mask_1 = np.ones(total_len, dtype=np.int8)
    keep_mask_2 = np.ones(total_len, dtype=np.int8)
    keep_mask_1[:INFORMATION_BITS] = np.resize(keep_pattern_1, INFORMATION_BITS)
    keep_mask_2[:INFORMATION_BITS] = np.resize(keep_pattern_2, INFORMATION_BITS)

    return {
        "systematic_stream_1": systematic_stream,
        "parity_stream_1_full": parity_stream_1,
        "parity_stream_2_full": parity_stream_2,
        "transmitted_parity_stream_1": parity_stream_1[keep_mask_1 == 1],
        "transmitted_parity_stream_2": parity_stream_2[keep_mask_2 == 1],
        "parity_keep_mask_1": keep_mask_1,
        "parity_keep_mask_2": keep_mask_2,
    }

def depuncture_received_parity(received, keep_mask):
    full = np.zeros(len(keep_mask), dtype=float)
    tx_index = 0
    for index in range(len(keep_mask)):
        if keep_mask[index] == 1:
            full[index] = received[tx_index]
            tx_index += 1
    return full

def conv_encode_75(info_bits):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    padded = np.concatenate([info_bits, np.zeros(2, dtype=np.int8)])
    s0 = 0
    s1 = 0
    out = []
    for u in padded:
        u = int(u)
        c0 = u ^ s0 ^ s1
        c1 = u ^ s1
        out.extend((c0, c1))
        s1 = s0
        s0 = u
    return np.array(out, dtype=np.int8)
