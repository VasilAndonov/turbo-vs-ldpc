import numpy as np
from ldpc.config import INFORMATION_BITS, SUPPORTED_CODE_RATES, RANDOM_SEED

def build_ldpc_parameters(rate_label):
    if rate_label == "1/3":
        return dict(column_weight=3)
    if rate_label == "1/2":
        return dict(column_weight=3)
    if rate_label == "3/4":
        return dict(column_weight=4)
    if rate_label == "7/8":
        return dict(column_weight=5)
    raise ValueError(rate_label)

def build_ra_ldpc_matrices(rate_label):
    params = build_ldpc_parameters(rate_label)
    column_weight = params["column_weight"]

    code_rate = SUPPORTED_CODE_RATES[rate_label]
    parity_bits = int(round(INFORMATION_BITS * (1.0 / code_rate - 1.0)))
    codeword_bits = INFORMATION_BITS + parity_bits

    rng = np.random.default_rng(RANDOM_SEED + abs(hash(rate_label)) % 1000)

    A = np.zeros((parity_bits, INFORMATION_BITS), dtype=np.int8)
    row_load = np.zeros(parity_bits, dtype=int)

    for col in range(INFORMATION_BITS):
        chosen_rows = []
        preferred = [(col * column_weight + offset * (parity_bits // max(column_weight, 1) + 1)) % parity_bits for offset in range(column_weight)]
        for candidate in preferred:
            window = [(candidate + shift) % parity_bits for shift in (-2, -1, 0, 1, 2)]
            best_row = None
            best_score = None
            for row in window:
                if row in chosen_rows:
                    continue
                score = row_load[row]
                if best_score is None or score < best_score:
                    best_score = score
                    best_row = row
            chosen_rows.append(best_row)

        chosen_rows = sorted(set(chosen_rows))
        while len(chosen_rows) < column_weight:
            remaining = np.argsort(row_load)
            for row in remaining:
                if row not in chosen_rows:
                    chosen_rows.append(int(row))
                    break

        for row in chosen_rows[:column_weight]:
            A[row, col] = 1
            row_load[row] += 1

    # Accumulator matrix B: lower bidiagonal
    B = np.zeros((parity_bits, parity_bits), dtype=np.int8)
    for row in range(parity_bits):
        B[row, row] = 1
        if row > 0:
            B[row, row - 1] = 1

    H = np.concatenate([A, B], axis=1)
    return H, A, B, codeword_bits, parity_bits

def build_edge_structure(H):
    check_count, variable_count = H.shape
    edge_variable = []
    check_edge_start = np.zeros(check_count + 1, dtype=np.int64)
    variable_neighbors = [[] for _ in range(variable_count)]

    edge_index = 0
    for check_index in range(check_count):
        variable_indices = np.where(H[check_index] == 1)[0]
        check_edge_start[check_index] = edge_index
        for variable_index in variable_indices:
            edge_variable.append(variable_index)
            variable_neighbors[variable_index].append(edge_index)
            edge_index += 1
    check_edge_start[check_count] = edge_index

    variable_edge_start = np.zeros(variable_count + 1, dtype=np.int64)
    variable_edges = np.zeros(edge_index, dtype=np.int64)
    fill_index = 0
    for variable_index in range(variable_count):
        variable_edge_start[variable_index] = fill_index
        for edge in variable_neighbors[variable_index]:
            variable_edges[fill_index] = edge
            fill_index += 1
    variable_edge_start[variable_count] = fill_index

    return np.asarray(edge_variable, dtype=np.int64), check_edge_start, variable_edges, variable_edge_start

def encode_ra_ldpc(information_bits, A, B):
    information_bits = np.asarray(information_bits, dtype=np.int8)
    syndrome = np.zeros(A.shape[0], dtype=np.int8)

    # syndrome = A * u over GF(2)
    for row in range(A.shape[0]):
        value = 0
        nz = np.where(A[row] == 1)[0]
        for col in nz:
            value ^= int(information_bits[col])
        syndrome[row] = value

    # B is lower bidiagonal => p[0]=s[0], p[i]=s[i] xor p[i-1]
    parity = np.zeros(B.shape[0], dtype=np.int8)
    if len(parity) > 0:
        parity[0] = syndrome[0]
        for index in range(1, len(parity)):
            parity[index] = syndrome[index] ^ parity[index - 1]

    return np.concatenate([information_bits, parity]).astype(np.int8)
