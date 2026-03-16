import numpy as np

def compute_syndrome(H, hard_bits):
    syndrome = np.zeros(H.shape[0], dtype=np.int8)
    for row in range(H.shape[0]):
        value = 0
        for col in np.where(H[row] == 1)[0]:
            value ^= int(hard_bits[col])
        syndrome[row] = value
    return syndrome

def decode_ldpc_normalized_minsum(received_symbols, sigma2, iteration_count, H, check_edge_start, edge_variable, variable_edges, variable_edge_start, llr_clip=20.0, alpha=0.8, damping=0.1):
    variable_count = len(received_symbols)
    edge_count = len(edge_variable)

    channel_llr = np.clip((2.0 / sigma2) * received_symbols, -llr_clip, llr_clip)

    variable_to_check = np.zeros(edge_count, dtype=float)
    check_to_variable = np.zeros(edge_count, dtype=float)
    posterior_history = np.zeros((iteration_count, variable_count), dtype=float)

    for edge in range(edge_count):
        variable_to_check[edge] = channel_llr[edge_variable[edge]]

    for iteration_index in range(iteration_count):
        # Check node update: normalized min-sum
        new_check_to_variable = np.zeros(edge_count, dtype=float)
        for check_index in range(len(check_edge_start) - 1):
            start = check_edge_start[check_index]
            end = check_edge_start[check_index + 1]
            degree = end - start
            if degree <= 1:
                continue

            signs = np.ones(degree, dtype=float)
            mags = np.zeros(degree, dtype=float)
            total_sign = 1.0
            min1 = 1e9
            min2 = 1e9
            min1_pos = -1

            for local in range(degree):
                msg = variable_to_check[start + local]
                if msg < 0:
                    signs[local] = -1.0
                    total_sign *= -1.0
                mags[local] = abs(msg)
                if mags[local] < min1:
                    min2 = min1
                    min1 = mags[local]
                    min1_pos = local
                elif mags[local] < min2:
                    min2 = mags[local]

            for local in range(degree):
                use_min = min2 if local == min1_pos else min1
                sign = total_sign * signs[local]
                updated = alpha * sign * use_min
                old = check_to_variable[start + local]
                new_check_to_variable[start + local] = (1.0 - damping) * updated + damping * old

        check_to_variable = np.clip(new_check_to_variable, -llr_clip, llr_clip)

        posterior = channel_llr.copy()
        for variable_index in range(variable_count):
            start = variable_edge_start[variable_index]
            end = variable_edge_start[variable_index + 1]
            for local in range(start, end):
                edge = variable_edges[local]
                posterior[variable_index] += check_to_variable[edge]
        posterior = np.clip(posterior, -llr_clip, llr_clip)
        posterior_history[iteration_index, :] = posterior

        hard = (posterior < 0.0).astype(np.int8)
        if np.all(compute_syndrome(H, hard) == 0):
            for later in range(iteration_index + 1, iteration_count):
                posterior_history[later, :] = posterior
            return posterior_history

        for variable_index in range(variable_count):
            start = variable_edge_start[variable_index]
            end = variable_edge_start[variable_index + 1]
            for local in range(start, end):
                edge = variable_edges[local]
                variable_to_check[edge] = posterior[variable_index] - check_to_variable[edge]
        variable_to_check = np.clip(variable_to_check, -llr_clip, llr_clip)

    return posterior_history
