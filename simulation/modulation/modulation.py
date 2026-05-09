import numpy as np
def modulation(x, N, constellation, rho_inv, base_vec, demod_indices, is2D, m, M):
    if not(constellation):
        return 1-2*x
    else:
        numbers = x * base_vec
        return constellation[rho_inv(1+np.sum(numbers.reshape(m,N/m)))]
    