import numpy as np
def demodulation(y, sigma, N, constellation, rho_inv, base_vec, demod_indices, is2D, m, M):
    if not constellation:#Constellation is empty, for BPSK
        llr = 2. * y / np.power(sigma,2)
    else:
        raise NotImplemented
        p0 = np.zeros((N,1))
        p1 = np.zeros((N,1))
        for iy in range(len(y)):
            index_y = m*iy
            p_ylx = np.exp(-np.pow(abs(y[iy]-constellation)/sigma,2)/2)
            sum_p_ylx = p_ylx.sum()
            for i_m in range(m):
                index_m = M/2 * (i_m -1)
                for i_c in range(M//2):
                    p1[index_y+i_m] += p_ylx[demod_indices[index_m+i_c]]
                p0[index_y+i_m] = sum_p_ylx - p1[index_y+i_m]
        llr = np.log(p0/p1)
    return llr
