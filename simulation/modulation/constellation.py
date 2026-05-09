def get_constellation(name:str, N:int):
    if name == 'BPSK':
        constellation = []
        rho_inv = []
        base_vec = []
        demod_indices = []
        is2D = False
        m = 1 #bits per symbol?
        M = 2**m
        return constellation, rho_inv, base_vec, demod_indices, is2D, m, M
    else:
        raise NotImplementedError(f'Constellation {name} is not implemented')
