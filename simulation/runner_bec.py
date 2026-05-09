import numpy as np
from time import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from fileio.matrix_io import get_reader
from simulation.decoders.peeling import jitpeeling_decoder as PeelingDecoder
import os
from time import time_ns

#filename = "TestHugeRate/2*120*599trunc.zip"
#filename = "essayExamples/XPG4*32*(641-0).zip"
#filename = "TestHugeRate/120.zip"
#H = get_reader()(filename)[:,:2856]
#filename = 'ISIT/ISIT_EX2_CUL.zip'
filename = 'ISIT/ISIT_example2_CUL.zip'
H:np.ndarray = get_reader()(filename)
print(H.shape)
K,N = H.shape
R = 1-K/N

error_rates = []
error_rates = [.087,.0875,.088,.0885,.089,.0895,.09,.0905]
error_rates = (.089,.0895,.09,.0905,.091,.092,.093,.094,.095,.096,.097,.098,.099,.1,.101)
#error_rates = [.21,.215,.22,.225,.23,.235,.24,.245,.25]
error_rates = [0.17,0.175,0.18,0.185,0.19,0.195,0.20,0.205,0.21]
#error_rates = [0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.20,0.205,0.21]

max_iter = 50
max_runs = int(1e7)
resolution = int(1e5)
max_err = 200

num_block_err = np.zeros((len(error_rates),1))
num_bit_err = np.zeros((len(error_rates),1))
num_iter = np.zeros((len(error_rates),1))
num_runs = np.zeros((len(error_rates),1))
H_matrix = H.astype('bool')

def init_worker():
    seed = ((1<<32)-1) & (os.getpid() ^ time_ns())
    seed ^= ((1<<32)-1) & (os.getpid() ^ time_ns())-1
    np.random.seed(seed)

round_per_sim = 1000
def simulate_multi_round(Err_collected):
    cur_had_runs = []
    cur_bit_errs = []
    cur_block_errs = []
    cur_iters = []
    for _ in range(round_per_sim):
        cur_had_run = [False for _ in error_rates]
        cur_block_err = [0 for _ in error_rates]
        cur_bit_err = [0 for _ in error_rates]
        cur_iter = [0 for _ in error_rates]

        x = np.random.rand(N)
        for i_error_rate,error_rate in enumerate(error_rates):
            if Err_collected[i_error_rate] >= max_err:
                continue
            mask = (x>1-error_rate).astype(bool)
            iter_this_time,bit_errors = PeelingDecoder(H_matrix,mask)
            cur_iter[i_error_rate] = iter_this_time
            if bit_errors:
                cur_had_run[i_error_rate] = False
                cur_block_err[i_error_rate] += 1
                cur_bit_err[i_error_rate] += bit_errors
                Err_collected[i_error_rate] += 1
            else:
                cur_had_run[i_error_rate] = True
        cur_had_runs.append(cur_had_run)
        cur_bit_errs.append(cur_bit_err)
        cur_block_errs.append(cur_block_err)
        cur_iters.append(cur_iter)
    return cur_had_runs, cur_bit_errs, cur_block_errs, cur_iters

if __name__ == '__main__':
    #mp.set_start_method("spawn")
    freeze_support()
    start = time()
    num_proc = int(cpu_count()*0.8)
    try:
        with Manager() as manager:
            Err_collected = manager.list([0 for _ in range(len(error_rates))])
            cnt = 0
            with Pool(num_proc, initializer=init_worker) as pool:
                for res in pool.imap_unordered(simulate_multi_round, (Err_collected for _ in range(max_runs//round_per_sim))):
                    for had_run, bit_err, block_err, iters in zip(*res):
                        for idx,(i_had_run, i_bit_err, i_block_err, i_iter) in enumerate(zip(had_run, bit_err, block_err, iters)):
                            if num_block_err[idx] >= max_err:
                                continue
                            if i_had_run == False:
                                num_block_err[idx] += 1
                                num_bit_err[idx] += i_bit_err
                                num_iter[idx] += i_iter
                                num_runs[idx] += 1
                            if i_had_run == True: # No error
                                num_runs[idx] += 1
                                num_iter[idx] += i_iter
                    if all(num_block_err >= max_err):
                        pool.terminate()
                        break
                    cnt += 1
                    if (cnt*round_per_sim)%resolution == 0:
                        rounds_already = cnt * round_per_sim
                        remain_rounds = 0
                        for i in num_block_err.reshape(-1):
                            if 0 == i:
                                remain_rounds = max_runs
                                continue
                            remain_rounds = max(remain_rounds, int((max_err-i)/i*rounds_already))
                        remain_rounds = min(remain_rounds,max_runs - rounds_already)
                        remain_time = (time()-start)/rounds_already*remain_rounds
                        print(f'Time elapsed: {time()-start}')
                        print(f'Expected Time Remaining: {remain_time} seconds')
                        print(f'Already ran {rounds_already} round of simulation')
                        print(f'Collected Errors: '+"\t".join(map(str,num_block_err)))
    except KeyboardInterrupt:
        print('Abort')


    print(f'Time elapsed: {time()-start}')

    print('BEC simulation is finished')
    bler = np.sum(num_block_err,axis=1)/np.sum(num_runs,axis=1)
    ber = np.sum(num_bit_err,axis=1)/np.sum(num_runs,axis=1)/N
    ave_iter = np.sum(num_iter,axis=1)/np.sum(num_runs,axis=1)
    print('Erasure :'+'\t'.join(map(str,error_rates)))
    print('num_runs:'+'\t'.join(map(str,num_runs.reshape(-1))))
    print('ave_bler:'+'\t'.join(map(str,bler)))
    print('ave_ber :'+'\t'.join(map(str,ber)))
    print('ave_iter:'+'\t'.join(map(str,ave_iter)))
    with open(f'{filename}_BEC_output.txt','a') as file:
        write = lambda x:file.write(x+'\n')
        write('Erasure :'+'\t'.join(map(str,error_rates)))
        write('num_runs:'+'\t'.join(map(str,num_runs.reshape(-1))))
        write('ave_bler:'+'\t'.join(map(str,bler)))
        write('ave_ber :'+'\t'.join(map(str,ber)))
        write('ave_iter:'+'\t'.join(map(str,ave_iter)))
