from fileio.matrix_io import get_reader
import argparse
from utils.poly_utils import polynomial_repr, poly_gcd
from itertools import combinations

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process some integers and a filename.")
    
    # Add arguments
    parser.add_argument('filename', type=str, help="The input filename")
    parser.add_argument('j', type=int, help="Integer j")
    parser.add_argument('l', type=int, help="Integer l")
    parser.add_argument('block_size', type=int, help="Block size (integer)")
    parser.add_argument('truncate', type=int, default=1, help="Truncation")

    # Parse arguments
    args = parser.parse_args()
    reader = get_reader()
    res_matrix = reader(args.filename)
    n_cols,n_rows = args.l,args.j
    p,t = args.block_size, args.truncate
    h,w = res_matrix.shape
    assert h == n_rows*(p-t),f'{h},{n_rows},{p},{t}'
    assert w == n_cols*(p-t)
    polys = []
    for block_idx in range(n_cols):
        res = 0
        for sub_idx in range(p-t):
            if res_matrix[0,sub_idx+(p-t)*block_idx]:
                res |= 1<<sub_idx
        for sub_idx in range(p-t,p):
            if res_matrix[sub_idx-(p-t)+1,(p-t)*block_idx]:
                res |= 1<<sub_idx
        polys.append(res)
    return polys,args

def xor_sum(*seqs):
    if seqs:
        return seqs[0]^xor_sum(*seqs[1:])
    return 0

if __name__ == "__main__":
    polys,args = main()
    main_poly = (1<<args.block_size)+1
    for poly in polys:
        print(polynomial_repr(poly))
    for subset_size in range(1,args.j+1):
        for poly_subset in combinations(polys,subset_size):
            if poly_gcd(main_poly,xor_sum(*poly_subset)) != 2+1:
                print('GCD Break')
                print(polynomial_repr(poly_gcd(main_poly,xor_sum(*poly_subset))))
                print(','.join((polynomial_repr(i) for i in poly_subset)))
                exit()
    print('GCD Hold')
