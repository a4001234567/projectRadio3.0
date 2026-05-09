from fileio.matrix_io import get_reader
import argparse
import numpy as np

def counts(items:list)->dict:
    books = dict()
    for item in items:
        books[item] = books.get(item,0)+1
    return books

def generate_func_from_books(books):
    sum_coef = sum(((deg-1)*coef for deg,coef in books.items()))
    return lambda x:sum(((deg-1)*coef*pow(x,deg-1) for deg,coef in books.items()))/sum_coef

def find_threshold(xs,ys,thres=1e-15):
    for x,y in zip(xs,ys):
        if y >= thres:
            return x
    return None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Density Evolution Calculation for LDPC")
    parser.add_argument('input', nargs='+', type=str, help="input filename")
    parser.add_argument('--output', '-o', type=str, help="output filename")

    args = parser.parse_args()
    reader = get_reader()

    for file in args.input:
        H_matrix = reader(file)
        print(H_matrix.shape)
        degrees = H_matrix.sum(axis=0)
        varbooks = counts(degrees)
        print('Variable nodes distribution')
        for var in sorted(varbooks.keys()):
            print(f'{int(var)}:{varbooks[var]}')
        degrees = H_matrix.sum(axis=1)
        chkbooks = counts(degrees)
        print('Check nodes distribution')
        for chk in sorted(chkbooks.keys()):
            print(f'{int(chk)}:{chkbooks[chk]}')
        func_lambda = generate_func_from_books(varbooks)
        func_rho = generate_func_from_books(chkbooks)

        def f(eps,x):
            return eps*func_lambda(1-func_rho(1-x))

        def fk(k,eps):
            res = 1
            for _ in range(k):
                res = f(eps,res)
            return res

        xs = [i/2000 for i in range(2000)]
        ys = [fk(100,t) for t in xs]
        
        eps_thres = find_threshold(xs,ys)
        print(f'thres:{eps_thres:.3f}')
