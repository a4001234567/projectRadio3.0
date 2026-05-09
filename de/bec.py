from typing import Tuple, Dict
from de.recipe import generate_books_from_recipe

# recipe: A map from column type to the number of columns of that type
#   where column type is the weights of each block in that column.

# book: A map from degree to the number of nodes with that degree
def find_threshold(xs,ys,thres = 1e-15):
    for x,y in zip(xs,ys):
        if y >= thres:
            return x
    raise ValueError(f"Threshold not found for xs={xs} and ys={ys}")

def generate_func_from_book(book):
    sum_coef = sum(((deg-1)*coef for deg,coef in book.items()))
    return lambda x: sum(((deg-1)*coef*pow(x,deg-1) for deg,coef in book.items()))/sum_coef

def findThresForRecipe(recipe:Dict[Tuple[int,...],int],block_size:int,truncate:int)->float:
    chk_book, var_book = generate_books_from_recipe(recipe,block_size,truncate)
    func_lambda = generate_func_from_book(var_book)
    func_rho = generate_func_from_book(chk_book)
    def f(eps,x):
        return eps*func_lambda(1-func_rho(1-x))
    def fk(k,eps):
        res = 1
        for _ in range(k):
            res = f(eps,res)
        return res

    xs = [i/1000 for i in range(1000)]
    try:
        ys = [fk(50,t) for t in xs]
    except Exception as e:
        raise e
    try:
        return find_threshold(xs,ys)
    except ValueError as e:
        raise ValueError(f"Threshold not found for recipe={recipe}, block_size={block_size}, truncate={truncate}")

def find_thres_from_books(var_book:Dict[int,int], chk_book:Dict[int,int])->float:
    func_lambda = generate_func_from_book(var_book)
    func_rho = generate_func_from_book(chk_book)
    def f(eps,x):
        return eps*func_lambda(1-func_rho(1-x))
    def fk(k,eps):
        res = 1
        for _ in range(k):
            res = f(eps,res)
        return res

    xs = [i/1000 for i in range(1000)]
    try:
        ys = [fk(50,t) for t in xs]
    except Exception as e:
        raise e
    try:
        return find_threshold(xs,ys)
    except ValueError as e:
        raise ValueError(f"Threshold not found for var_book={var_book}, chk_book={chk_book}")