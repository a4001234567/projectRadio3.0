from itertools import combinations
from typing import Dict, Tuple, Iterable
from utils.combinatorics import allSubsets

def add_to_books(books:Dict[int,int],degree:int,num:int)->None:
    if degree in books:
        books[degree] += num
    else:
        books[degree] = num

def generate_books_from_recipe(recipe:Dict[Tuple[int,...],int],block_size:int,truncate:int)->Tuple[Dict[int,int],Dict[int,int]]:
    num_rows = recipe.keys().__iter__().__next__().__len__()
    main_degrees_chk = list((0 for _ in range(num_rows)))
    missings = list((0 for _ in range(num_rows)))
    var_books = dict()
    chk_books = dict()
    for dish,num in recipe.items():
        if 0 == num: continue
        main_degree_var = 0;missing_var = 0
        for idx,weight in enumerate(dish):
            main_degrees_chk[idx] += weight*num
            missings[idx] += num*weight*truncate
            main_degree_var += weight
            missing_var += weight*truncate
        total_var_node = block_size-truncate
        add_to_books(var_books,main_degree_var,num*(total_var_node-missing_var))
        add_to_books(var_books,main_degree_var-1,num*missing_var)
    for main_degree,missing in zip(main_degrees_chk,missings):
        add_to_books(chk_books,main_degree,block_size-truncate-missing)
        add_to_books(chk_books,main_degree-1,missing)
    return chk_books,var_books

def find_composition(part_sums:int,num_part:int)->Iterable[Tuple[int,...]]:
    for markers in combinations(range(1,part_sums+num_part),num_part-1):
        yield tuple(b - a - 1 for a, b in zip((0,) + markers, markers + (part_sums+num_part,)))

def isRecipeSatisfyRect(recipe:Dict[Tuple[int,...],int])->bool:
    numRows = recipe.keys().__iter__().__next__().__len__()
    cntRect:Dict[Tuple[int,...], int] = dict()
    for colType, cnt in recipe.items():
        if 0 == cnt:
            continue
        zeroPos = [idx for idx, val in enumerate(colType) if 0 == val]
        for subset in allSubsets(zeroPos, min_size=1):
            cntRect[tuple(subset)] = cntRect.get(tuple(subset), 0) + cnt
    for key, val in cntRect.items():
        if len(key)+val > numRows:
            return False
    return True