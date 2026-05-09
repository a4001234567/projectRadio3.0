from math import sqrt
from codegen.qc.ruler import CircularRuler
from codegen.qc.grid import generate_rulers, write_board
from codegen.analysis.cr import check_CR, CR_checker
from fileio.matrix_io import Writer
import numpy as np

log_records = {}
def mod_log(base:int, num:int, mod:int)->int:
    return log_records[(base,num,mod)]

def is_prime(num_p:int)->bool:
    if num_p < 2:
        return False
    for i in range(2, int(sqrt(num_p))+1):
        if num_p % i == 0:
            return False
    return True

def check_primitive(num_c:int,num_p:int)->bool:
    global log_records
    already_seen = set()
    for i in range(0,num_p-1):
        c_power = pow(num_c,i,num_p)
        if c_power in already_seen:
            return False
        log_records[(num_c,c_power,num_p)] = i
        already_seen.add(c_power)
    return True

def check_xiao_condition(num_b:int):
    '''
    Check if the positive integer b satisifies the condition of Xiao:
        1. 20b+1 is a prime number
        2. there exists a primitive element alpha in the field GF(20b+1)
        3. alpha^{4b}+1 = alpha^e, where e < 20b+1 is an odd integer
    '''
    num_p = 20*num_b + 1
    if not is_prime(num_p):
        return False
    for alpha in range(2,num_p):
        if check_primitive(alpha,num_p):
            e = mod_log(alpha,(pow(alpha,4*num_b,num_p)+1)%num_p,num_p)
            if e % 2 == 1:
                return True, alpha, e
    return False

def generate_ingredients_P2(num_b:int, alpha:int, e:int, dimD:int, dimK:int):
    num_p = 20*num_b + 1
    Bs = []
    for i in range(dimD):
        Bs.append([[num_p-pow(alpha,2*j+4*i*num_b,num_p)] for j in range(dimK)])
    return Bs

def make_board(num_b:int, alpha:int, e:int, dimD:int, dimK:int,truncate:int=0):
    ingredients = generate_ingredients_P2(num_b, alpha, e, dimD, dimK)
    block_size = 20*num_b+1
    ingredients[0][1] = []
    ingredients[0][3] = []
    ingredients[1][5] = []
    ingredients[1][7] = []
    ingredients[2][0] = []
    ingredients[2][2] = []
    ingredients[3][4] = []
    ingredients[3][6] = []
    print(ingredients)
    rulers = generate_rulers(ingredients, block_size)
    write_board(rulers, block_size,\
                filename=f'XPG{dimD}*{dimK}*({block_size}-{truncate}).zip',\
                comments=f'b={num_b},alpha={alpha},e={e}',\
                truncate=truncate,compress=True, mode='sparse')
    #return board

if __name__ == '__main__':
    assert check_xiao_condition(2) == (True,6,3)
    assert check_xiao_condition(41) == (True,2,713)
    assert check_xiao_condition(30) == (True,7,79)
    assert check_xiao_condition(31) == False
    #ingredients = generate_ingredients_P2(41,2,713,3,3)
    #rulers = generate_rulers(ingredients, 20*41+1)
    #board = make_board(rulers, 20*41+1, 0)
    #assert check_CR(board) == True
    '''b = 35
    result = check_xiao_condition(b)
    assert result
    _, alpha, e = result
    print(result)
    ingredients = generate_ingredients_P2(b,alpha,e,5,b)
    block_size = 20*b+1
    rulers = generate_rulers(ingredients, block_size)
    board = draw_board(rulers, block_size, 0)
    #print(board.shape)
    #exit()
    assert check_CR(board) == True
    print('All tests passed!')'''
    #check_xiao_condition(32) = True, 3, 631
    make_board(num_b = 32, alpha=3, e=631, dimD=4, dimK=8, truncate=0)
    exit()
    truncate = 0
    b = 30
    for k in range(30,5*b+1,20):
        print(f'k = {k}')
        result = check_xiao_condition(b)
        if not result: continue
        block_size = 20*b+1
        _, alpha, e = result
        print(f'Found b={b}, alpha={alpha}, e={e}')
        make_board(b,alpha,e,5,k,truncate=truncate)
        print('Check passed! Written to File')
        #writer(f'XPG5*35*({block_size}-{truncate}).zip',board,mode='sparse',comments=f'b={b},alpha={alpha},e={e}',compress=True)
        #with open('XPG5*35*{block_size}-{truncate}')
        #    f.write(str(board))

from math import sqrt
from codegen.qc.grid import generate_rulers, write_board

log_records = {}
def mod_log(base:int, num:int, mod:int)->int:
    return log_records[(base,num,mod)]

def is_prime(num_p:int)->bool:
    if num_p < 2:
        return False
    for i in range(2, int(sqrt(num_p))+1):
        if num_p % i == 0:
            return False
    return True

def check_primitive(num_c:int,num_p:int)->bool:
    global log_records
    already_seen = set()
    for i in range(0,num_p-1):
        c_power = pow(num_c,i,num_p)
        if c_power in already_seen:
            return False
        log_records[(num_c,c_power,num_p)] = i
        already_seen.add(c_power)
    return True

def check_xiao_condition(num_b:int):
    '''
    Check if the positive integer b satisifies the condition of Xiao:
        1. 12b+1 is a prime number
        2. there exists a primitive element alpha in the field GF(12b+1)
        3. alpha^{4b}+1 = alpha^e, where e < 12b+1 is an odd integer
    '''
    num_p = 12*num_b + 1
    if not is_prime(num_p):
        return False
    for alpha in range(2,num_p):
        if check_primitive(alpha,num_p):
            e = mod_log(alpha,(pow(alpha,4*num_b,num_p)-1+num_p)%num_p,num_p)
            if e % 2 == 1:
                return True, alpha, e
    return False

def generate_ingredients_P2(num_b:int, alpha:int, e:int, dimD:int, dimK:int):
    num_p = 12*num_b + 1
    Bs = []
    for i in range(dimD):
        if 0 == i:
            Bs.append([[0] for _ in range(dimK)])
        else:
            Bs.append([[pow(alpha, 2*num_b*(i-1)+2*j, num_p)] for j in range(dimK)])
    return Bs

def make_board(num_b:int, alpha:int, e:int, dimD:int, dimK:int,truncate:int=0):
    ingredients = generate_ingredients_P2(num_b, alpha, e, dimD, dimK)
    block_size = 12*num_b+1
    rulers = generate_rulers(ingredients, block_size)
    write_board(rulers, block_size,\
                filename=f'XQPG{dimD}*{dimK}*({block_size}-{truncate}).zip',\
                comments=f'b={num_b},alpha={alpha},e={e}',\
                truncate=truncate,compress=True, mode='sparse')

if __name__ == '__main__':
    assert check_xiao_condition(6) == (True, 5, 33)
    assert check_xiao_condition(61) == (True, 6, 145)
    b = 61
    res, alpha, e = check_xiao_condition(b)
    make_board(b, alpha, e, 4, 61, truncate=0)
    exit()
    assert check_xiao_condition(2) == (True,6,3)
    assert check_xiao_condition(41) == (True,2,713)
    assert check_xiao_condition(30) == (True,7,79)
    assert check_xiao_condition(31) == False
    #ingredients = generate_ingredients_P2(41,2,713,3,3)
    #rulers = generate_rulers(ingredients, 20*41+1)
    #board = make_board(rulers, 20*41+1, 0)
    #assert check_CR(board) == True
    '''b = 35
    result = check_xiao_condition(b)
    assert result
    _, alpha, e = result
    print(result)
    ingredients = generate_ingredients_P2(b,alpha,e,5,b)
    block_size = 20*b+1
    rulers = generate_rulers(ingredients, block_size)
    board = draw_board(rulers, block_size, 0)
    #print(board.shape)
    #exit()
    assert check_CR(board) == True
    print('All tests passed!')'''
    #check_xiao_condition(32) = True, 3, 631
    make_board(num_b = 32, alpha=3, e=631, dimD=4, dimK=8, truncate=0)
    exit()
    truncate = 0
    b = 30
    for k in range(30,5*b+1,20):
        print(f'k = {k}')
        result = check_xiao_condition(b)
        if not result: continue
        block_size = 20*b+1
        _, alpha, e = result
        print(f'Found b={b}, alpha={alpha}, e={e}')
        make_board(b,alpha,e,5,k,truncate=truncate)
        print('Check passed! Written to File')
        #writer(f'XPG5*35*({block_size}-{truncate}).zip',board,mode='sparse',comments=f'b={b},alpha={alpha},e={e}',compress=True)
        #with open('XPG5*35*{block_size}-{truncate}')
        #    f.write(str(board))