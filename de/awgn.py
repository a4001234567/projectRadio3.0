import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Mapping, Union
from de.phi import func_phi, func_phi_inv
from de.recipe import generate_books_from_recipe

func_power = np.power

def apply_rho(msg_distribution,func_rho:dict):
    rho_result = 0
    for exp,coef in func_rho.items():
        rho_result += coef*func_phi_inv(1-func_power(1-func_phi(msg_distribution),exp-1))
    return rho_result

def apply_lambda(msg_distribution,func_lambda:dict):
    lambda_result = 0
    for exp,coef in func_lambda.items():
        lambda_result += coef*((exp-1)*msg_distribution)
    return lambda_result

def density_evolution(sigmaS,func_rho,func_lambda):
    '''
    func_rho is the degree distribution of check nodes
    func_lambda is the degree distribution of varaible nodes
    '''
    initial_msg = 2/sigmaS
    var_distribution = initial_msg
    cdf_val = 1e5
    for l in range(5):
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        chk_distribution = apply_rho(var_distribution,func_rho)
        chk_message = apply_lambda(chk_distribution,func_lambda)
        var_distribution = initial_msg + chk_message
        #x = torch.sqrt(2.*var_distribution)
        #cdf_val = torch.distributions.Normal(0,1).cdf(x)
        cdf_val:float = stats.norm(loc=var_distribution,scale=np.sqrt(2.*var_distribution)).cdf(0)
        if cdf_val < 1e-4:# or (cdf_val > 0.1 and l >= 1):
            break
    return 1-cdf_val

def find_thres(func_rho,func_lambda):
    '''
    dc,dv
    '''
    sum_lambda = sum(v for k,v in func_lambda.items())
    for k,v in func_lambda.items():
        func_lambda[k] = v/sum_lambda
    sum_rho = sum(v for k,v in func_rho.items())
    for k,v in func_rho.items():
        func_rho[k] = v/sum_rho
    rate = 1-(sum(k*v for k,v in func_lambda.items())/sum(k*v for k,v in func_rho.items()))
    #print(rate)
    threshold = 0
    for i in range(2,-10,-1):
        if density_evolution(threshold+pow(2,i),func_rho,func_lambda) >= 1 - 1e-4:
            threshold += pow(2,i)
    return threshold

# Derive the below two functions from the cooresponding equations:
# Eb/N0 = 1/(2 R sigma^2)
# Reference: Ryan, William, and Shu Lin. "Channel codes: classical and modern." Cambridge University press, 2009.
# Chapter 1:1.5.1.3:Page 15
def convert_snr_to_ebno(snr,R):
    return -10*np.log10(2*R*snr)

def convert_ebno_to_snr(ebno,R):
    return 2*R*np.power(10,-ebno/10)

def generate_func_from_book(book:Mapping[int,Union[int,float]]):
    sum_coef = sum(((deg-1)*coef for deg,coef in book.items()))
    return {deg:(deg-1)*coef/sum_coef for deg,coef in book.items()}

def findThresForRecipe(recipe:Dict[Tuple[int,...],int],block_size:int,truncate:int)->float:
    chk_book, var_book = generate_books_from_recipe(recipe,block_size,truncate)
    func_lambda = generate_func_from_book(var_book)
    func_rho = generate_func_from_book(chk_book)
    thres = find_thres(func_rho,func_lambda)
    H = recipe.keys().__iter__().__next__().__len__()
    W = sum(recipe.values())
    rate = 1-H/W
    thres = convert_snr_to_ebno(thres,rate)
    return thres

def find_thres_from_books(var_book:Dict[int,int], chk_book:Dict[int,int], rate:float)->float:
    func_lambda = generate_func_from_book(var_book)
    func_rho = generate_func_from_book(chk_book)
    thres = find_thres(func_rho,func_lambda)
    thres = convert_snr_to_ebno(thres,rate)
    return thres