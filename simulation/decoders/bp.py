import numpy as np
from typing import List, Tuple

EPS = 1e-10
THRES = 1-EPS
tanh = np.tanh;arctanh = np.arctanh;clip = np.clip
copyto=np.copyto;addat = np.add.at;mulat = np.multiply.at
sign = np.sign

def make_Flooding_BP_Decoder(max_iter:int,
        indx: np.ndarray,
        indy: np.ndarray,
        H: int,
        W: int):
    prods = np.ones(H)
    assert max_iter > 1, "max_iter must be greater than 1"
    def Flooding_BP_decoder(llr: np.ndarray) -> Tuple[np.ndarray, int]:
        llr = llr.reshape(-1)
        sums = llr.copy()
        vals = tanh(.5*sums[indy])
        for t in range(max_iter):
            valSigns = sign(vals)
            valSigns[valSigns == 0] = 1.
            vals = valSigns * np.maximum(np.abs(vals), EPS)
            mulat(prods,indx,vals)
            vals = 2.*arctanh(clip(prods[indx]/vals,-THRES,THRES))
            copyto(sums,llr)
            addat(sums,indy,vals)
            x_hat:np.ndarray = (sums < 0.)
            if not np.any(x_hat):
                prods.fill(1.)
                break
            vals = tanh(.5*(sums[indy]-vals))
            prods.fill(1.)
        return x_hat.reshape(-1,1), t+1
    return Flooding_BP_decoder
