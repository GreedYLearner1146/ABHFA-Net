import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

def Bhattacharyya_coeff(x,y): 
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x1 = x.unsqueeze(1).expand(n, m, d)
    y1 = y.unsqueeze(0).expand(n, m, d)


    # Inputs for variational computation. Use mean and std of the given inputs.
    x_m = torch.mean(x1)
    y_m = torch.mean(y1)
    x_std = torch.std(x1)
    y_std  = torch.std(y1)

    # The reparameterization trick. The second half of the parenthesis is the normal distribution.
    P1 = x_m + x_std*(1/(torch.sqrt(torch.abs(2*np.pi*x_std*x_std))))*torch.exp(-((x1-x_m)*(x1-x_m))/(2*x_std*x_std))
    Q1 = y_m + y_std*(1/(torch.sqrt(torch.abs(2*np.pi*y_std*y_std))))*torch.exp(-(y1-y_m)*(y1-y_m)/(2*y_std*y_std))

    # The Bhattacharyya coefficient for two gaussian distributions.
    BC = torch.sqrt(torch.abs(P1*Q1)).sum(2)

    return BC
