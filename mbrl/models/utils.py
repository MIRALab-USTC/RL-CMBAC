import numpy as np

def square_wass2_gaussian(m1, std1, m2, std2):
    m_l2 = (m1-m2) ** 2
    std_l2 = (std1-std2) ** 2
    square_w2 = m_l2.sum(-1) + std_l2.sum(-1)
    return square_w2

def kl_gaussian(m1, var1, m2, var2):
    a = np.log(np.prod(var2, axis=-1, keepdims=True)/(np.prod(var1, axis=-1, keepdims=True)+1e-6))
    b = np.sum(var1/(var2+1e-6), axis=-1, keepdims=True)
    c = np.sum((m2-m1)**2 / (var2+1e-6), axis=-1, keepdims=True)
    n = m1.shape[1]
    return 0.5 * (a-n+b+c) 

