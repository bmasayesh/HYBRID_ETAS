# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 2D ETAS parameters with uniform horizontal 
background. This code will use LLrecipes to call most of the functions. 
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
import LLrecipes
from scipy.optimize import minimize

############################# Functions & Classes #############################
def fR(r, d, q):
    cnorm = (q / np.pi) * np.power(d, 2*q)
    fr = cnorm * np.power(np.square(d) + np.square(r), -(1+q))
    return fr

def calculate_ETASrate(time, t, r, mu, Na, c, p, d, q):
    R = mu + np.sum(Na * np.power(c + time - t, -p) * fR(r, d, q))
    return R

def LL1value(t, m, ti, Ri, Mc, mu, K, alpha, c, p, d0, gamma, q, Nind, Nij):
    Na = K * np.power(10.0, alpha*(m-Mc))
    d  = d0 * np.power(10.0, gamma * m)
    LL1 = 0
    for i in range(len(ti)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        R = calculate_ETASrate(ti[i], t[nij], Ri[i, nij], mu, Na[nij], c, p, d[nij], q)
        LL1 += np.log(R)
    return LL1

def LL2value(t, m, ti, Mc, mu, K, alpha, c, p, d0, gamma, q, T1, T2, A, TmaxTrig):
    rho = K * np.power(10.0, alpha*(m-Mc))
    I = mu * A * (T2-T1) + LLrecipes.integrate(t, rho, c, p, T1, T2, TmaxTrig)
    return I

def nLLETAS(arguments, LL_GR, t, m, ti, Ri, Mc, T1, T2, Nind, Nij, A, TmaxTrig):
    mu = np.square(arguments[0])
    K = np.square(arguments[1])
    alpha = np.square(arguments[2])
    c = np.square(arguments[3])
    p = np.square(arguments[4])
    d0 = np.square(arguments[5])
    gamma = np.square(arguments[6])
    q = np.square(arguments[7])

    LL1 = LL1value(t, m, ti, Ri, Mc, mu, K, alpha, c, p, d0, gamma, q, Nind, Nij)
    LL2 = LL2value(t, m, ti, Mc, mu, K, alpha, c, p, d0, gamma, q, T1, T2, A, TmaxTrig)
    LL = LL_GR + LL1 -LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.4f  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f --> Nback=%.1f Ntot=%.1f nLL=%f\r' % (mu, K, alpha, c, p, d0, gamma, q, LL2, len(ti), nLL)))
    sys.stdout.flush()
    return nLL

def LLoptimize(LL_GR, x0, t, m, ti, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig):
    res = minimize(nLLETAS, np.sqrt(x0), args=(LL_GR, t, m, ti, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mu = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    return mu, K, alpha, c, p, d0, gamma, q, -res.fun

def setstartparameter(A):
    mu = 1.0/A
    K     = 0.05   
    alpha = 1.0                
    c     = 0.01    # [days]
    p     = 1.3
    d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 1.0     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q])

def determine_ETASparameter(tall, latall, lonall, zall, mall, Mcut, T1, T2, Z1, Z2, A, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: t, lat, lon, m     sequence of earthquake times, location and magnitudes
           Mcut               cutoff magnitude
           T1, T2             [days] start and end time of the LL-fit
           stdmin             [km] minimum standard deviation (std) for kernel smoothing of background activity
           Nnearest           value used for kernel smoothing if std(Nnearest) > stdmin
           TmaxTrig           [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    ind = ((mall>= Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2))
    t = tall[ind]
    lat = latall[ind]
    lon = lonall[ind]
    m = mall[ind]
    ind = ((t>=T1) & (zall>=Z1) & (zall<=Z2))
    ti = t[ind]
    lati = lat[ind]
    loni = lon[ind]
    mi = m[ind]
    print('\n\t total events (M>=%.1f): N=%d  (fitted: N=%d)\n' % (Mcut, len(t), len(ti)))
    N, b, bstd, LL_GR = LLrecipes.calculate_N_and_bvalue(mi, Mcut) ## Mc=2 & dMbin=0.1
    print("\n\t GR Maximum Likelihood fit: b=%.2f +- %.2f  LL_GR = %s\n" % (b, bstd, LL_GR))
    
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    Ri = np.zeros((len(lati), len(lat)))
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist2D(lati[i], loni[i], lat[nij], lon[nij])

    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    x0 = setstartparameter(A)
    mu, K, alpha, c, p, d0, gamma, q, LL = LLoptimize(LL_GR, x0, t, m, ti, Ri, Mcut, T1, T2, Nind, Nij, A, TmaxTrig)
    print("RESULT:  mu=%f  K=%f  alpha=%f  c=%f   p=%f  d0=%f  gamma=%f   q=%f   LL=%f" % (mu*A, K, alpha, c, p, d0, gamma, q, LL))
    return mu*A, K, alpha, c, p, d0, gamma, q, b, LL


