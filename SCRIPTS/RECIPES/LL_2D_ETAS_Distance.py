# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This is a Function code for estimating of 2D ETAS parameters by considering 
anisotropic kernel using distance to the fault plane of mainshock with uniform horizontal 
background. This code will use LLrecipes to call most of the functions. 
'''
########################## Importing Required Modules #########################
import sys
import LLrecipes
import numpy as np
from scipy.optimize import minimize
eps = 1e-6
############################# Functions & Classes #############################
def fR(r, d, q):
    cnorm = (q / np.pi) * np.power(d, 2*q)
    fr = cnorm * np.power(np.square(d) + np.square(r), -(1+q))
    return fr

def calculate_ETASrate(tmain, Namain, time, rf, df, cf, t, r, mu, Na, c, p, d, q):
    R = mu + np.sum(Na * np.power(c + time - t, -p) * fR(r, d, q))
    if time > tmain:
        R += Namain * np.power(c + time - tmain, -p) * cf * fR(rf, df, q)
    return R

def LL1value(tmain, Namain, t, m, ti, Ri, rfi, mu, Na, c, p, d, q, dfault, cfault, Nind, Nij):
    LL1 = 0
    NN = len(ti)
    R = np.zeros(NN)
    for i in range(NN):
        NI = Nind[i]
        nij = Nij[i,:NI]
        R[i] = calculate_ETASrate(tmain, Namain, ti[i], rfi[i], dfault, cfault, t[nij], Ri[i, nij], mu, Na[nij], c, p, d[nij], q)
    LL1 = np.sum(np.log(R))
    return LL1

def LL2valueETAS_mainshock(tmain, Namain, t, m, ti, Mc, mu, K, alpha, c, p, d0, gamma, q, T1, T2, A, TmaxTrig):
    rho = K * np.power(10.0, alpha*(m-Mc))
    I = mu * A * (T2-T1)  + LLrecipes.integrate(t, rho, c, p, T1, T2, TmaxTrig)
    if tmain < T2:
        I += LLrecipes.integratemain(tmain, Namain, c, p, T1, T2, TmaxTrig)
    return I

def nLLETAS(arguments, rfault, LL_GR, A, tmain, Mmain, t, m, ti, Ri, rfi, Mc, T1, T2, Nind, Nij, TmaxTrig):
    mu = np.square(arguments[0])
    K = np.square(arguments[1])
    alpha = np.square(arguments[2])
    c = np.square(arguments[3])
    p = np.square(arguments[4])
    d0 = np.square(arguments[5])
    gamma = np.square(arguments[6])
    q = np.square(arguments[7])
    dfault = np.square(arguments[8])
    d  = d0 * np.power(10.0, gamma * m)
    Na = K * np.power(10.0, alpha*(m-Mc))
    Namain = K * np.power(10.0, alpha*(Mmain-Mc))
    # Normalization factor for the mainshock spatial anisotropic kernel:
    Agrid = A / len(rfault)  # Test that this is really dr^2 = 1
    cfault = 1.0 / np.sum(Agrid * fR(rfault, dfault, q))

    LL1 = LL1value(tmain, Namain, t, m, ti, Ri, rfi, mu, Na, c, p, d, q, dfault, cfault, Nind, Nij)
    LL2 = LL2valueETAS_mainshock(tmain, Namain, t, m, ti, Mc, mu, K, alpha, c, p, d0, gamma, q, T1, T2, A, TmaxTrig)

    LL = LL_GR + LL1 -LL2
    nLL = -LL
    if np.isnan(nLL):
        nLL = 1e10
    sys.stdout.write('\r'+str('search: mu=%.3f  K=%.4f  alpha=%.2f  c=%.4f  p=%.2f d0=%.4f  gamma=%.2f  q=%.2f dfault=%.2f --> Ntot=%.1f nLL=%f\r' % (mu, K, alpha, c, p, d0, gamma, q, dfault, len(ti), nLL)))
    sys.stdout.flush()
    return nLL

def LLoptimize(rfault, LL_GR, x0, tmain, Mmain, t, m, ti, Ri, rfi, Mcut, T1, T2, A, Nind, Nij, TmaxTrig):
    res = minimize(nLLETAS, np.sqrt(x0), args=(rfault, LL_GR, A, tmain, Mmain, t, m, ti, Ri, rfi, Mcut, T1, T2, Nind, Nij, TmaxTrig), method='BFGS', options={'gtol': 1e-01})
    mu = np.square(res.x[0])
    K = np.square(res.x[1])
    alpha = np.square(res.x[2])
    c = np.square(res.x[3])
    p = np.square(res.x[4])
    d0 = np.square(res.x[5])
    gamma = np.square(res.x[6])
    q = np.square(res.x[7])
    dfault = np.square(res.x[8])
    return mu, K, alpha, c, p, d0, gamma, q, dfault, -res.fun

def setstartparameter(A):
    mu = 1.0/A
    K     = 0.05   
    alpha = 1.0                
    c     = 0.01    # [days]
    p     = 1.3
    d0    = 0.013   # ... with d = d0 * 10^(gamma*M) 
    gamma = 0.5     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    q     = 0.7     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    dfault = 1.0    # for distance-to-fault decay of mainshock-triggered events
    
    # mu = 0.24
    # K     = 0.059   
    # alpha = 0.52                
    # c     = 0.039    # [days]
    # p     = 0.99
    # d0    = 0.015   # ... with d = d0 * 10^(gamma*M) 
    # gamma = 0.37     # L(M) = d0*10^(gamma*M); d0=0.013  gamma=0.5  (Wells & Coppersmith 1994; RLD normal faulting)
    # q     = 0.87     # spatial probability density: f(r) = (q-1)/(pi*D^2) * ( 1 + r^2/D^2 )^(-(1+q))
    # dfault = 1.0    # for distance-to-fault decay of mainshock-triggered events
    
    return np.asarray([mu, K, alpha, c, p, d0, gamma, q, dfault])

def determine_ETASparameter(latf, lonf, rfault, tmain, Mmain, tall, latall, lonall, mall, Mcut, T1, T2, A, TmaxTrig):
    '''
    Search of the minimum -LL value for the ETAS-model

    Input: latf, lonf, rfault  gridpoints and distance to fault
           tmain, Mmain        mainshock time and magnitude
           t, lat, lon, z, m   sequence of earthquake times, location and magnitudes
           Mcut                cutoff magnitude
           T1, T2              [days] start and end time of the LL-fit
           A                   [km^2] surface area of fit region
           Z1, Z2              depth interval for LL-fit
           TmaxTrig            [days] maximum length of aftershock triggering
    Return estimated parameters and LL-value
    '''
    ind = ((mall>Mcut) & (tall>=T1))
    mm = mall[ind]
    ## Log-Likelihood of GR by maximum likelihood estimation of b-value
    N, b, bstd, LL_GR = LLrecipes.calculate_N_and_bvalue(mm, Mcut)
    print("\n\t GR Maximum Likelihood fit: b=%.2f +- %.2f  LL_GR = %s\n" % (b, bstd, LL_GR))
    
    t, lat, lon, m, ti, lati, loni, mi = LLrecipes.select_targetevents_2D(tmain, tall, latall, lonall, mall, Mcut, T1, T2, TmaxTrig)
    print('\n\t total events (M>=%.2f): N=%d  (fitted: N=%d) Mmax=%.2f (%.2f in fit-period)\n' % (Mcut, len(t), len(ti), np.max(m), np.max(mi)))    
    
    print(" --> RESULT for time interval [%.1f  %.1f]:\n" % (T1, T2))
    Nind, Nij = LLrecipes.select_triggerevents(t, ti, TmaxTrig)
    
    rfi = np.zeros(len(lati))
    Ri = np.zeros((len(lati), len(lat)))
    for i in range(len(lati)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        Ri[i, nij] = LLrecipes.dist2D(lati[i], loni[i], lat[nij], lon[nij])
        imin = np.argmin(LLrecipes.dist2D(lati[i], loni[i], latf, lonf))
        rfi[i] = rfault[imin]
    
    x0 = setstartparameter(A)
    mu, K, alpha, c, p, d0, gamma, q, dfault, LL = LLoptimize(rfault, LL_GR, x0, tmain, Mmain, t, m, ti, Ri, rfi, Mcut, T1, T2, A, Nind, Nij, TmaxTrig)  
    print("RESULTS:  mu=%f  c=%f   p=%f  K=%f  alpha=%f  d0=%f  gamma=%f   q=%f   LL=%f" % (mu, c, p, K, alpha, d0, gamma, q, LL))

    return mu*A, K, alpha, c, p, d0, gamma, q, dfault, b, LL

