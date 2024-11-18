########################## Importing Required Modules #########################
import numpy as np
from scipy.special import erf
from scipy.special import hyp2f1
eps = 1e-6
day2sec = 24 * 60 * 60.0

############################# Functions & Classes #############################
def dist2D(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3
    R = R0 * np.arccos(
        np.clip(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)), -1, 1))
    return R

def dist3D(lat1, lon1, z1, lat2, lon2, z2):
    """
    Distance (in [km]) between points given as [lat,lon, depth]
    """
    R0 = 6367.3        
    Rxy = R0 * np.arccos(
        np.clip(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)), -1, 1))
    R = np.sqrt(np.square(Rxy) + np.square(z1-z2))
    return R

def calculate_N_and_bvalue(mm, Mc):
    '''
    Calculates the mean magnitute, the b-value based 
    on the maximum likelihood estimation, the b-value and its standard deviation.
    Input parameters:
    mall            event magnitudes
    Mc              cutoff-magnitude
    Output parameters:
    Nvalue          N(m>=Mc)
    bvalue          b-value
    bstdAki         standard deviation of b-value according to Aki (1965)
    '''
    m = mm[(mm>=Mc)]
    if len(m)>=1:
        # calculate the b-value (maximum likelihood):
        bvalue = (1.0/(np.mean(m)-(Mc))) * np.log10(np.exp(1.0))
        # calculate the standard deviation (Aki 1965):
        bstdAki = bvalue/np.sqrt(1.0*len(m))
        N_magLargerMc = len(m)
        # calculate the log-likelihood
        LL = N_magLargerMc * np.log(np.log(10) * bvalue) - np.log(10) * bvalue * np.sum(m - Mc)
    else:
        N_magLargerMc = float('NaN')
        bvalue = float('NaN')
        bstdAki = float('NaN')
        LL = float('NaN')
    return N_magLargerMc, bvalue, bstdAki, LL

def calculate_znorm(z, stdz, Z1, Z2):
    '''
    This function normalize z of target events with std.
    We need normalized depth for estimating background rate.
    erf: Gauss error function 
    '''
    zn1 = np.abs(Z1 - z) / (np.sqrt(2.0) * stdz)
    zn2 = np.abs(Z2 - z) / (np.sqrt(2.0) * stdz)
    integ = 0.5 * (erf(zn1) + erf(zn2))
    return 1.0/integ

def gaussian1D(x, sig):
    res = np.exp(-0.5 * np.power(x/sig, 2.0)) / (np.sqrt(2 * np.pi) * sig)
    return res

def gaussian2D(x, sig):
    res = np.exp(-0.5 * np.power(x/sig, 2.)) / (2 * np.pi * np.power(sig, 2.))
    return res

def calculate_mu(cznorm, dzii, stdzi, pbacki, T1, T2, A):
    '''
    Function for calculate background rate. It returns background rate of target
    events and sum of bacground probability as mutotal.
    '''
    mu = np.zeros(len(stdzi))
    for i in range(len(stdzi)):
        mu[i] = np.sum(pbacki * cznorm * gaussian1D(dzii[i, :], stdzi)) / (A * (T2 - T1)) 
    Nbacktot = np.sum(pbacki)
    return mu, Nbacktot

def calculate_mu_2D(Rii, stdi, pbacki, T1, T2):
    mu = np.zeros(len(stdi))
    for i in range(len(stdi)):
        mu[i] = np.sum(pbacki * gaussian2D(Rii[i, :], stdi)) / (T2 - T1) 
    mutot = np.sum(pbacki)
    return mu, mutot    

def calculate_frmain(Rii, stdi, cznorm, dzii, stdzi, pmaini):
    frmain = np.zeros(len(stdi))
    for i in range(len(stdi)):
        frmain[i] = np.sum(pmaini * gaussian2D(Rii[i, :], stdi) * cznorm * gaussian1D(dzii[i, :], stdzi)) 
    frmain /= np.sum(pmaini)
    return frmain

def determine_normalization_factor_2D(d, q):
    cnorm = (q/np.pi) * np.power(d, 2*q)
    return cnorm

def determine_normalization_factor_3D(d, q):
    R = 10000.0
    d2 = np.power(d, 2.0)
    integ = (4.0/3.0) * np.pi * np.power(R, 3.0) * np.power(d2, -q-1) * hyp2f1(1.5, q+1, 2.5, -np.square(R/d))
    return 1.0/integ

def fR(r, d, q, cnorm):
    fr = cnorm * np.power(np.square(d) + np.square(r), -(1+q))
    return fr

def integ_func(x, d, q):
    # Integral [d^2 + x^2]^{-q} dz (see Mathematica)
    res = x * np.power(d, -2*q) * hyp2f1(0.5, q, 1.5, -np.square(x/d))
    return res

def integration_zrange(d, q, z, Z1, Z2, cnorm):
    fac = np.pi / q
    I =  cnorm * fac * (np.sign(z-Z1)*integ_func(np.abs(z-Z1), d, q) + np.sign(Z2-z)*integ_func(np.abs(Z2-z), d, q))
    return I

def integrate(t, rho, c, p, T1, T2, TmaxTrig):
    ta = T1 - t
    ta[(ta<0)] = 0.0
    tb = T2 - t
    tb[(tb>TmaxTrig)] = TmaxTrig
    t1 = c + ta
    t2 = c + tb
    if p==1:
        dum1 = np.log(t1)
        dum2 = np.log(t2)
    else:
        dum1 = np.power(t1, 1.0-p) / (1.0-p)
        dum2 = np.power(t2, 1.0-p) / (1.0-p)
    ft = np.sum(rho * (dum2 - dum1))
    return ft

def integratemain(tmain, Namain, c, p, T1, T2, TmaxTrig):
    ft = 0
    if T2 > tmain and T1 < tmain+TmaxTrig:
        t1 = 0.0
        if T1 > tmain:
            t1 = T1 - tmain
        t2 = T2 - tmain
        if t2 > TmaxTrig:
            t2 = TmaxTrig
        if p==1:
            dum1 = np.log(c+t1)
            dum2 = np.log(c+t2)
        else:
            dum1 = np.power(c+t1, 1.0-p) / (1.0-p)
            dum2 = np.power(c+t2, 1.0-p) / (1.0-p)
        ft = Namain * (dum2 - dum1)
    return ft

def LL2valueETAS_2mainshocks(tmain1, Namain1, tmain2, Namain2, t, mutot, Na, c, p, T1, T2, TmaxTrig):
    I = mutot * (T2-T1) + integrate(t, Na, c, p, T1, T2, TmaxTrig)
    I += integratemain(tmain1, Namain1, c, p, T1, T2, TmaxTrig)
    I += integratemain(tmain2, Namain2, c, p, T1, T2, TmaxTrig)
    return I

def calculate_detectionfactor_2mainshocks(tmain1, Namain1, tmain2, Namain2, time, t, mutot, NaZinteg, c, p, bDT):
    R0 = mutot + np.sum(NaZinteg * np.power(c + time - t, -p))
    if time > tmain1:
        R0 += Namain1 * np.power(c + time - tmain1, -p)
    if time > tmain2:
        R0 += Namain2 * np.power(c + time - tmain2, -p)
    R = (1.0 - np.exp(-R0*bDT)) / bDT
    fac = R / R0
    return fac

def calculate_detectionfactor_2mainshocks1(tmain1, Namain1, time, t, mutot, NaZinteg, c, p, bDT):
    R0 = mutot + np.sum(NaZinteg * np.power(c + time - t, -p))
    if time > tmain1:
        R0 += Namain1 * np.power(c + time - tmain1, -p)
    # if time > tmain2:
    #     R0 += Namain2 * np.power(c + time - tmain2, -p)
    R = (1.0 - np.exp(-R0*bDT)) / bDT
    fac = R / R0
    return fac

def LL2valueETASI_2mainshocks(tsteps, tmain1, Namain1, tmain2, Namain2, t, mutot, NaZinteg, c, p, TmaxTrig, bDT):
    R0 = mutot * np.ones(len(tsteps))
    for i, time in enumerate(tsteps):
        nij = ((t<time) & (t>time-TmaxTrig))
        R0[i] += np.sum(NaZinteg[nij] * np.power(c + time - t[nij], -p))
    R0[(tsteps>tmain1)] += Namain1 * np.power(c + tsteps[(tsteps>tmain1)] - tmain1, -p)
    R0[(tsteps>tmain2)] += Namain2 * np.power(c + tsteps[(tsteps>tmain2)] - tmain2, -p)
    R = (1.0 - np.exp(-R0*bDT)) / bDT
    I = np.trapz(R, tsteps)
    return I

def logpdf_HAINZL_GR(Mc, b, N0, m):
    x = N0 * np.power(10.0, -b * (m - Mc))
    #pdfm = np.log(10) * b * x * np.exp(-x) / (1.0 - np.exp(-N0))
    logx = np.log(N0) - np.log(10) * b * (m - Mc)
    if N0 > 100:
        logdenom = 0
    else:
        logdenom = np.log(1.0 - np.exp(-N0))
    logpdfm = np.log(np.log(10) * b) + logx - x - logdenom 
    return logpdfm

                                
def LLGR_2mainshocks(tmain1, Namain1, tmain2, Namain2, t, ti, mi, Mc, mutot, NaZinteg, c, p, Nind, Nij, b, bDT):
    LL = 0.0
    for i in range(len(ti)):
        NI = Nind[i]
        nij = Nij[i,:NI]
        R0 = mutot + np.sum(NaZinteg[nij] * np.power(c + ti[i] - t[nij], -p))
        if ti[i] > tmain1:
            R0 += Namain1 * np.power(c + ti[i] - tmain1, -p)
        if ti[i] > tmain2:
            R0 += Namain2 * np.power(c + ti[i] - tmain2, -p)
        LL += logpdf_HAINZL_GR(Mc, b, R0*bDT, mi[i])
    return LL

def select_triggerevents(t, ti, TmaxTrig):
    '''
    This function returns a matrix with number of target events in row and all 
    events in column. For each ti in row we check which events could trigger it.
    It also returns an array to show the number of events could trigger the each
    target event.
    '''
    indall = np.arange(len(t))
    Nind = np.zeros(len(ti), dtype=int)
    Nij = np.zeros((len(ti), len(t)), dtype=int)
    for i, time in enumerate(ti):
        ind = ((t<time) & (t>=ti[i]-TmaxTrig))
        NI = len(indall[ind])
        Nind[i] = NI
        Nij[i, :NI] = indall[ind]
    return Nind, Nij

def select_targetevents_2D(tmain, tall, latall, lonall, mall, Mcut, T1, T2, TmaxTrig):
    ind = ((mall>=Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2) & (np.abs(tall-tmain)>eps))
    t = tall[ind]
    lat = latall[ind]
    lon = lonall[ind]
    m = mall[ind]
    # ind = (t>=T1)
    ind = ((mall>=Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2) & (tall>=T1))
    ti = tall[ind]
    lati = latall[ind]
    loni = lonall[ind]
    mi = mall[ind]
    return t, lat, lon, m, ti, lati, loni, mi

def select_targetevents_3D(tmain, tall, latall, lonall, zall, mall, Mcut, T1, T2, Z1, Z2, TmaxTrig):
    ind = ((mall>=Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2) & (np.abs(tall-tmain)>eps))
    t = tall[ind]
    lat = latall[ind]
    lon = lonall[ind]
    z = zall[ind]
    m = mall[ind]
    ind = ((mall>=Mcut) & (tall>=T1-TmaxTrig) & (tall<=T2) & (tall>=T1) & (zall>=Z1) & (zall<=Z2))
    ti = tall[ind]
    lati = latall[ind]
    loni = lonall[ind]
    zi = zall[ind]
    mi = mall[ind]
    return t, lat, lon, z, m, ti, lati, loni, zi, mi

def define_tsteps_2mainshocks(tmain1, tmain2, T1, T2, ti):
    teps = 1.0 /day2sec
    fac = (0.1, 0.3, 0.5, 0.8)
    tsteps = np.append(T1, np.unique(ti))
    tsteps = np.append(tsteps, T2)
    ts = []
    for i, fi in enumerate(fac): 
        ts = np.append(ts, tsteps[:-1] + fi * (tsteps[1:] - tsteps[:-1]))
    tsteps = np.append(tsteps, ts)
    if tmain1 > T1 and tmain1 < T2:
        tsteps = np.append(tsteps, tmain1)
        tsteps = np.append(tsteps, tmain1+teps)
    if tmain2 > T1 and tmain2 < T2:
        tsteps = np.append(tsteps, tmain2)
        tsteps = np.append(tsteps, tmain2+teps)
    tsteps = np.sort(tsteps)
    return tsteps

def define_tsteps(T1, T2, ti):
    # fac = np.logspace(-3, np.log10(0.99), 100)
    # teps = 1.0 /day2sec
    fac = (0.1, 0.3, 0.5, 0.8)
    tsteps = np.append(T1, np.unique(ti))
    tsteps = np.append(tsteps, T2)
    ts = []
    for i, fi in enumerate(fac): 
        ts = np.append(ts, tsteps[:-1] + fi * (tsteps[1:] - tsteps[:-1]))
    tsteps = np.append(tsteps, ts)
    tsteps = np.sort(tsteps)
    return tsteps

def define_tsteps_mainshock(tmain, T1, T2, ti):
    teps = 1.0 /day2sec
    fac = (0.1, 0.3, 0.5, 0.8)
    tsteps = np.append(T1, np.unique(ti))
    tsteps = np.append(tsteps, T2)
    ts = []
    for i, fi in enumerate(fac): 
        ts = np.append(ts, tsteps[:-1] + fi * (tsteps[1:] - tsteps[:-1]))
    tsteps = np.append(tsteps, ts)
    if tmain > T1 and tmain < T2:
        tsteps = np.append(tsteps, tmain)
        tsteps = np.append(tsteps, tmain+teps)
    tsteps = np.sort(tsteps)
    return tsteps



