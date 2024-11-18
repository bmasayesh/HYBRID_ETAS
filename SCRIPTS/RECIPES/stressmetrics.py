
import numpy as np
from numpy import linalg as LA


def cmbfix(strike, dip, rake, sxx, syy, szz, sxy, syz, szx, f, skempton):
    '''
    calculate Coulomb Stress for fixed receiver (based on cmbfix.f) 
        
    Input:	
    stress tensor, friction coefficient
    receiver orientation parameter (strike, dip and rake)
    return:												    
    Coulomb stress (cmb) 
    '''

    DEG2RAD = np.pi/180.0
    
    s11 = sxx
    s12 = sxy
    s13 = szx
    s21 = sxy
    s22 = syy
    s23 = syz
    s31 = szx
    s32 = syz
    s33 = szz
    strike_radian = strike * DEG2RAD
    dip_radian    = dip * DEG2RAD
    rake_radian   = rake * DEG2RAD

    ns1 =  np.sin(dip_radian) * np.cos(strike_radian + 0.5 * np.pi)
    ns2 =  np.sin(dip_radian) * np.sin(strike_radian + 0.5 * np.pi)
    ns3 = -np.cos(dip_radian)

    rst1 = np.cos(strike_radian)
    rst2 = np.sin(strike_radian)
    rst3 = 0.0

    rdi1 = np.cos(dip_radian) * np.cos(strike_radian + 0.5 * np.pi)
    rdi2 = np.cos(dip_radian) * np.sin(strike_radian + 0.5 * np.pi)
    rdi3 = np.sin(dip_radian)
        
    ts1 = rst1 * np.cos(rake_radian) - rdi1 * np.sin(rake_radian)
    ts2 = rst2 * np.cos(rake_radian) - rdi2 * np.sin(rake_radian)
    ts3 = rst3 * np.cos(rake_radian) - rdi3 * np.sin(rake_radian)

    sigg = 0.0
    tau  = 0.0
    sigg += ns1 * s11 * ns1
    tau  += ts1 * s11 * ns1    
    sigg += ns2 * s21 * ns1
    tau  += ts2 * s21 * ns1
    sigg += ns3 * s31 * ns1
    tau  += ts3 * s31 * ns1
        
    sigg += ns1 * s12 * ns2
    tau  += ts1 * s12 * ns2    
    sigg += ns2 * s22 * ns2
    tau  += ts2 * s22 * ns2
    sigg += ns3 * s32 * ns2
    tau  += ts3 * s32 * ns2
        
    sigg += ns1 * s13 * ns3
    tau  += ts1 * s13 * ns3    
    sigg += ns2 * s23 * ns3
    tau  += ts2 * s23 * ns3
    sigg += ns3 * s33 * ns3
    tau  += ts3 * s33 * ns3

    p = -skempton * (sxx + syy + szz) / 3.0
    cmb = tau + f * (sigg + p) 
    return cmb

def stressmetrics(strike, dip, rake, Sxx, Syy, Szz, Sxy, Syz, Szx, f, skempton):

    # Coulomb stress for fixed receiver:
    CFS = cmbfix(strike, dip, rake, Sxx, Syy, Szz, Sxy, Syz, Szx, f, skempton)

    # Coulomb stress for variable mechanism:
    ddeg = 30.0   # standard deviation strike, dip, rake
    df = 0.1      # standard deviation friction
    Nsample = 1000
    strikei = np.random.normal(strike, ddeg, Nsample)
    dipi = np.random.normal(dip, ddeg, Nsample)
    rakei = np.random.normal(rake, ddeg, Nsample)
    fi = np.random.normal(f, df, Nsample)
    CFSi = cmbfix(strikei, dipi, rakei, Sxx, Syy, Szz, Sxy, Syz, Szx, fi, skempton)
    VM = np.sum(CFSi[(CFSi>0.0)]) / (1.0 * Nsample)

    s11, s12, s13, s21, s22, s23, s31, s32, s33 = Sxx, Sxy, Szx, Sxy, Syy, Syz, Szx, Syz, Szz
    w, v = LA.eig(np.array([[s11, s12, s13], [s21, s22, s23], [s31, s32, s33]]))

    # Maximum Shear
    # x1 = w[0]
    # x3 = w[2]
    
    x1 = max(w)
    x3 = min(w)
    MS = np.absolute(x1-x3)/2.0

    # van Mises Stress:
    IN1 = w[0] + w[1] + w[2]
    IN2 = w[0]*w[1] + w[1]*w[2] + w[0]*w[2]
    VMS = np.sqrt(IN1*IN1 - 3*IN2)

    return CFS, VM, MS, VMS
