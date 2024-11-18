import sys
import numpy as np

PI2 = 2 * np.pi
EPS = 1e-6

def variablesC0(C0common):
    # COMMON /C0/ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D
    ALP1 = C0common[0]
    ALP2 = C0common[1]
    ALP3 = C0common[2]
    ALP4 = C0common[3]
    ALP5 = C0common[4]
    SD = C0common[5]
    CD = C0common[6]
    SDSD = C0common[7]
    CDCD = C0common[8]
    SDCD = C0common[9]
    S2D = C0common[10]
    C2D = C0common[11]
    return ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D

def DCCON0(ALPHA, DIP):
    '''
    *******************************************************************
    *****   CALCULATE MEDIUM CONSTANTS AND FAULT-DIP CONSTANTS    *****
    *******************************************************************
    
    ***** INPUT
    *****   ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
    *****   DIP   : DIP-ANGLE (DEGREE)
    '''
    #C0common: ALP1, ALP2, ALP3, ALP4, ALP5, SD, CD, SDSD, CDCD, SDCD, S2D, C2D
    #           0     1     2     3     4    5   6    7     8     9    10   11
    deg2rad = np.pi / 180.0
    C0common = np.zeros(12)
    C0common[0] = (1.0 - ALPHA)/2.0   # ALP1
    C0common[1] = ALPHA/2.0           # ALP2
    C0common[2] = (1.0 - ALPHA)/ALPHA # ALP3
    C0common[3] = 1.0 - ALPHA         # ALP4
    C0common[4] = ALPHA               # ALP5
    SD = np.sin(DIP * deg2rad)
    CD = np.cos(DIP * deg2rad)
    if np.abs(CD < EPS):
        CD = 0.0
        if SD > 0.0:
            SD = 1.0   
        if SD < 0.0:
            SD = -1.0
    SDSD = SD * SD  
    SDCD = SD * CD
    CDCD = CD * CD
    C0common[5] = SD                  # SD
    C0common[6] = CD                  # CD 
    C0common[7] = SDSD                # SDSD
    C0common[8] = CDCD                # CDCD
    C0common[9] = SDCD                # SDCD
    C0common[10] = 2.0 * SDCD         # S2D
    C0common[11] = CDCD - SDSD        # C2D
    return C0common

def DCCON2(XI, ET, Q, SD, CD, KXI, KET):
    '''
    **********************************************************************
    *****   CALCULATE STATION GEOMETRY CONSTANTS FOR FINITE SOURCE   *****
    **********************************************************************
    
    ***** INPUT
    *****   XI,ET,Q : STATION COORDINATES IN FAULT SYSTEM
    *****   SD,CD   : SIN, COS OF DIP-ANGLE
    *****   KXI,KET : KXI=1, KET=1 MEANS R+XI<EPS, R+ET<EPS, RESPECTIVELY
    '''
    ### CAUTION ### IF XI,ET,Q ARE SUFFICIENTLY SMALL, THEY ARE SET TO ZER0
    XI[(np.abs(XI)<EPS)] = 0.0
    ET[(np.abs(ET)<EPS)] = 0.0
    Q[(np.abs(Q)<EPS)] = 0.0
    XI2 = np.square(XI)
    ET2 = np.square(ET)
    Q2 = np.square(Q)
    R2 = XI2 + ET2 + Q2
    R = np.sqrt(R2)
    R3 = R * R2
    R5 = R3 * R2
    Y = ET * CD + Q * SD
    D = ET * SD - Q * CD
    TT = np.arctan(XI*ET/(Q*R))
    TT[(Q==0.0)] = 0.0
    X11 = np.zeros(len(XI))
    X32 = np.zeros(len(XI))
    Y11 = np.zeros(len(XI))
    Y32 = np.zeros(len(XI))
    ALX = -np.log(R - XI)
    ALX[(KXI != 1)] = np.log(R[(KXI != 1)] + XI[(KXI != 1)])
    X11[(KXI != 1)] = 1.0 / (R[(KXI != 1)] * (R[(KXI != 1)] + XI[(KXI != 1)]))
    X32[(KXI != 1)] = (R[(KXI != 1)] + R[(KXI != 1)] + XI[(KXI != 1)]) * X11[(KXI != 1)] * X11[(KXI != 1)] / R[(KXI != 1)]
    ALE = -np.log(R - ET)
    ALE[(KET != 1)] = np.log(R[(KET != 1)] + ET[(KET != 1)])
    Y11[(KET != 1)] = 1.0 / (R[(KET != 1)] * (R[(KET != 1)] + ET[(KET != 1)]))
    Y32[(KET != 1)] = (R[(KET != 1)] + R[(KET != 1)] + ET[(KET != 1)]) * Y11[(KET != 1)] * Y11[(KET != 1)] / R[(KET != 1)]
    EY = SD/R - Y * Q/R3
    EZ = CD/R + D * Q/R3
    FY = D/R3 + XI2 * Y32 * SD
    FZ = Y/R3 + XI2 * Y32 * CD
    GY = 2.0 * X11 * SD - Y * Q * X32
    GZ = 2.0 * X11 * CD + D * Q * X32
    HY = D * Q * X32 + XI * Q * Y32 * SD
    HZ = Y * Q * X32 + XI * Q * Y32 * CD
    return XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ

def UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI, ET, Q, DISL1, DISL2, DISL3):
    '''
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-A)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************
    
    ***** INPUT
    *****   XI,ET,Q     : STATION COORDINATES IN FAULT SYSTEM
    *****   DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    ***** OUTPUT
    *****   U(12)       : DISPLACEMENT AND THEIR DERIVATIVES
    '''
    ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D = variablesC0(C0common)
    XY = XI * Y11
    QX = Q * X11
    QY = Q * Y11
    #
    U0=0.0; U1=0.0; U2=0.0; U3=0.0; U4=0.0;  U5=0.0; U6=0.0; U7=0.0; U8=0.0; U9=0.0; U10=0.0; U11=0.0
    #======================================
    #=====  STRIKE-SLIP CONTRIBUTION  =====
    #======================================
    if DISL1 != 0.0:
        DU0 =    TT/2.0 +ALP2*XI*QY
        DU1 =           ALP2*Q/R
        DU2 = ALP1*ALE -ALP2*Q*QY
        DU3 =-ALP1*QY  -ALP2*XI2*Q*Y32
        DU4 =          -ALP2*XI*Q/R3
        DU5 = ALP1*XY  +ALP2*XI*Q2*Y32
        DU6 = ALP1*XY*SD        +ALP2*XI*FY+D/2.0*X11
        DU7 =                    ALP2*EY
        DU8 = ALP1*(CD/R+QY*SD) -ALP2*Q*FY
        DU9 = ALP1*XY*CD        +ALP2*XI*FZ+Y/2.0*X11
        DU10=                    ALP2*EZ
        DU11=-ALP1*(SD/R-QY*CD) -ALP2*Q*FZ
        U0 += DISL1/PI2 * DU0
        U1 += DISL1/PI2 * DU1
        U2 += DISL1/PI2 * DU2
        U3 += DISL1/PI2 * DU3
        U4 += DISL1/PI2 * DU4
        U5 += DISL1/PI2 * DU5
        U6 += DISL1/PI2 * DU6
        U7 += DISL1/PI2 * DU7
        U8 += DISL1/PI2 * DU8
        U9 += DISL1/PI2 * DU9
        U10 += DISL1/PI2 * DU10
        U11 += DISL1/PI2 * DU11
    #======================================
    #=====    DIP-SLIP CONTRIBUTION   =====
    #======================================
    if DISL2 != 0.0:
        DU0 =           ALP2*Q/R
        DU1 =   TT/2.0 +ALP2*ET*QX
        DU2 = ALP1*ALX -ALP2*Q*QX
        DU3 =        -ALP2*XI*Q/R3
        DU4 = -QY/2.0 -ALP2*ET*Q/R3
        DU5 = ALP1/R +ALP2*Q2/R3
        DU6 =                      ALP2*EY
        DU7 = ALP1*D*X11+XY/2.0*SD +ALP2*ET*GY
        DU8 = ALP1*Y*X11          -ALP2*Q*GY
        DU9 =                      ALP2*EZ
        DU10= ALP1*Y*X11+XY/2.0*CD +ALP2*ET*GZ
        DU11=-ALP1*D*X11          -ALP2*Q*GZ
        U0 += DISL2/PI2 * DU0
        U1 += DISL2/PI2 * DU1
        U2 += DISL2/PI2 * DU2
        U3 += DISL2/PI2 * DU3
        U4 += DISL2/PI2 * DU4
        U5 += DISL2/PI2 * DU5
        U6 += DISL2/PI2 * DU6
        U7 += DISL2/PI2 * DU7
        U8 += DISL2/PI2 * DU8
        U9 += DISL2/PI2 * DU9
        U10 += DISL2/PI2 * DU10
        U11 += DISL2/PI2 * DU11
    #========================================
    #=====  TENSILE-FAULT CONTRIBUTION  =====
    #========================================
    if DISL3 != 0.0:
        DU0 =-ALP1*ALE -ALP2*Q*QY
        DU1 =-ALP1*ALX -ALP2*Q*QX
        DU2 =   TT/2.0 -ALP2*(ET*QX+XI*QY)
        DU3 =-ALP1*XY  +ALP2*XI*Q2*Y32
        DU4 =-ALP1/R   +ALP2*Q2/R3
        DU5 =-ALP1*QY  -ALP2*Q*Q2*Y32
        DU6 =-ALP1*(CD/R+QY*SD)  -ALP2*Q*FY
        DU7 =-ALP1*Y*X11         -ALP2*Q*GY
        DU8 = ALP1*(D*X11+XY*SD) +ALP2*Q*HY
        DU9 = ALP1*(SD/R-QY*CD)  -ALP2*Q*FZ
        DU10= ALP1*D*X11         -ALP2*Q*GZ
        DU11= ALP1*(Y*X11+XY*CD) +ALP2*Q*HZ
        U0 += DISL3/PI2 * DU0
        U1 += DISL3/PI2 * DU1
        U2 += DISL3/PI2 * DU2
        U3 += DISL3/PI2 * DU3
        U4 += DISL3/PI2 * DU4
        U5 += DISL3/PI2 * DU5
        U6 += DISL3/PI2 * DU6
        U7 += DISL3/PI2 * DU7
        U8 += DISL3/PI2 * DU8
        U9 += DISL3/PI2 * DU9
        U10 += DISL3/PI2 * DU10
        U11 += DISL3/PI2 * DU11
    return U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11

def UB(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI, ET, Q, DISL1, DISL2, DISL3):
    '''
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-B)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************
    
    ***** INPUT
    *****   XI,ET,Q : STATION COORDINATES IN FAULT SYSTEM
    *****   DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    ***** OUTPUT
    *****   U(12) : DISPLACEMENT AND THEIR DERIVATIVES
    '''
    ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D = variablesC0(C0common)
    RD = R + D
    D11 = 1.0 / (R * RD)
    AJ2 = XI * Y/RD * D11
    AJ5 = -(D + Y * Y/RD) * D11
    if CD != 0.0:
        AI4 = np.zeros(len(XI))
        X = np.sqrt(XI2 + Q2)
        AI4 = 1.0/CDCD * ( XI/RD * SDCD + 2.0* np.arctan((ET*(X+Q*CD)+X*(R+X)*SD)/(XI*(R+X)*CD)) )
        AI4[(XI==0)] = 0.0
        AI3 = (Y * CD/RD - ALE + SD * np.log(RD)) / CDCD
        AK1 = XI * (D11 - Y11 * SD) / CD
        AK3 = (Q * Y11 - Y * D11) / CD
        AJ3 = (AK1 - AJ2 * SD) / CD
        AJ6 = (AK3 - AJ5 * SD) / CD
    else:
        RD2 = RD * RD
        AI3 = (ET/RD + Y * Q/RD2 - ALE) / 2.0
        AI4 = XI * Y/RD2/2.0
        AK1 = XI * Q/RD * D11
        AK3 = SD/RD * (XI2 * D11 - 1.0)
        AJ3 = -XI/RD2 * (Q2 * D11 - 0.5)
        AJ6 = -Y/RD2 * (XI2 * D11 - 0.5)
    XY = XI * Y11
    AI1 = -XI/RD * CD - AI4 * SD
    AI2 = np.log(RD) + AI3 * SD
    AK2 = 1.0/R + AK3 * SD
    AK4 = XY * CD - AK1 * SD
    AJ1 = AJ5 * CD - AJ6 * SD
    AJ4 = -XY - AJ2 * CD + AJ3 * SD
    QX = Q * X11
    QY = Q * Y11
    #
    U0=0.0; U1=0.0; U2=0.0; U3=0.0; U4=0.0;  U5=0.0; U6=0.0; U7=0.0; U8=0.0; U9=0.0; U10=0.0; U11=0.0
    #======================================
    #=====  STRIKE-SLIP CONTRIBUTION  =====
    #======================================
    if DISL1 != 0.0:
        DU0 = -XI*QY-TT -ALP3*AI1*SD
        DU1 = -Q/R      +ALP3*Y/RD*SD
        DU2 =  Q*QY     -ALP3*AI2*SD
        DU3 = XI2*Q*Y32 -ALP3*AJ1*SD
        DU4 = XI*Q/R3   -ALP3*AJ2*SD
        DU5 =-XI*Q2*Y32 -ALP3*AJ3*SD
        DU6 =-XI*FY-D*X11 +ALP3*(XY+AJ4)*SD
        DU7 =-EY          +ALP3*(1.0/R+AJ5)*SD
        DU8 = Q*FY        -ALP3*(QY-AJ6)*SD
        DU9 =-XI*FZ-Y*X11 +ALP3*AK1*SD
        DU10=-EZ          +ALP3*Y*D11*SD
        DU11= Q*FZ        +ALP3*AK2*SD
        U0 += DISL1/PI2 * DU0
        U1 += DISL1/PI2 * DU1
        U2 += DISL1/PI2 * DU2
        U3 += DISL1/PI2 * DU3
        U4 += DISL1/PI2 * DU4
        U5 += DISL1/PI2 * DU5
        U6 += DISL1/PI2 * DU6
        U7 += DISL1/PI2 * DU7
        U8 += DISL1/PI2 * DU8
        U9 += DISL1/PI2 * DU9
        U10 += DISL1/PI2 * DU10
        U11 += DISL1/PI2 * DU11
    #======================================
    #=====    DIP-SLIP CONTRIBUTION   =====
    #======================================
    if DISL2 != 0.0:
        DU0 = -Q/R      +ALP3*AI3*SDCD
        DU1 = -ET*QX-TT -ALP3*XI/RD*SDCD
        DU2 = Q*QX     +ALP3*AI4*SDCD
        DU3 = XI*Q/R3     +ALP3*AJ4*SDCD
        DU4 = ET*Q/R3+QY  +ALP3*AJ5*SDCD
        DU5 =-Q2/R3       +ALP3*AJ6*SDCD
        DU6 =-EY          +ALP3*AJ1*SDCD
        DU7 =-ET*GY-XY*SD +ALP3*AJ2*SDCD
        DU8 = Q*GY        +ALP3*AJ3*SDCD
        DU9 =-EZ          -ALP3*AK3*SDCD
        DU10=-ET*GZ-XY*CD -ALP3*XI*D11*SDCD
        DU11= Q*GZ        -ALP3*AK4*SDCD
        U0 += DISL2/PI2 * DU0
        U1 += DISL2/PI2 * DU1
        U2 += DISL2/PI2 * DU2
        U3 += DISL2/PI2 * DU3
        U4 += DISL2/PI2 * DU4
        U5 += DISL2/PI2 * DU5
        U6 += DISL2/PI2 * DU6
        U7 += DISL2/PI2 * DU7
        U8 += DISL2/PI2 * DU8
        U9 += DISL2/PI2 * DU9
        U10 += DISL2/PI2 * DU10
        U11 += DISL2/PI2 * DU11
    #========================================
    #=====  TENSILE-FAULT CONTRIBUTION  =====
    #========================================
    if DISL3 != 0.0:
        DU0  = Q*QY           -ALP3*AI3*SDSD
        DU1  = Q*QX           +ALP3*XI/RD*SDSD
        DU2 = ET*QX+XI*QY-TT -ALP3*AI4*SDSD
        DU3 =-XI*Q2*Y32 -ALP3*AJ4*SDSD
        DU4 =-Q2/R3     -ALP3*AJ5*SDSD
        DU5 = Q*Q2*Y32  -ALP3*AJ6*SDSD
        DU6 = Q*FY -ALP3*AJ1*SDSD
        DU7 = Q*GY -ALP3*AJ2*SDSD
        DU8 =-Q*HY -ALP3*AJ3*SDSD
        DU9 = Q*FZ +ALP3*AK3*SDSD
        DU10= Q*GZ +ALP3*XI*D11*SDSD
        DU11=-Q*HZ +ALP3*AK4*SDSD
        U0 += DISL3/PI2 * DU0
        U1 += DISL3/PI2 * DU1
        U2 += DISL3/PI2 * DU2
        U3 += DISL3/PI2 * DU3
        U4 += DISL3/PI2 * DU4
        U5 += DISL3/PI2 * DU5
        U6 += DISL3/PI2 * DU6
        U7 += DISL3/PI2 * DU7
        U8 += DISL3/PI2 * DU8
        U9 += DISL3/PI2 * DU9
        U10 += DISL3/PI2 * DU10
        U11 += DISL3/PI2 * DU11
    return U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11

def UC(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI, ET, Q, Z, DISL1, DISL2, DISL3):
    '''
    ********************************************************************
    *****    DISPLACEMENT AND STRAIN AT DEPTH (PART-C)             *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    ********************************************************************
    
    ***** INPUT
    *****   XI,ET,Q,Z   : STATION COORDINATES IN FAULT SYSTEM
    *****   DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    ***** OUTPUT
    *****   U(12) : DISPLACEMENT AND THEIR DERIVATIVES
    '''
    ALP1,ALP2,ALP3,ALP4,ALP5,SD,CD,SDSD,CDCD,SDCD,S2D,C2D = variablesC0(C0common)  
    C = D + Z
    X53 = (8.0 * R2 + 9.0 * R * XI + 3.0 * XI2) * X11 * X11 * X11/R2
    Y53 = (8.0 * R2 + 9.0 * R * ET + 3.0 * ET2) * Y11 * Y11 * Y11/R2
    H = Q * CD - Z
    Z32 = SD/R3 - H * Y32
    Z53 = 3.0 * SD/R5 - H * Y53
    Y0 = Y11 - XI2 * Y32
    Z0 = Z32 - XI2 * Z53
    PPY = CD/R3 + Q * Y32 * SD
    PPZ = SD/R3 - Q * Y32 * CD
    QQ = Z * Y32 + Z32 + Z0
    QQY = 3.0 * C * D/R5 - QQ * SD
    QQZ = 3.0 * C * Y/R5 - QQ * CD + Q * Y32
    XY = XI * Y11
    QX = Q * X11
    QY = Q * Y11
    QR = 3.0 * Q/R5
    CQX = C * Q * X53
    CDR = (C + D) / R3
    YY0 = Y/R3 - Y0 * CD
    #
    U0=0.0; U1=0.0; U2=0.0; U3=0.0; U4=0.0;  U5=0.0; U6=0.0; U7=0.0; U8=0.0; U9=0.0; U10=0.0; U11=0.0
    #======================================
    #=====  STRIKE-SLIP CONTRIBUTION  =====
    #======================================
    if DISL1 != 0.0:
        DU0 = ALP4*XY*CD           -ALP5*XI*Q*Z32
        DU1 = ALP4*(CD/R+2.0*QY*SD) -ALP5*C*Q/R3
        DU2 = ALP4*QY*CD           -ALP5*(C*ET/R3-Z*Y11+XI2*Z32)
        DU3 = ALP4*Y0*CD                  -ALP5*Q*Z0
        DU4 =-ALP4*XI*(CD/R3+2.0*Q*Y32*SD) +ALP5*C*XI*QR
        DU5 =-ALP4*XI*Q*Y32*CD            +ALP5*XI*(3.0*C*ET/R5-QQ)
        DU6 =-ALP4*XI*PPY*CD    -ALP5*XI*QQY
        DU7 = ALP4*2.0*(D/R3-Y0*SD)*SD-Y/R3*CD - ALP5*(CDR*SD-ET/R3-C*Y*QR)
        DU8 =-ALP4*Q/R3+YY0*SD  +ALP5*(CDR*CD+C*D*QR-(Y0*CD+Q*Z0)*SD)
        DU9 = ALP4*XI*PPZ*CD    -ALP5*XI*QQZ
        DU10= ALP4*2.0*(Y/R3-Y0*CD)*SD+D/R3*CD -ALP5*(CDR*CD+C*D*QR)
        DU11= YY0*CD    -ALP5*(CDR*SD-C*Y*QR-Y0*SDSD+Q*Z0*CD)
        U0 += DISL1/PI2 * DU0
        U1 += DISL1/PI2 * DU1
        U2 += DISL1/PI2 * DU2
        U3 += DISL1/PI2 * DU3
        U4 += DISL1/PI2 * DU4
        U5 += DISL1/PI2 * DU5
        U6 += DISL1/PI2 * DU6
        U7 += DISL1/PI2 * DU7
        U8 += DISL1/PI2 * DU8
        U9 += DISL1/PI2 * DU9
        U10 += DISL1/PI2 * DU10
        U11 += DISL1/PI2 * DU11
    #======================================
    #=====    DIP-SLIP CONTRIBUTION   =====
    #======================================
    if DISL2 != 0.0:
        DU0 = ALP4*CD/R -QY*SD -ALP5*C*Q/R3
        DU1 = ALP4*Y*X11       -ALP5*C*ET*Q*X32
        DU2 =     -D*X11-XY*SD -ALP5*C*(X11-Q2*X32)
        DU3 =-ALP4*XI/R3*CD +ALP5*C*XI*QR +XI*Q*Y32*SD
        DU4 =-ALP4*Y/R3     +ALP5*C*ET*QR
        DU5 =    D/R3-Y0*SD +ALP5*C/R3*(1.0-3.0*Q2/R2)
        DU6 =-ALP4*ET/R3+Y0*SDSD -ALP5*(CDR*SD-C*Y*QR)
        DU7 = ALP4*(X11-Y*Y*X32) -ALP5*C*((D+2.0*Q*CD)*X32-Y*ET*Q*X53)
        DU8 =  XI*PPY*SD+Y*D*X32 +ALP5*C*((Y+2.0*Q*SD)*X32-Y*Q2*X53)
        DU9 =      -Q/R3+Y0*SDCD -ALP5*(CDR*CD+C*D*QR)
        DU10= ALP4*Y*D*X32       -ALP5*C*((Y-2.0*Q*SD)*X32+D*ET*Q*X53)
        DU11=-XI*PPZ*SD+X11-D*D*X32-ALP5*C*((D-2.0*Q*CD)*X32-D*Q2*X53)
        U0 += DISL2/PI2 * DU0
        U1 += DISL2/PI2 * DU1
        U2 += DISL2/PI2 * DU2
        U3 += DISL2/PI2 * DU3
        U4 += DISL2/PI2 * DU4
        U5 += DISL2/PI2 * DU5
        U6 += DISL2/PI2 * DU6
        U7 += DISL2/PI2 * DU7
        U8 += DISL2/PI2 * DU8
        U9 += DISL2/PI2 * DU9
        U10 += DISL2/PI2 * DU10
        U11 += DISL2/PI2 * DU11
    #========================================
    #=====  TENSILE-FAULT CONTRIBUTION  =====
    #========================================
    if DISL3 != 0.0:
        DU0 =-ALP4*(SD/R+QY*CD)   -ALP5*(Z*Y11-Q2*Z32)
        DU1 = ALP4*2.0*XY*SD+D*X11 -ALP5*C*(X11-Q2*X32)
        DU2 = ALP4*(Y*X11+XY*CD)  +ALP5*Q*(C*ET*X32+XI*Z32)
        DU3 = ALP4*XI/R3*SD+XI*Q*Y32*CD+ALP5*XI*(3.0*C*ET/R5-2.0*Z32-Z0)
        DU4 = ALP4*2.0*Y0*SD-D/R3 +ALP5*C/R3*(1.0-3.0*Q2/R2)
        DU5 =-ALP4*YY0           -ALP5*(C*ET*QR-Q*Z0)
        DU6 = ALP4*(Q/R3+Y0*SDCD)   +ALP5*(Z/R3*CD+C*D*QR-Q*Z0*SD)
        DU7 =-ALP4*2.0*XI*PPY*SD-Y*D*X32 + ALP5*C*((Y+2.0*Q*SD)*X32-Y*Q2*X53)
        DU8 =-ALP4*(XI*PPY*CD-X11+Y*Y*X32) + ALP5*(C*((D+2.0*Q*CD)*X32-Y*ET*Q*X53)+XI*QQY)
        DU9 =-ET/R3+Y0*CDCD -ALP5*(Z/R3*SD-C*Y*QR-Y0*SDSD+Q*Z0*CD)
        DU10= ALP4*2.0*XI*PPZ*SD-X11+D*D*X32 - ALP5*C*((D-2.0*Q*CD)*X32-D*Q2*X53)
        DU11= ALP4*(XI*PPZ*CD+Y*D*X32) + ALP5*(C*((Y-2.0*Q*SD)*X32+D*ET*Q*X53)+XI*QQZ)
        U0 += DISL3/PI2 * DU0
        U1 += DISL3/PI2 * DU1
        U2 += DISL3/PI2 * DU2
        U3 += DISL3/PI2 * DU3
        U4 += DISL3/PI2 * DU4
        U5 += DISL3/PI2 * DU5
        U6 += DISL3/PI2 * DU6
        U7 += DISL3/PI2 * DU7
        U8 += DISL3/PI2 * DU8
        U9 += DISL3/PI2 * DU9
        U10 += DISL3/PI2 * DU10
        U11 += DISL3/PI2 * DU11
    return U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11

def calculate_KXI_KET(XI0, XI1, ET0, ET1, Q):
    R12 = np.sqrt(np.square(XI0) + np.square(ET1) + np.square(Q))
    R21 = np.sqrt(np.square(XI1) + np.square(ET0) + np.square(Q))
    R22 = np.sqrt(np.square(XI1) + np.square(ET1) + np.square(Q))
    KXI0 = np.zeros(len(XI0))
    KXI1 = np.zeros(len(XI0))
    KET0 = np.zeros(len(XI0)) 
    KET1 = np.zeros(len(XI0))
    KXI0[((XI0<0.0) & (R21+XI1<EPS))] = 1
    KXI1[((XI0<0.0) & (R22+XI1<EPS))] = 1
    KET0[((ET0<0.0) & (R12+ET1<EPS))] = 1
    KET1[((ET0<0.0) & (R22+ET1<EPS))] = 1
    return KXI0, KXI1, KET0, KET1

def DC3D(ALPHA, X, Y, Z, DEPTH, DIP, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3):
    '''
    ********************************************************************
    *****                                                          *****
    *****    DISPLACEMENT AND STRAIN AT DEPTH                      *****
    *****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****
    *****              CODED BY  Y.OKADA ... SEP.1991              *****
    *****              REVISED ... NOV.1991, APR.1992, MAY.1993,   *****
    *****                          JUL.1993, MAY.2002              *****
    ********************************************************************
    
    ***** INPUT
    *****   ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)
    *****   X,Y,Z : ARRAY OF COORDINATES OF OBSERVING POINTS
    *****   DEPTH : DEPTH OF REFERENCE POINT
    *****   DIP   : DIP-ANGLE (DEGREE)
    *****   AL1,AL2   : FAULT LENGTH RANGE
    *****   AW1,AW2   : FAULT WIDTH RANGE
    *****   DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS
    ***** OUTPUT
    *****   UX, UY, UZ  : DISPLACEMENT ( UNIT=(UNIT OF DISL))
    *****   UXX,UYX,UZX : X-DERIVATIVE ( UNIT=(UNIT OF DISL))
    *****   UXY,UYY,UZY : Y-DERIVATIVE        (UNIT OF X,Y,Z,DEPTH,AL,AW) )
    *****   UXZ,UYZ,UZZ : Z-DERIVATIVE
    '''
    if len(Z[(Z>0)]) > 0:
        sys.exit('\n\t At least one observation point with Z>0! ... EXIT\n')
    C0common = DCCON0(ALPHA, DIP)
    SD = C0common[5]
    CD = C0common[6]
    XI0 = X - AL1
    XI1 = X - AL2
    XI0[(np.abs(XI0)<EPS)] = 0.0
    XI1[(np.abs(XI1)<EPS)] = 0.0
    #======================================
    #=====  REAL-SOURCE CONTRIBUTION  =====
    #======================================
    D = DEPTH + Z
    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    ET0 = P - AW1
    ET1 = P - AW2
    Q[(np.abs(Q)<EPS)] = 0.0
    ET0[(np.abs(ET0)<EPS)] = 0.0
    ET1[(np.abs(ET1)<EPS)] = 0.0
    #
    U0=0.0; U1=0.0; U2=0.0; U3=0.0; U4=0.0;  U5=0.0; U6=0.0; U7=0.0; U8=0.0; U9=0.0; U10=0.0; U11=0.0
    #--------------------------------
    #----- REJECT SINGULAR CASE -----
    #--------------------------------
    #----- ON FAULT EDGE
    if len(Q[((Q==0.0) & (XI0*XI1<=0.0) & (ET0*ET1==0.0))]) > 0:
        sys.exit('\n\t At least one observation point ON FAULT EDGE! ... EXIT\n')
    if len(Q[((Q==0.0) & (ET0*ET1<=0.0) & (XI0*XI1==0.0))]) > 0:
        sys.exit('\n\t At least one observation point ON FAULT EDGE! ... EXIT\n')
    #----- ON NEGATIVE EXTENSION OF FAULT EDGE
    KXI0, KXI1, KET0, KET1 = calculate_KXI_KET(XI0, XI1, ET0, ET1, Q)
    # K=0 & J=0:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI0, ET0, Q, SD, CD, KXI0, KET0)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET0, Q, DISL1, DISL2, DISL3)
    DU0 = -DUA0
    DU1 = -DUA1 * CD + DUA2 * SD
    DU2 = -DUA1 * SD - DUA2 * CD
    DU3 = -DUA3
    DU4 = -DUA4 * CD + DUA5 * SD
    DU5 = -DUA4 * SD - DUA5 * CD
    DU6 = -DUA6
    DU7 = -DUA7 * CD + DUA8 * SD
    DU8 = -DUA7 * SD - DUA8 * CD
    DU9 = -DUA9
    DU10 = -DUA10 * CD + DUA11 * SD
    DU11 = -DUA10 * SD - DUA11 * CD
    DU9 *= -1
    DU10 *= -1
    DU11 *= -1
    U0+=DU0; U1+=DU1; U2+=DU2; U3+=DU3; U4+=DU4 ; U5+=DU5 ; U6+=DU6; U7+=DU7; U8+=DU8; U9+=DU9; U10+=DU10; U11+=DU11
    # K=0 & J=1:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI1, ET0, Q, SD, CD, KXI0, KET1)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET0, Q, DISL1, DISL2, DISL3)
    DU0 = -DUA0
    DU1 = -DUA1 * CD + DUA2 * SD
    DU2 = -DUA1 * SD - DUA2 * CD
    DU3 = -DUA3
    DU4 = -DUA4 * CD + DUA5 * SD
    DU5 = -DUA4 * SD - DUA5 * CD
    DU6 = -DUA6
    DU7 = -DUA7 * CD + DUA8 * SD
    DU8 = -DUA7 * SD - DUA8 * CD
    DU9 = -DUA9
    DU10 = -DUA10 * CD + DUA11 * SD
    DU11 = -DUA10 * SD - DUA11 * CD
    DU9 *= -1
    DU10 *= -1
    DU11 *= -1
    U0-=DU0; U1-=DU1; U2-=DU2; U3-=DU3; U4-=DU4 ; U5-=DU5 ; U6-=DU6; U7-=DU7; U8-=DU8; U9-=DU9; U10-=DU10; U11-=DU11
    # K=1 & J=0:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI0, ET1, Q, SD, CD, KXI1, KET0)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET1, Q, DISL1, DISL2, DISL3)
    DU0 = -DUA0
    DU1 = -DUA1 * CD + DUA2 * SD
    DU2 = -DUA1 * SD - DUA2 * CD
    DU3 = -DUA3
    DU4 = -DUA4 * CD + DUA5 * SD
    DU5 = -DUA4 * SD - DUA5 * CD
    DU6 = -DUA6
    DU7 = -DUA7 * CD + DUA8 * SD
    DU8 = -DUA7 * SD - DUA8 * CD
    DU9 = -DUA9
    DU10 = -DUA10 * CD + DUA11 * SD
    DU11 = -DUA10 * SD - DUA11 * CD
    DU9 *= -1
    DU10 *= -1
    DU11 *= -1
    U0-=DU0; U1-=DU1; U2-=DU2; U3-=DU3; U4-=DU4 ; U5-=DU5 ; U6-=DU6; U7-=DU7; U8-=DU8; U9-=DU9; U10-=DU10; U11-=DU11
    # K=1 & J=1:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI1, ET1, Q, SD, CD, KXI1, KET1)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET1, Q, DISL1, DISL2, DISL3)
    DU0 = -DUA0
    DU1 = -DUA1 * CD + DUA2 * SD
    DU2 = -DUA1 * SD - DUA2 * CD
    DU3 = -DUA3
    DU4 = -DUA4 * CD + DUA5 * SD
    DU5 = -DUA4 * SD - DUA5 * CD
    DU6 = -DUA6
    DU7 = -DUA7 * CD + DUA8 * SD
    DU8 = -DUA7 * SD - DUA8 * CD
    DU9 = -DUA9
    DU10 = -DUA10 * CD + DUA11 * SD
    DU11 = -DUA10 * SD - DUA11 * CD
    DU9 *= -1
    DU10 *= -1
    DU11 *= -1
    U0+=DU0; U1+=DU1; U2+=DU2; U3+=DU3; U4+=DU4 ; U5+=DU5 ; U6+=DU6; U7+=DU7; U8+=DU8; U9+=DU9; U10+=DU10; U11+=DU11
    #=======================================
    #=====  IMAGE-SOURCE CONTRIBUTION  =====
    #=======================================
    D = DEPTH - Z
    P = Y * CD + D * SD
    Q = Y * SD - D * CD
    ET0 = P - AW1
    ET1 = P - AW2
    Q[(np.abs(Q)<EPS)] = 0.0
    ET0[(np.abs(ET0)<EPS)] = 0.0
    ET1[(np.abs(ET1)<EPS)] = 0.0
    #--------------------------------
    #----- REJECT SINGULAR CASE -----
    #--------------------------------
    #----- ON FAULT EDGE
    if len(Q[((Q==0.0) & (XI0*XI1<=0.0) & (ET0*ET1==0.0))]) > 0:
        sys.exit('\n\t At least one observation point ON FAULT EDGE! ... EXIT\n')
    if len(Q[((Q==0.0) & (ET0*ET1<=0.0) & (XI0*XI1==0.0))]) > 0:
        sys.exit('\n\t At least one observation point ON FAULT EDGE! ... EXIT\n')
    #----- ON NEGATIVE EXTENSION OF FAULT EDGE
    KXI0, KXI1, KET0, KET1 = calculate_KXI_KET(XI0, XI1, ET0, ET1, Q)
    # K=0 & J=0:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI0, ET0, Q, SD, CD, KXI0, KET0)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET0, Q, DISL1, DISL2, DISL3)
    DUB0, DUB1, DUB2, DUB3, DUB4, DUB5, DUB6, DUB7, DUB8, DUB9, DUB10, DUB11 = UB(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET0, Q, DISL1, DISL2, DISL3)
    DUC0, DUC1, DUC2, DUC3, DUC4, DUC5, DUC6, DUC7, DUC8, DUC9, DUC10, DUC11 = UC(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET0, Q, Z, DISL1, DISL2, DISL3)
    DU0 = DUA0 + DUB0 + Z * DUC0
    DU1 = (DUA1 + DUB1 + Z * DUC1) * CD - (DUA2 + DUB2 + Z * DUC2) * SD
    DU2 = (DUA1 + DUB1 - Z * DUC1) * SD + (DUA2 + DUB2 - Z * DUC2) * CD
    DU3 = DUA3 + DUB3 + Z * DUC3
    DU4 = (DUA4 + DUB4 + Z * DUC4) * CD - (DUA5 + DUB5 + Z * DUC5) * SD
    DU5 = (DUA4 + DUB4 - Z * DUC4) * SD + (DUA5 + DUB5 - Z * DUC5) * CD
    DU6 = DUA6 + DUB6 + Z * DUC6
    DU7 = (DUA7 + DUB7 + Z * DUC7) * CD - (DUA8 + DUB8 + Z * DUC8) * SD
    DU8 = (DUA7 + DUB7 - Z * DUC7) * SD + (DUA8 + DUB8 - Z * DUC8) * CD
    DU9 = DUA9 + DUB9 + Z * DUC9
    DU10 = (DUA10 + DUB10 + Z * DUC10) * CD - (DUA11 + DUB11 + Z * DUC11) * SD
    DU11 = (DUA10 + DUB10 - Z * DUC10) * SD + (DUA11 + DUB11 - Z * DUC11) * CD
    DU9 += DUC0
    DU10 += DUC1 * CD - DUC2 * SD
    DU11 += -DUC1 * SD - DUC2 * CD
    U0+=DU0; U1+=DU1; U2+=DU2; U3+=DU3; U4+=DU4 ; U5+=DU5 ; U6+=DU6; U7+=DU7; U8+=DU8; U9+=DU9; U10+=DU10; U11+=DU11
    # K=0 & J=1:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI1, ET0, Q, SD, CD, KXI0, KET1)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET0, Q, DISL1, DISL2, DISL3)
    DUB0, DUB1, DUB2, DUB3, DUB4, DUB5, DUB6, DUB7, DUB8, DUB9, DUB10, DUB11 = UB(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET0, Q, DISL1, DISL2, DISL3)
    DUC0, DUC1, DUC2, DUC3, DUC4, DUC5, DUC6, DUC7, DUC8, DUC9, DUC10, DUC11 = UC(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET0, Q, Z, DISL1, DISL2, DISL3)
    DU0 = DUA0 + DUB0 + Z * DUC0
    DU1 = (DUA1 + DUB1 + Z * DUC1) * CD - (DUA2 + DUB2 + Z * DUC2) * SD
    DU2 = (DUA1 + DUB1 - Z * DUC1) * SD + (DUA2 + DUB2 - Z * DUC2) * CD
    DU3 = DUA3 + DUB3 + Z * DUC3
    DU4 = (DUA4 + DUB4 + Z * DUC4) * CD - (DUA5 + DUB5 + Z * DUC5) * SD
    DU5 = (DUA4 + DUB4 - Z * DUC4) * SD + (DUA5 + DUB5 - Z * DUC5) * CD
    DU6 = DUA6 + DUB6 + Z * DUC6
    DU7 = (DUA7 + DUB7 + Z * DUC7) * CD - (DUA8 + DUB8 + Z * DUC8) * SD
    DU8 = (DUA7 + DUB7 - Z * DUC7) * SD + (DUA8 + DUB8 - Z * DUC8) * CD
    DU9 = DUA9 + DUB9 + Z * DUC9
    DU10 = (DUA10 + DUB10 + Z * DUC10) * CD - (DUA11 + DUB11 + Z * DUC11) * SD
    DU11 = (DUA10 + DUB10 - Z * DUC10) * SD + (DUA11 + DUB11 - Z * DUC11) * CD
    DU9 += DUC0
    DU10 += DUC1 * CD - DUC2 * SD
    DU11 += -DUC1 * SD - DUC2 * CD
    U0-=DU0; U1-=DU1; U2-=DU2; U3-=DU3; U4-=DU4 ; U5-=DU5 ; U6-=DU6; U7-=DU7; U8-=DU8; U9-=DU9; U10-=DU10; U11-=DU11
    # K=1 & J=0:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI0, ET1, Q, SD, CD, KXI1, KET0)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET1, Q, DISL1, DISL2, DISL3)
    DUB0, DUB1, DUB2, DUB3, DUB4, DUB5, DUB6, DUB7, DUB8, DUB9, DUB10, DUB11 = UB(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET1, Q, DISL1, DISL2, DISL3)
    DUC0, DUC1, DUC2, DUC3, DUC4, DUC5, DUC6, DUC7, DUC8, DUC9, DUC10, DUC11 = UC(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI0, ET1, Q, Z, DISL1, DISL2, DISL3)
    DU0 = DUA0 + DUB0 + Z * DUC0
    DU1 = (DUA1 + DUB1 + Z * DUC1) * CD - (DUA2 + DUB2 + Z * DUC2) * SD
    DU2 = (DUA1 + DUB1 - Z * DUC1) * SD + (DUA2 + DUB2 - Z * DUC2) * CD
    DU3 = DUA3 + DUB3 + Z * DUC3
    DU4 = (DUA4 + DUB4 + Z * DUC4) * CD - (DUA5 + DUB5 + Z * DUC5) * SD
    DU5 = (DUA4 + DUB4 - Z * DUC4) * SD + (DUA5 + DUB5 - Z * DUC5) * CD
    DU6 = DUA6 + DUB6 + Z * DUC6
    DU7 = (DUA7 + DUB7 + Z * DUC7) * CD - (DUA8 + DUB8 + Z * DUC8) * SD
    DU8 = (DUA7 + DUB7 - Z * DUC7) * SD + (DUA8 + DUB8 - Z * DUC8) * CD
    DU9 = DUA9 + DUB9 + Z * DUC9
    DU10 = (DUA10 + DUB10 + Z * DUC10) * CD - (DUA11 + DUB11 + Z * DUC11) * SD
    DU11 = (DUA10 + DUB10 - Z * DUC10) * SD + (DUA11 + DUB11 - Z * DUC11) * CD
    DU9 += DUC0
    DU10 += DUC1 * CD - DUC2 * SD
    DU11 += -DUC1 * SD - DUC2 * CD
    U0-=DU0; U1-=DU1; U2-=DU2; U3-=DU3; U4-=DU4 ; U5-=DU5 ; U6-=DU6; U7-=DU7; U8-=DU8; U9-=DU9; U10-=DU10; U11-=DU11
    # K=1 & J=1:
    XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ = DCCON2(XI1, ET1, Q, SD, CD, KXI1, KET1)
    DUA0, DUA1, DUA2, DUA3, DUA4, DUA5, DUA6, DUA7, DUA8, DUA9, DUA10, DUA11 = UA(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET1, Q, DISL1, DISL2, DISL3)
    DUB0, DUB1, DUB2, DUB3, DUB4, DUB5, DUB6, DUB7, DUB8, DUB9, DUB10, DUB11 = UB(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET1, Q, DISL1, DISL2, DISL3)
    DUC0, DUC1, DUC2, DUC3, DUC4, DUC5, DUC6, DUC7, DUC8, DUC9, DUC10, DUC11 = UC(C0common, XI2, ET2, Q2, R, R2, R3, R5, Y, D, TT, ALX, ALE, X11, Y11, X32, Y32, EY, EZ, FY, FZ, GY, GZ, HY, HZ, XI1, ET1, Q, Z, DISL1, DISL2, DISL3)
    DU0 = DUA0 + DUB0 + Z * DUC0
    DU1 = (DUA1 + DUB1 + Z * DUC1) * CD - (DUA2 + DUB2 + Z * DUC2) * SD
    DU2 = (DUA1 + DUB1 - Z * DUC1) * SD + (DUA2 + DUB2 - Z * DUC2) * CD
    DU3 = DUA3 + DUB3 + Z * DUC3
    DU4 = (DUA4 + DUB4 + Z * DUC4) * CD - (DUA5 + DUB5 + Z * DUC5) * SD
    DU5 = (DUA4 + DUB4 - Z * DUC4) * SD + (DUA5 + DUB5 - Z * DUC5) * CD
    DU6 = DUA6 + DUB6 + Z * DUC6
    DU7 = (DUA7 + DUB7 + Z * DUC7) * CD - (DUA8 + DUB8 + Z * DUC8) * SD
    DU8 = (DUA7 + DUB7 - Z * DUC7) * SD + (DUA8 + DUB8 - Z * DUC8) * CD
    DU9 = DUA9 + DUB9 + Z * DUC9
    DU10 = (DUA10 + DUB10 + Z * DUC10) * CD - (DUA11 + DUB11 + Z * DUC11) * SD
    DU11 = (DUA10 + DUB10 - Z * DUC10) * SD + (DUA11 + DUB11 - Z * DUC11) * CD
    DU9 += DUC0
    DU10 += DUC1 * CD - DUC2 * SD
    DU11 += -DUC1 * SD - DUC2 * CD
    U0+=DU0; U1+=DU1; U2+=DU2; U3+=DU3; U4+=DU4 ; U5+=DU5 ; U6+=DU6; U7+=DU7; U8+=DU8; U9+=DU9; U10+=DU10; U11+=DU11
    return U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11
