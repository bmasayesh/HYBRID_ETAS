# translated by Sebastian Hainzl from DC3D.f of Okada and pscokada.f of Ronjiang Wang 
# tested in TEST-OKADA-PYTHON-CODE/compare_with_Wangcode.py

import sys
import numpy as np
import translated_DC3D_from_Okada as dc3d

def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3        
    R = R0 * np.arccos(
        np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)))
    return R

def stresstensor_pscokada(alpha, mu, lmda, lats, lons, zs, lat, lon, zz, strike, dip, rake, L, W, slip):
    '''
    Calculation of the deformation and stresses components in Aki & Richard coordinate system

    Input: alpha, mu [Pa], lmda [Pa]     Rock parameters alpha, shear and lambda modulus
           lats, lons, zs [km]           Location of the slip patch
                                         ... coordinates are given for top-center of each subfault (as in fsp-slip models)
           lat, lon, z [km]              Location, where stress is calculated (vectors)
           strike, dip, rake             Mechanism of the slip patch
           slip [m]                      Slip value of the patch
           L [km], W [km]                Length and Width of the slip patch
    OUTPUT: ux, uy, uz                   [m] Deformation in Aki-Richard coordinates:
                                         # x = N-direction
                                         # y = E-direction
                                         # z = depth-direction
    sxx, syy, szz, sxy, syz, szx         [Pa] stress components
    '''
 
    x = np.sign(lat-lats) * dist(lats, lons, lat, lons)
    y = np.sign(lon-lons) * dist(lats, lons, lats, lon)
    km2m = 1000.0
    zs *= km2m
    x *= km2m
    y *= km2m
    z = zz * km2m
    L *= km2m
    W *= km2m
    deg2rad = np.pi /180.0
    st = strike * deg2rad
    csst = np.cos(st)
    ssst = np.sin(st)
    cs2st = np.cos(2.0*st)
    ss2st = np.sin(2.0*st)
    di = dip * deg2rad
    csdi = np.cos(di)
    ssdi = np.sin(di)
    # rake: gegen Uhrzeigersinn  & slip_z: positiv in Richtung Tiefe 
    DISL1 = np.cos(deg2rad * rake) * slip
    DISL2 = -np.sin(deg2rad * rake) * slip
    DISL3= 0.0
    AL1 = -0.5 * L
    AL2 = 0.5 * L
    AW1 = -W      
    AW2 = 0
    # transform from Aki's to Okada's system:
    X = x * csst + y * ssst
    Y = x * ssst - y * csst
    Z = -z      # z:  corresponds to the recording depth zrec
    depth = zs  # zs: corresponds to the depth of slip (reference depth: zref)
    #
    UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ = dc3d.DC3D(alpha, X, Y, Z, depth, dip, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3)
    # transform from Okada's to Aki's system:
    ux = UX * csst + UY * ssst          # back rotation (strike --> -strike: csst ---> csst & ssst --> -ssst)      
    uy = UX * ssst - UY * csst
    uz = -UZ 
    strain1 = UXX * csst * csst + UYY * ssst * ssst + 0.5 * (UXY + UYX) * ss2st
    strain2 = UXX * ssst * ssst + UYY * csst * csst - 0.5 * (UXY + UYX) * ss2st
    strain3 = UZZ
    strain4 = 0.5 * (UXX - UYY) * ss2st - 0.5 * (UXY + UYX) * cs2st
    strain5 = -0.5 * (UZX + UXZ) * ssst + 0.5 * (UYZ + UZY) * csst
    strain6 = -0.5 * (UZX + UXZ) * csst - 0.5 * (UYZ + UZY) * ssst
    eii = strain1 + strain2 + strain3
    sxx = lmda * eii + 2.0 * mu * strain1
    syy = lmda * eii + 2.0 * mu * strain2
    szz = lmda * eii + 2.0 * mu * strain3
    sxy = 2.0 * mu * strain4
    syz = 2.0 * mu * strain5
    szx = 2.0 * mu * strain6
    return ux, uy, uz, sxx, syy, szz, sxy, syz, szx 
