'''
Calculation of distance to fault for different slip models in 3D.
It has calculte distance to the sliped subfaults (slip > 0) or distance to the 
subfaults with slip larger than 15% of maximum slip (slip > 0.15 * max_slip)
'''

########################## Importing Required Modules #########################
import sys
import numpy as np
sys.path.append('RECIPES')
import read_fsp_file

############################# Functions & Classes #############################
def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3        
    R = R0 * np.arccos(
        np.clip(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)), -1, 1))
    return R

def alldist(lat1, lon1, z1, lat2, lon2, z2):
    """
    Distance (in [km]) between points given as [lat,lon]
    """
    R0 = 6367.3        
    Rxy = R0 * np.arccos(
        np.clip(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)), -1, 1))
    R = np.sqrt(np.square(Rxy) + np.square(z1-z2))
    return R

def define_grid_circle(Seqdir, dr, name):
    Mcut = 1.95
    Z1 = 0.0
    Z2 = 30.0
    R = 100.0     # [km]
    T1 = -300.0   # [days] start time of the sequence
    T2 = 100.0    # [days] end time of the sequence
    data = np.loadtxt('%s/california_1981_2022-Mcut%.2f-R100.0km-T%.0f-%.0fdays-%s.kat' % (Seqdir, Mcut, T1, T2, name), skiprows=1)
    latmain = data[0, 2]
    lonmain = data[0, 3]
    lat2km = dist(latmain-0.5, lonmain, latmain+0.5, lonmain)
    lon2km = dist(latmain, lonmain-0.5, latmain, lonmain+0.5)
    gridr = np.arange(0.5*dr, 100, dr)
    gridlat = np.sort(np.append(latmain-gridr/lat2km, latmain+gridr/lat2km))
    gridlon = np.sort(np.append(lonmain-gridr/lon2km, lonmain+gridr/lon2km))
    gridz = np.arange(Z1+0.5*dr, Z2, dr)
    Nlat = len(gridlat)
    Nlon = len(gridlon)
    Nz = len(gridz)
    lats = np.zeros(Nlat*Nlon)
    lons = np.zeros(Nlat*Nlon)
    n = 0
    for i in range(Nlat):
        for j in range(Nlon):
            lats[n] = gridlat[i]
            lons[n] = gridlon[j]
            n += 1
            
    r = dist(lats, lons, latmain, lonmain)
    ind = (r<=R)  
    lonc = lons[ind]
    latc = lats[ind]
    NSurf = len(lonc)
    lati = np.zeros(NSurf*Nz)
    loni = np.zeros(NSurf*Nz)
    zi = np.zeros(NSurf*Nz)
    m = 0
    for i in range(NSurf):
        for j in range(Nz):
            lati[m] = latc[i]
            loni[m] = lonc[i]
            zi[m] = gridz[j]
            m += 1   
    return lati, loni, zi

####################### Inputs and Outputs  directories #######################
Seqdir  = '../INPUTS/CATALOGS/SEQUENCES'                            ## directory of sequences
Slipdir  = '../INPUTS/SLIP-MODELS'                                  ## directory of slip models
Distdir_all = '../OUTPUTS/DISTANCE-RESULTS/ALL-SUBFAULTS/3D'        ## directory of distance to all subfaults
Distdir_max = '../OUTPUTS/DISTANCE-RESULTS/MAX-SLIP-SUBFAULTS/3D'   ## directory of diatance to subfaults with slip > 15% of max slip 

############################ Defining of Parameters ###########################  
dr = 1.0       # [km] grid-spacing
Z1 = 0.0       # [km] minimum depth
Z2 = 30.0      # [km] maximum depth

###############################################################################    
names = ['Landers', 'BajaCalifornia', 'HectorMine', 'Ridgecrest', 'Northridge', 'SuperstitionHill']
dr = 1.0   # [km] spatial grid spacing in x, y, z direction

for nname in range(len(names)):
# for nname in range(0, 1):
    name = names[nname]

    # Observation points:
    gridlat, gridlon, gridz = define_grid_circle(Seqdir, dr, name)
    Ntot = len(gridlat)
    
    # slip models:
    if name == 'Landers':
        slipmodels = ['s1992LANDER01WALD', 's1992LANDER01COHE',  's1992LANDER01COTT', 's1992LANDER01HERN', 's1992LANDER01ZENG']
    elif name == 'BajaCalifornia':
        slipmodels = ['s2010ELMAYO01MEND', 's2010ELMAYO01WEIx']
    elif name == 'HectorMine':
        slipmodels = ['s1999HECTOR01JIxx', 's1999HECTOR01JONS', 's1999HECTOR01KAVE', 's1999HECTOR01SALI']
    elif name == 'Ridgecrest':
        slipmodels = ['s2019RIDGEC02JINx', 's2019RIDGEC02ROSS', 's2019RIDGEC04XUxx']
    elif name == 'Northridge':
        slipmodels = ['s1994NORTHR01DREG', 's1994NORTHR01HART', 's1994NORTHR01HUDN', 's1994NORTHR01SHEN', 's1994NORTHR01WALD', 's1994NORTHR01ZENG']
    elif name == 'SuperstitionHill':
        slipmodels = ['s1987SUPERS01LARS', 's1987SUPERS01WALD']
        
    for ns in range(len(slipmodels)):
        
        print('\n%s:\n' % (name))
        print('%s:\n' % (slipmodels[ns]))
        
        datname = '%s/%s.fsp' % (Slipdir, slipmodels[ns])
        lat_hypo, lon_hypo, z_hypo, yy, mm, dd, M, Nfault, strike, dip, rake, rake_all, L, W, lat, lon, z, slip = read_fsp_file.read_fsp_file(datname)
        slips = np.array(slip)
        ind1 = slips > 0
        lats = np.array(lat)
        lons = np.array(lon)
        zs = np.array(z)
        lats1 = lats[ind1]
        lons1 = lons[ind1]
        zs1 = zs[ind1]
        
        max_slip = max(slips)
        ind2 = slips > (0.15 * max_slip)
        lats2 = lats[ind2]
        lons2 = lons[ind2]
        zs2 = zs[ind2]
        
        dis1 = np.zeros(Ntot)
        for i in range(Ntot):
            dis1[i] = np.min(alldist(gridlat[i], gridlon[i], gridz[i], lats1, lons1, zs1))
        
        dis2 = np.zeros(Ntot)
        for i in range(Ntot):
            dis2[i] = np.min(alldist(gridlat[i], gridlon[i], gridz[i], lats2, lons2, zs2))
            
        outname1 = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir_all, slipmodels[ns], dr)
        np.savetxt(outname1, np.column_stack((gridlat, gridlon, gridz, dis1)), fmt='%f\t%f\t%f\t%e', header='lat\tlon\tz\tdist')
        print('\n\t OUTPUT: %s\n' % (outname1))
        
        outname2 = '%s/distance2fault-%s-dr%.1fkm.out' % (Distdir_max, slipmodels[ns], dr)
        np.savetxt(outname2, np.column_stack((gridlat, gridlon, gridz, dis2)), fmt='%f\t%f\t%f\t%e', header='lat\tlon\tz\tdist')
        print('\n\t OUTPUT: %s\n' % (outname2))
