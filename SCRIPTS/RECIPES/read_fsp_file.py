# written by Shubham Sharma 
############### Function For Reading the SRCMOD Slip models ###################

def read_fsp_file(inname):
    Nfault = []
    lat = []
    lon = []
    z = []
    slip = []
    rake = []
    strike = []
    dip = []
    L = []
    W = []
    run = False
    fin = open(inname, "r")
    for line in fin:
        column = line.split()
        Ncol = len(column)
        if column:
            Ncol = len(column)
            if column[0] == "%":
                #print(column)
                if Ncol >= 7 and column[1] == "LAT" and column[2] == "LON" and column[3] == "X==EW":
                    run = True
                    setrakeall = True
                    if Ncol > 7 and column[7] == "RAKE":
                        setrakeall = False
                    #print(setrakeall)
                for n in range(Ncol):
                    if column[n].find('/', 5) > 0:
                        date = column[n].split('/')
                        mm = int(date[0])
                        dd = int(date[1])
                        yy = int(date[2])
                    if column[n] == 'Loc' and Ncol >= n+4 and column[n+3] == '=':
                        lat_hypo = float(column[n+4])
                        lon_hypo = float(column[n+7])
                        z_hypo = float(column[n+10])
                    if column[n] == 'Mw' and Ncol >= n+2 and column[n+1] == '=':
                        M = float(column[n+2])
                    if column[n] == 'STRK' and Ncol >= n+2 and column[n+1] == '=':
                        strike_all = float(column[n+2])
                    if column[n] == 'DIP' and Ncol >= n+2 and column[n+1] == '=':
                        dip_all = float(column[n+2])
                    if column[n] == 'RAKE' and Ncol >= n+2 and column[n+1] == '=':
                        rake_all = float(column[n+2])
                    if column[n] == 'STRIKE' and Ncol >= n+2 and column[n+1] == '=' :
                        strike_all = float(column[n+2])
                    if column[n] == 'DIP' and Ncol >= n+2 and column[n+1] == '=' :
                        dip_all = float(column[n+2])
                    if column[n] == 'Dx' and Ncol >= n+2 and column[n+1] == '=' :
                        Dx = float(column[n+2])
                    if column[n] == 'Dz' and Ncol >= n+2 and column[n+1] == '=' :
                        Dz = float(column[n+2])
                    if column[n] == 'Nsbfs' and column[n+1] == '=':
                        Nfault.append(int(column[n+2]))
            elif run:
                strike.append(strike_all)
                dip.append(dip_all)
                L.append(Dx)
                W.append(Dz)
                lat.append(float(column[0]))
                lon.append(float(column[1]))
                z.append(float(column[4]))
                slip.append(float(column[5]))
                if setrakeall:
                    rake.append(rake_all)
                else:
                    rake.append(float(column[6]))
    fin.close()
    return lat_hypo, lon_hypo, z_hypo, yy, mm, dd, M, Nfault, strike, dip, rake, rake_all, L, W, lat, lon, z, slip 
