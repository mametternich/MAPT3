def boundariesDiagnotic(self,plateID=None,plot=False):
        """
        This function returns of an automatic diagnotic on the plate
        boundary type and fill the internal field .btype that have the
        same lenght as self.lonb.
        NOTE 1: You have to have import before vorticity, divergence data,
                temperature and continental composition data.
        NOTE 2: You can give a plate ID value to have a verbose diagnostic on
                a particular plate.
        """
        if len(self.hvorb) != len(self.lonb):
            self.im('You have to import vorticity data before',error=True)
        elif len(self.hdivb) != len(self.lonb):
            self.im('You have to import divergence data before',error=True)
        elif len(self.Tb) != len(self.lonb):
            self.im('You have to import temperature data before',error=True)
        else:
            self.im('Automatic plates boundaries diagnostic')
            mSub1 = self.hdivb < -1000
            mSub2 = self.vrb < -50
            mSub  = mSub1 * mSub2
            mRid1 = self.hdivb > +1000
            mRid2 = self.vrb > +50
            mRid  = mRid1 * mRid2
            mTra1 = abs(self.hvorb) > 10000
            mTra  = mTra1 * ~mSub * ~mRid
            mOth  = ~mTra * ~mSub * ~mRid
            pbtype = np.zeros(self.lonb.shape,dtype=np.int32)
            pbtype[mSub] = 1
            pbtype[mRid] = 2
            pbtype[mTra] = 3
            pbtype[mOth] = 4
            self.pbtype = pbtype
            self.im('Creat a mask to mask internal plate bounaries')
            mask = np.ones(self.lonb.shape,dtype=bool)
            for i in range(len(self.lonb)):
                if self.platecouple[i,0] == self.platecouple[i,1]:
                    mask[i] = False
            # --- count
            sub = np.count_nonzero(pbtype[mask] == 1)/len(pbtype[mask])
            rid = np.count_nonzero(pbtype[mask] == 2)/len(pbtype[mask])
            tra = np.count_nonzero(pbtype[mask] == 3)/len(pbtype[mask])
            oth = np.count_nonzero(pbtype[mask] == 4)/len(pbtype[mask])
            print('-'*30)
            self.im('Automatic Plate boundary diagnostic: ALL PLATES')
            self.im('Subduction: '+str(int(sub*10000)/100)+'%')
            self.im('MOR:        '+str(int(rid*10000)/100)+'%')
            self.im('Transform:  '+str(int(tra*10000)/100)+'%')
            self.im('Other:      '+str(int(oth*10000)/100)+'%')
            # Compute the heat flux
            depth_subsurf = 3.2*1000 # depth in meters
            k0 = 3.15 # W.m^{-1}.K^{-1}
            q0 = k0*(self.T-0.12)/depth_subsurf
            # compute a temp compositionary field: 1 if continent, 0 elsewhere
            COMPO = np.zeros(len(self.x),dtype=np.int32)
            COMPO[self.cont > 0] = 1
            # prepare indices
            ids = np.arange(len(self.x))
            subDir = np.zeros(len(pbtype),dtype=np.int32)-1
            for i in range(len(pbtype)):
                if pbtype[i] == 1:
                    p1,p2 = self.platecouple[i]
                    ms1   = self.plateID  == p1
                    ms2   = self.plateID  == p2
                    dist1 = np.sqrt((self.xb[i]-self.x[ms1])**2+(self.yb[i]-self.y[ms1])**2+(self.zb[i]-self.z[ms1])**2)
                    dist2 = np.sqrt((self.xb[i]-self.x[ms2])**2+(self.yb[i]-self.y[ms2])**2+(self.zb[i]-self.z[ms2])**2)
                    maxDist = 0.035
                    mh1   = dist1 <= maxDist
                    mh2   = dist2 <= maxDist
                    if np.count_nonzero(mh1) == 0:
                        q01   = 0
                        comp1 = 0
                    else:
                        q01   = np.mean(q0[ids[ms1][mh1]])
                        comp1 = np.mean(COMPO[ids[ms1][mh1]])
                    if np.count_nonzero(mh2) == 0:
                        q02   = 0
                        comp2 = 0
                    else:
                        q02   = np.mean(q0[ids[ms2][mh2]])
                        comp2 = np.mean(COMPO[ids[ms2][mh2]])
                    if comp1 >= 0.75 and comp2 < 0.75:
                        subDir[i] = p2
                    elif comp2 >= 0.75 and comp1 < 0.75:
                        subDir[i] = p1
                    else:
                        if q01 >= q02:
                            subDir[i] = p2
                        else:
                            subDir[i] = p1
                            
            # now cleaning ! Test all plate couples that have in common a subduction boundary and
            # and check if the solution is consistent and homogene for the entire plate boundary
            tested_couples = []
            bad_couples    = []
            subDir_new = subDir.copy()
            for i in range(len(self.xb)):
                couplei = list(self.platecouple[i,:])
                if couplei not in tested_couples and couplei[::-1] not in tested_couples and \
                couplei not in bad_couples    and couplei[::-1] not in bad_couples:
                    cp1 = couplei[0]
                    cp2 = couplei[1]
                    m11 = self.platecouple[:,0] == cp1
                    m12 = self.platecouple[:,1] == cp2
                    m21 = self.platecouple[:,0] == cp2
                    m22 = self.platecouple[:,1] == cp1
                    m   = m11*m12 + m21*m22
                    ms  = self.pbtype[m] == 1
                    if np.count_nonzero(ms) == 0:
                        bad_couples.append(couplei)
                    else:
                        tested_couples.append(couplei)
                        idb = np.arange(len(self.lonb))
                        subDirp1 = np.count_nonzero(subDir[idb[m][ms]] == cp1)/len(idb[m][ms])
                        subDirp2 = 1-subDirp1
                        if subDirp1 >= 0.75:
                            # dominated by the subduction of the plate p1 beneath the plate p2
                            subDir_new[idb[m][ms]] = cp1
                        elif subDirp2 >= 0.75:
                            # dominated by the subduction of the plate p2 beneath the plate p1
                            subDir_new[idb[m][ms]] = cp2

            subDir_old  = subDir.copy()
            self.subDir = subDir_new
            
            if plot:
                fig = plt.figure()
                ax  = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
                ax.set_title('Diagnostic on the plate boundaries: All bounaries')
                ax.set_global()
                #ax.scatter(self.lon,self.lat,s=1,c=self.plateID,cmap='jet',transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mSub],self.latb[mSub],s=3,c='blue',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mSub],self.latb[mSub],s=3,c=subDir[mSub],cmap='jet',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mRid],self.latb[mRid],s=3,c='red',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mTra],self.latb[mTra],s=3,c='green',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mOth],self.latb[mOth],s=3,c='k',alpha=1,transform=ccrs.PlateCarree())
                plt.show()
                #
                fig = plt.figure()
                ax  = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
                ax.set_title('Diagnostic on the plate boundaries: Subduction zones WITHOUT cleaning')
                ax.set_global()
                ax.scatter(self.lon,self.lat,s=1,c=self.plateID,cmap='jet',transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mSub],self.latb[mSub],s=3,c=subDir_old[mSub],cmap='jet',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mRid],self.latb[mRid],s=3,c='black',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mTra],self.latb[mTra],s=3,c='black',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mOth],self.latb[mOth],s=3,c='k',alpha=1,transform=ccrs.PlateCarree())
                plt.show()
                #
                fig = plt.figure()
                ax  = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
                ax.set_title('Diagnostic on the plate boundaries: Subduction zones WITH cleaning')
                ax.set_global()
                ax.scatter(self.lon,self.lat,s=1,c=self.plateID,cmap='jet',transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mSub],self.latb[mSub],s=3,c=subDir_new[mSub],cmap='jet',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mRid],self.latb[mRid],s=3,c='black',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mTra],self.latb[mTra],s=3,c='black',alpha=1,transform=ccrs.PlateCarree())
                ax.scatter(self.lonb[mOth],self.latb[mOth],s=3,c='k',alpha=1,transform=ccrs.PlateCarree())
                plt.show()

            if plateID is not None:
                mp1 = self.platecouple[:,0] == plateID
                mp2 = self.platecouple[:,1] == plateID
                mp  = mp1 + mp2
                mask = np.ones(self.lonb[mp].shape,dtype=bool)
                for i in range(len(self.lonb[mp])):
                    if self.platecouple[:,0][mp][i] == self.platecouple[:,1][mp][i]:
                        mask[i] = False
                pbtypei = pbtype[mp][mask]
                sub = np.count_nonzero(pbtypei == 1)/len(pbtypei)
                rid = np.count_nonzero(pbtypei == 2)/len(pbtypei)
                tra = np.count_nonzero(pbtypei == 3)/len(pbtypei)
                oth = np.count_nonzero(pbtypei == 4)/len(pbtypei)
                print('-'*30)
                self.im('Automatic Plate boundary diagnostic: pID='+str(plateID))
                self.im('Subduction: '+str(int(sub*10000)/100)+'%')
                self.im('MOR:        '+str(int(rid*10000)/100)+'%')
                self.im('Transform:  '+str(int(tra*10000)/100)+'%')
                self.im('Other:      '+str(int(oth*10000)/100)+'%')