import sys, numpy, scipy.ndimage, matplotlib, matplotlib.pyplot as plt, agama

plt.rc('font', size=8)
plt.rc('axes', linewidth=0.5)
plt.rc('xtick.major', size=2.5)
plt.rc('ytick.major', size=2.5)

if sys.hexversion <= 0x03080000:  # built-in dict was not order-preserving before Python 3.8
    import collections
    dict = collections.OrderedDict
fap = numpy.load('APOGEE_DR17.npz')
fde = numpy.load('DESI_MWS_EDR.npz')
fgl = numpy.load('GALAH_DR4.npz')
fge = numpy.load('GaiaESO_DR5.npz')
fsd = numpy.load('SDSS.npz')
fra = numpy.load('RAVE_DR6.npz')
fll = numpy.load('LAMOST_DR9_lr.npz')
flm = numpy.load('LAMOST_DR9_mr.npz')
sidap = fap['dr3_source_id']
sidde = fde['dr3_source_id']
sidgl = fgl['gaiadr3_source_id']
sidge = fge['dr3_source_id']
sidsd = fsd['dr3_source_id']
sidra = fra['dr3_source_id']
sidll = fll['dr3_source_id']
sidlm = flm['dr3_source_id']

# data from the Gaia archive for the union of the RVS sample and all other ground-based surveys;
# note that this archive contains more data than needed for the plot,
# namely it also has photometry, parallax and PM, so can be used for various galactic dynamics applications
# with the combined 6d sample from all surveys
dic = numpy.load('CombineGaia.npz')
gaia_id   = dic['source_id']
gaia_ra   = dic['ra']
gaia_dec  = dic['dec']
gaia_gmag = dic['phot_g_mean_mag']
gaia_vlos = dic['radial_velocity']
gaia_vlose= dic['radial_velocity_error']
gaia_grvs = dic['grvs_mag']
gaia_teff = dic['rv_template_teff']
gaia_snr  = dic['rv_expected_sig_to_noise']
gaia_num  = dic['rv_nb_transits']
uservs    = numpy.isfinite(gaia_vlos)
# apply velocity offset: hot stars - Blomme+22, otherwise Katz+22
gaia_vlos -= numpy.where(gaia_teff>=8500,
    numpy.where((gaia_grvs>=6) * (gaia_grvs<=12), 7.98 - 1.125*gaia_grvs, 0),
    numpy.where(gaia_grvs>=11, 2.81 - 0.5586*gaia_grvs + 0.02755*gaia_grvs**2, 0) )
# Babusiaux+22 error inflation factor
gaia_vlose *= numpy.where( gaia_grvs>=12, 16.554- 2.4899*gaia_grvs + 0.09933*gaia_grvs**2,
              numpy.where( gaia_grvs>=8,  0.318 + 0.3884*gaia_grvs - 0.02778*gaia_grvs**2, 0 ) )

def xmatch(a, b):
    '''
    given two arrays a,b with unique elements, return indices of elements of a and b that appear in both arrays
    '''
    ua=numpy.unique(a)
    ub=numpy.unique(b)
    c = numpy.concatenate((a, b))
    o = numpy.argsort(c, kind='mergesort')
    s = c[o]
    u = numpy.where(numpy.hstack((s[1:]==s[:-1], False)))[0]
    return o[u], o[u+1]-len(a)

class Catalogue(object):
    def __init__(self, **kw):
        self.dashes = kw['dashes']
        order = numpy.argsort(kw['source_id'], kind='mergesort')  # preserve original order in case of ties
        for k,v in kw.items():
            if isinstance(v, numpy.ndarray):  setattr(self, k, v[order])   # all arrays are sorted by gaia source_id
            else: setattr(self, k, v)
        usid, uind = numpy.unique(self.source_id, return_index=True)
        # usid[0] may be 0, meaning no xmatch in gaia, but all other values should be present in gaia_id
        if usid[0]<=0:  usid = usid[1:]; uind = uind[1:]
        assert usid[0]>0
        iu, ig = xmatch(usid, gaia_id)
        assert iu[0]==0 and numpy.all(iu[1:]-iu[:-1] == 1)  # iu should be 0,1,2,...
        self.ind_gaia = ig   # index of each unique star from this catalogue in the global table
        for k,v in kw.items():
            if isinstance(v, numpy.ndarray):
                setattr(self, 'uniq_'+k, getattr(self, k)[uind])   # initially, pick up the first of any duplicate entries
        assert numpy.all(self.uniq_source_id == usid)
        print(self)

    def __str__(self):
        return (('%-10s: %i entries, %i unique sources, %i Gaia xmatches, %i unique xmatches, '
            '%i with vlos, %i good vlos, %i with Gaia vlos') %
            (self.name, len(self.obj_id), len(numpy.unique(self.obj_id)), numpy.sum(self.source_id>0), len(self.uniq_source_id),
            numpy.sum(numpy.isfinite(self.vlos)),
            numpy.sum(numpy.isfinite(self.vlos)*self.quality_flag),
            numpy.sum(numpy.isfinite(gaia_vlos[self.ind_gaia])) ))

    def compare(cat1, cat2):
        usid1 = cat1.uniq_source_id[numpy.isfinite(cat1.uniq_vlos) * cat1.uniq_quality_flag]
        usid2 = cat2.uniq_source_id[numpy.isfinite(cat2.uniq_vlos) * cat2.uniq_quality_flag]
        return len(xmatch(usid1, usid2)[0])


cats = [
    Catalogue(name='Gaia', xpos=12, ypos=16, ang=15, obj_id=gaia_id[uservs], source_id=gaia_id[uservs],
        vlos=gaia_vlos[uservs], e_vlos=gaia_vlose[uservs],
        quality_flag=((gaia_snr>=4) * (gaia_num>=4) * (gaia_vlose>0))[uservs],
        dashes=[9999,1] ),
    Catalogue(name='Gaia-ESO', xpos=17.3, ypos=0.42, ang=8, obj_id=range(len(sidge)), source_id=sidge,
        vlos=fge['vrad'] - 0* numpy.nan_to_num(fge['vrad_offset']), e_vlos=fge['e_vrad'],
        quality_flag=(fge['vrad_flag']<=0) * (fge['e_vrad']<4.99),   # add more filters?
        feh=fge['feh'], logg=fge['logg'], teff=fge['teff'],
        dashes=[4,1,1,1,1,1] ),
    Catalogue(name='GALAH', xpos=12, ypos=0.23, ang=0, obj_id=fgl['sobject_id'], source_id=sidgl,
        vlos=fgl['rv_comp_1'], e_vlos=fgl['e_rv_comp_1'],
        quality_flag=(fgl['flag_sp']<=1) * (fgl['snr_px_ccd3']>=30) * (fgl['e_rv_comp_1']<1.0),
        feh=fgl['fe_h'], logg=fgl['logg'], teff=fgl['teff'],
        dashes=[2,1] ),
    Catalogue(name='RAVE', xpos=11, ypos=1, ang=0, obj_id=fra['rave_obs_id'], source_id=sidra,
        vlos=fra['vlos'] - 0* fra['vlos_correction'], e_vlos=fra['e_vlos'],
        quality_flag=(fra['correlation']>=10) * (abs(fra['vlos_correction'])<=10) * (fra['snr']>=20),
        feh=fra['mh'], logg=fra['logg'], teff=fra['teff'],
        dashes=[4,1,1,1] ),
    Catalogue(name='LAMOST', xpos=15, ypos=10, ang=0, obj_id=fll['designation'], source_id=sidll,
        vlos=fll['vlos'] + 5.0, e_vlos=fll['e_vlos'],  # add a fixed velocity offset of 5 km/s to LAMOST LR
        quality_flag=(fll['snr']>=10) * (fll['e_vlos']>0),
        feh=fll['feh'], logg=fll['logg'], teff=fll['teff'],
        dashes=[8,2] ),
    Catalogue(name='SDSS', xpos=20, ypos=20, ang=50, obj_id=fsd['objid'], source_id=sidsd,
        vlos=fsd['vlos'], e_vlos=fsd['e_vlos'],  # suggested multiplicative and additive factors for error inflation: (1.3,2km/s) or (1.5,1.5)
        quality_flag=(fsd['gmag']<25) * (fsd['e_vlos']<200) * (fsd['programme'].astype(str) != 'premarvels_preselect'), # * numpy.isfinite(fsd['teff']),
        feh=fsd['feh'], logg=fsd['logg'], teff=fsd['teff'],
        dashes=[4,1,4,1,1,1] ),
    Catalogue(name='DESI', xpos=20, ypos=2, ang=50, obj_id=range(len(sidde)), source_id=sidde,
        vlos=fde['vlos'], e_vlos=fde['e_vlos'], quality_flag=(fde['snr']>5),
        dashes=[4,2] ),
    Catalogue(name='APOGEE', xpos=16.2, ypos=0.07, ang=0, obj_id=fap['apogee_id'], source_id=sidap,
        vlos=fap['vhelio_avg'],
        #e_vlos=fap['verr'],
        #e_vlos= ( (3.5*fap['verr']**1.2)**2 + 0.072**2)**0.5,  # APW+20
        e_vlos= ( (2.0*fap['verr']**1.2)**2 + 0.06**2)**0.5,  # my own estimate
        quality_flag=(fap['nvisits']>=1) * (fap['snr']>=10) * numpy.isfinite(fap['verr']) *
        #(fap['vscatter']<0.5) * (fap['verr']<0.3) * (fap['nvisits']>=3) *   # supposedly a clean and binary-free sample
        (fap['aspcapflag'] & (2**10 + 2**16 + 2**17 + 2**23) == 0) *
        (fap['starflag']   & (2**0  + 2**3  + 2**4  + 2**9  + 2**12 + 2**13 + 2**16) == 0),
        feh=fap['fe_h'], logg=fap['logg'], teff=fap['teff'],
        dashes=[8,1,1,1] ),
]

xm = numpy.array([ [a.compare(b) for b in cats] for a in cats])
feh, teff, logg = numpy.zeros((3, len(gaia_id))) * numpy.nan
for i in range(len(cats)):
    if hasattr(cats[i], 'uniq_feh'):
        primary = cats[i].name in ['APOGEE', 'GALAH', 'GaiaESO']
        def merge(a, b, primary_b):
            at = a[cats[i].ind_gaia]
            overwrite = numpy.isfinite(b) if primary_b else numpy.isnan(at)
            at[overwrite] = b[overwrite]
            a[cats[i].ind_gaia] = at
        merge(feh , cats[i].uniq_feh , primary)
        merge(teff, cats[i].uniq_teff, primary)
        merge(logg, cats[i].uniq_logg, primary)

print('total # of stars with Fe/H: %i, Teff: %i, logg: %i' %
    (numpy.sum(numpy.isfinite(feh)), numpy.sum(numpy.isfinite(teff)), numpy.sum(numpy.isfinite(logg)) ))
# this combined sample may be used in other applications besides just this plot

def projectHammer(lon, lat):
    sinlat, coslat = numpy.sin(lat), numpy.cos(lat)
    sinlon2,coslon2= numpy.sin(lon/2), numpy.cos(lon/2)
    X = 2*sinlon2 * coslat / (1 + coslat*coslon2)**.5
    Y = sinlat / (1 + coslat*coslon2)**.5
    return X, Y

project=projectHammer
gridm=numpy.linspace(5, 21, 65)
gride=numpy.logspace(-1.7, 2.0, 75)
img = numpy.ones((len(gride)-1, len(gridm)-1, 3))

plt.figure(figsize=(7.2, 3.6), dpi=150)
ay = plt.axes([0.695, 0.38, 0.3, 0.60])
az = plt.axes([0.695, 0.08, 0.3, 0.30])
cmap=plt.get_cmap('circle')
for i in range(len(cats)):
    color=numpy.array([0.6,0.6,0.6]) if i==0 else numpy.array(cmap( i*1.0/(len(cats)-1) ))[0:3]
    if i==0: x,y=0,0
    else:
        ang = 2*numpy.pi * i/(len(cats)-1)
        if i==3 or i==6: ang -= 0.05
        if i==4 or i==1: ang += 0.05
        x=numpy.sin(ang)
        y=numpy.cos(ang)
    ax = plt.axes([0.325 + x*0.23 - 0.1, 0.50 + y*0.33 - 0.1, 0.20, 0.20])

    ig   = cats[i].ind_gaia
    cgmag= gaia_gmag[ig]
    l, b = agama.transformCelestialCoords(agama.fromICRStoGalactic, gaia_ra[ig] * numpy.pi/180, gaia_dec[ig] * numpy.pi/180)
    X, Y = project(l, b)
    gridX, gridY = numpy.linspace(-2, 2, 201), numpy.linspace(-1, 1, 101)
    hist = numpy.histogram2d(X, Y, bins=(gridX, gridY))[0]
    img  = plt.get_cmap('Greys')( numpy.log(hist+1).T / numpy.log(numpy.max(hist)) )
    XX   = numpy.tile  ( (gridX[1:]+gridX[:-1])/2, len(gridY)-1).reshape(len(gridY)-1, len(gridX)-1)
    YY   = numpy.repeat( (gridY[1:]+gridY[:-1])/2, len(gridX)-1).reshape(len(gridY)-1, len(gridX)-1)
    img[(XX/2.0)**2 + YY**2 > 1.0] = 0
    ax.imshow(img, extent=[-2,2,-1,1], aspect='auto', origin='lower', interpolation='none')
    ax.add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, color=color, lw=1.0, fill=False, clip_on=False))
    ax.set_xlim(2.02,-2.02); ax.set_ylim(-1.01,1.01)
    ax.set_axis_off()
    ax.text(0, 1.01, cats[i].name, ha='center', va='bottom', fontsize=10 if i>0 else 11, color=color)
    ay.text(cats[i].xpos, cats[i].ypos, cats[i].name, ha='center', va='center', rotation=cats[i].ang, fontsize=8, color=color)

    filt = numpy.isfinite(cats[i].uniq_vlos) * cats[i].uniq_quality_flag
    hist = numpy.histogram2d(cgmag[filt], cats[i].uniq_e_vlos[filt], bins=(gridm, gride))[0]
    ay.contourf((gridm[1:]+gridm[:-1])/2, (gride[1:]*gride[:-1])**0.5,
        numpy.log(scipy.ndimage.gaussian_filter(hist, 1.0)).T / numpy.log(numpy.max(hist)),
        colors=matplotlib.colors.rgb2hex(color), levels=[0.5,1.0], alpha=0.15)
    ct = ay.contour((gridm[1:]+gridm[:-1])/2, (gride[1:]*gride[:-1])**0.5,
        numpy.log(scipy.ndimage.gaussian_filter(hist, 1.0)).T / numpy.log(numpy.max(hist)),
        colors=matplotlib.colors.rgb2hex(color), levels=[0.5,], linewidths=1.0)
    if hasattr(ct, 'collections'):
        for cc in ct.collections:
            cc.set_dashes([(0,cats[i].dashes)])
    else:
        ct.set_dashes([(0,cats[i].dashes)])
    gmin,gmax = numpy.floor(numpy.nanmin(cgmag[filt])), numpy.ceil(numpy.nanmax(cgmag[filt]))
    gnodes = numpy.linspace(gmin, gmax, int(gmax-gmin)*4+1)
    spl = agama.splineLogDensity(gnodes, numpy.nan_to_num(cgmag[filt]), numpy.ones(numpy.sum(filt)),
        infLeft=False, infRight=False)
    ggrid = numpy.linspace(gmin, gmax, int(gmax-gmin)*12+1)
    az.fill_between(ggrid, ggrid*0+0.1, numpy.exp(spl(ggrid)), edgecolor=color, label=cats[i].name,
        facecolor=numpy.hstack([color, 0.1])).set_dashes([(0,cats[i].dashes)])

ay.set_yscale('log')
ay.set_xlim(min(gridm), max(gridm))
ay.set_ylim(min(gride), max(gride))
az.set_yscale('log')
az.set_xlim(min(gridm), max(gridm))
az.set_ylim(50,2e7)
az.set_yticks(numpy.logspace(2, 7, 6))
ay.set_xticks(numpy.linspace(6, 20, 8))
az.set_xticks(numpy.linspace(6, 20, 8))
az.set_xlabel('G', labelpad=1)
az.set_ylabel('number of stars', labelpad=1)
ay.set_xticklabels([])
ay.set_ylabel('      velocity uncertainty [km/s]', labelpad=0)
ay.set_yticklabels(['','','0.1','1','10','100'])
ay.tick_params(direction='in', which='both', top=False, right=False)
az.tick_params(direction='in', which='both', top=False, right=False)
plt.savefig('fig2_spectroscopic_surveys.pdf')
