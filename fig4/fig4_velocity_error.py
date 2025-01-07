import numpy, matplotlib.pyplot as plt, matplotlib
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
plt.figure(figsize=(3.3, 3.3))
plt.axes([0.12, 0.12, 0.85, 0.85])

def angular_distance(ra0, dec0, ra1, dec1):
    '''
    Compute the angular distance between two points on a sphere (coordinates expressed in degrees)
    '''
    d2r = numpy.pi/180  # degrees to radians
    return 2 * numpy.arcsin( (numpy.sin( (dec0-dec1)*0.5 * d2r )**2 +
        numpy.cos(dec0 * d2r) * numpy.cos(dec1 * d2r) * numpy.sin( (ra0-ra1)*0.5 * d2r )**2 )**0.5 ) / d2r

def remove_galaxies(ra,dec):
    return (
    (angular_distance(ra, dec,  10.7,  41.3) >  3.0) *  # M31
    (angular_distance(ra, dec,  81.3, -69.8) > 15.0) *  # LMC
    (angular_distance(ra, dec,  13.2, -72.8) > 10.0) *  # SMC
    (angular_distance(ra, dec, 285.0, -30.5) >  8.0) *  # Sgr
    (angular_distance(ra, dec, 100.4, -51.0) >  2.0) *  # Car
    (angular_distance(ra, dec, 260.0,  57.9) >  2.0) *  # Dra
    (angular_distance(ra, dec,  40.0, -34.4) >  2.0) *  # Fnx
    (angular_distance(ra, dec,  15.0, -33.7) >  2.0) *  # Scl
    (angular_distance(ra, dec, 153.3,  -1.6) >  2.0) *  # Sxt
    (angular_distance(ra, dec, 227.3,  67.2) >  2.0) *  # UMi
    (angular_distance(ra, dec, 114.5,  38.9) >  0.2) *  # NGC2419
    True)

rrl = numpy.loadtxt('Li2023_RRLcatalogue_addcolumns.txt', str)
rrl = rrl[remove_galaxies(rrl[:,1].astype(float), rrl[:,2].astype(float)) *
    (rrl[:,10].astype(float) < 0.2*rrl[:,9].astype(float)) *  # rel dist error
    (rrl[:, 7].astype(float) < 0.5)  # ebv
    ]

dist, e_dist, plx, e_plx, pmra, e_pmra, pmdec, e_pmdec = rrl[:,(9,10,12,13,14,15,16,17)].astype(float).T
rel_e_dist = numpy.minimum(e_dist/dist, e_plx/numpy.maximum(plx, 1e-10))
pm = (pmra**2 + pmdec**2)**0.5
e_pm = (e_pmra**2 + e_pmdec**2 + 2*0.025**2)**0.5
err1 = 4.74 * e_pm * dist
err2 = 4.74 * pm * dist * rel_e_dist

plt.scatter(dist, err1, color='r', linewidths=0, edgecolors='none', s=0.5, alpha=0.3, rasterized=True)
plt.scatter(dist, err2, color='b', linewidths=0, edgecolors='none', s=0.5, alpha=0.3, rasterized=True)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5, 100)
plt.ylim(0.1, 500)

#plt.plot([1, 5, 30], [5, 25, 25], color='k')
#plt.plot([1, 5, 30], [0.2, 1, 25], color='k')
plt.plot(0.7, 170, 'bo', ms=4, mew=0)
plt.plot(0.7, 100, 'ro', ms=4, mew=0)
plt.text(0.6, 300, 'source of uncertainty', ha='left', va='center', fontsize=8)
plt.text(0.9, 170, 'distance', ha='left', va='center', fontsize=8)
plt.text(0.9, 100, 'proper motion', ha='left', va='center', fontsize=8)
plt.text(1.6, 0.2, 'systematic', ha='left', va='bottom', rotation=30, fontsize=8)
plt.text( 10, 1.5, 'random',     ha='left', va='bottom', rotation=50, fontsize=8)
plt.text(  1, 3.0, 'parallax',   ha='left', va='bottom', rotation=35, fontsize=8)
plt.text(  8, 50., 'photometry', ha='left', va='bottom', rotation=-8, fontsize=8)
#plt.scatter(numpy.nan, numpy.nan, color='b', linewidths=0, edgecolors='none', s=2, label='distance'

class MyFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self, value, pos=None):
        if   value==0.1: return '$\\mathdefault{0.1}$'
        if   value==1:   return '$\\mathdefault{1}$'
        elif value==10:  return '$\\mathdefault{10}$'
        elif value==100: return '$\\mathdefault{100}$'
        else: return matplotlib.ticker.LogFormatterMathtext.__call__(self, value)
#plt.legend(loc='lower left', frameon=False, fontsize=8)
plt.gca().xaxis.set_major_formatter(MyFormatter())
plt.gca().yaxis.set_major_formatter(MyFormatter())
plt.xlabel('distance [kpc]', fontsize=8)
plt.ylabel('velocity uncertainty [km/s]', fontsize=8, labelpad=0)
plt.savefig('fig4_velocity_error.pdf')
