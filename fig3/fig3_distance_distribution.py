import numpy, matplotlib.pyplot as plt, matplotlib
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
plt.figure(figsize=(3.3, 3.3))
plt.axes([0.14, 0.14, 0.85, 0.85])

d_clus = numpy.loadtxt('globclust.txt', usecols=(1,))
d_dsph = numpy.loadtxt('dsph.txt', usecols=(1,))
strm   = numpy.loadtxt('streams.txt', str)
d_strm = (strm[:, 3].astype(float) + strm[:,2].astype(float) + strm[:,1].astype(float)) / 3
db_apo = numpy.loadtxt('APOGEE.txt')
db_rrl = numpy.loadtxt('RRLyrae.txt')
db_xp  = numpy.loadtxt('gaia_bprp.txt')
db_p5  = numpy.loadtxt('gaia_plx5.txt')

def show(sample, **params):
    mdist = numpy.sort(-numpy.nan_to_num(sample))
    mdist = mdist[mdist<0]
    n1 = 20          # up to n1: show steps
    n2 = len(mdist)  # above n1: connect individual points by lines
    x = numpy.hstack([numpy.repeat(mdist[:n1], 2), mdist[n1:n2]])
    y = numpy.hstack([numpy.repeat(numpy.linspace(0, n1, n1+1), 2)[1:-1], numpy.linspace(n1+1, n2, n2-n1)])
    plt.plot(-x, y, **params)

show(d_clus, label='globular clusters')
show(d_dsph, label='satellites')
show(d_strm, label='streams')
plt.plot(db_rrl[:,0], db_rrl[:,1], label='RR Lyrae')
plt.plot(db_apo[:,0], db_apo[:,1], label='APOGEE')
plt.plot(db_xp[::-1,0], numpy.cumsum(db_xp[::-1,1]), label='BP/RP spectra')
plt.plot(db_p5[::-1,0], numpy.cumsum(db_p5[::-1,1]), label=r'${\varpi/\epsilon_\varpi>5}$', color='gray')
plt.plot(db_p5[::-1,0], numpy.cumsum(db_p5[::-1,2]), label=r'${\varpi/\epsilon_\varpi>5}$, RVS', color='black')
class MyFormatter(matplotlib.ticker.LogFormatterMathtext):
    def __call__(self, value, pos=None):
        if   value==1:   return '$\\mathdefault{1}$'
        elif value==10:  return '$\\mathdefault{10}$'
        elif value==100: return '$\\mathdefault{100}$'
        else: return matplotlib.ticker.LogFormatterMathtext.__call__(self, value)
plt.legend(loc='lower left', frameon=False, fontsize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1, 300)
plt.ylim(0.8, 2e8)
plt.gca().set_xticklabels(['','0.1','1','10','100'])
plt.gca().yaxis.set_major_formatter(MyFormatter())
plt.xlabel('distance [kpc]', fontsize=8)
plt.ylabel('number of objects within a given distance', fontsize=8, labelpad=4)
plt.savefig('fig3_distance_distribution.pdf')
