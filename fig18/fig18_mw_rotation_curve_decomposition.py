import numpy, agama, matplotlib.pyplot as plt
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
plt.figure(figsize=(3.3, 2.4))
plt.axes([0.14, 0.14, 0.83, 0.83])

def getRC(filename):
    tab=numpy.loadtxt(filename, delimiter=',')
    breaks = numpy.hstack([0, numpy.where(tab[1:,0] < tab[:-1,0])[0]+1, len(tab)])
    return [agama.Spline(tab[breaks[i]:breaks[i+1],0], tab[breaks[i]:breaks[i+1],1]) for i in range(len(breaks)-1)]

r = numpy.linspace(5, 20, 31)
for files, labels, colors in [
    ['Wegg19_halo.txt', 'Wegg19', 'lime'],
    ['Nitschai21.txt', 'Nitschai21', 'mediumvioletred'],
    ['Hattori21.txt', 'Hattori21', 'chocolate'],
    ['Binney24.txt', 'Binney24', 'cornflowerblue']]:
    spl = getRC(files)
    if len(spl)==1:  # Wegg
        vtot = numpy.where((r>=8)*(r<=8.5), 217.0, numpy.nan)
        vhal = spl[0](r)
    else:
        vtot = spl[0](r)
        vhal = spl[1](r)
    vbar = (vtot**2-vhal**2)**0.5
    plt.plot(r, vtot, color=colors, label=labels)
    plt.plot(r, vhal, color=colors, dashes=[4,2])
    plt.plot(r, vbar, color=colors, dashes=[1,1])
plt.xlim(5,20)
plt.ylim(0,250)
plt.xlabel('radius [kpc]', labelpad=3)
plt.ylabel('circular velocity [km/s]', labelpad=4)
plt.legend(loc='lower right', frameon=False, fontsize=8)
plt.savefig('fig18_mw_rotation_curve_decomposition.pdf')
