import numpy, matplotlib.pyplot as plt
import agama
agama.setUnits(length=1, velocity=1, mass=1)
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)

b15 = agama.Potential('MWPotential2014.ini')
m17 = agama.Potential('McMillan17.ini')
p22 = agama.Potential('PriceWhelan22.ini')
h24 = agama.Potential('Hunter24_axi.ini')
m17mc = numpy.loadtxt('McMillan17_MCMCsample.txt')
plt.figure(figsize=(3.3, 2.4))
ax1 = plt.axes([0.135, 0.14, 0.83, 0.83])
r   = numpy.logspace(-1, 2.5, 71)
z   = numpy.linspace(0, 2.5, 51)
xyz = numpy.column_stack((r,r*0,r*0))
zyx = numpy.column_stack((z*0+8.2,z*0,z))
ax1.plot(r, (-r * b15.force(xyz)[:,0])**0.5, label='Bovy15', c='b', dashes=[2,1])
ax1.plot(r, (-r * m17.force(xyz)[:,0])**0.5, label='McMillan17', c='g', dashes=[5,2,1,2])
ax1.plot(r, (-r * p22.force(xyz)[:,0])**0.5, label='PriceWhelan22', c='gray', dashes=[4,2])
ax1.plot(r, (-r * h24.force(xyz)[:,0])**0.5, label='Hunter24', c='r')
r0 = 8.2
print('local Vcirc: B15=%5.1f, M17=%5.1f, P22=%5.1f, H24=%5.1f' % (
    (-r0*b15.force(r0,0,0)[0])**0.5, (-r0*m17.force(r0,0,0)[0])**0.5, (-r0*p22.force(r0,0,0)[0])**0.5, (-r0*h24.force(r0,0,0)[0])**0.5))
m17rc = []
for i in range(100):
    par = m17mc[i*50]
    m17rc.append((-r * agama.Potential(
        dict(type='disk', surfaceDensity=par[ 0], scaleRadius=par[ 1], scaleHeight=par[ 2], innerCutoffRadius=par[ 3]),
        dict(type='disk', surfaceDensity=par[ 5], scaleRadius=par[ 6], scaleHeight=par[ 7], innerCutoffRadius=par[ 8]),
        dict(type='disk', surfaceDensity=par[10], scaleRadius=par[11], scaleHeight=par[12], innerCutoffRadius=par[13]),
        dict(type='disk', surfaceDensity=par[15], scaleRadius=par[16], scaleHeight=par[17], innerCutoffRadius=par[18]),
        dict(type='spheroid', densityNorm=par[20], axisRatioZ=par[21], gamma=par[22], beta=par[23], scaleRadius=par[24], outerCutoffRadius=par[25]),
        dict(type='spheroid', densityNorm=par[26], axisRatioZ=par[27], gamma=par[28], beta=par[29], scaleRadius=par[30], outerCutoffRadius=par[31] if par[31]>0 else numpy.inf),
    ).force(xyz)[:,0])**0.5)
m17rc = numpy.vstack(m17rc)
m17l, m17h = numpy.percentile(m17rc, [16,84], axis=0)
ax1.fill_between(r, m17l, m17h, color='g', alpha=0.33, lw=0)
ax1.set_xscale('log')
ax1.set_xlim(0.1, 300)
ax1.set_ylim(0, 250)
ax1.set_xticklabels(['','0.1','1','10','100'])
ax1.set_xticklabels(['']*9+['0.3']+['']*7+['3']+['']*7+['30']+['']*7+['300'], minor=True)
ax1.set_xlabel('radius [kpc]', labelpad=3)
ax1.set_ylabel('circular velocity [km/s]')
ax1.legend(loc='lower right', frameon=False, fontsize=8, handlelength=2.7)
plt.savefig('fig20_mw_rotation_curve_compilation.pdf')

plt.figure(figsize=(6.6, 3.7))
ax = plt.axes([0.08, 0.09, 0.9, 0.9])
data = numpy.loadtxt('mwmass.txt', str)
prev = None
methods = ['DF', 'sim', 'TME', 'Je', 'stream']
methnam = ['DF', 'simulations', 'TME', 'Jeans', 'stream']
methind = [1, 1, 1, 1, 1]
# dummy
plt.errorbar([numpy.nan], [numpy.nan], c='w', lw=0, label='$\,$')
# real
for line in data:
    radius = float(line[1])
    mass   = 10**float(line[2])
    mminus =-10**(float(line[2])-float(line[3])) + mass
    mplus  = 10**(float(line[2])+float(line[4])) - mass
    name   = line[0]
    plt.errorbar(numpy.array([radius]), numpy.array([mass]), color=line[8], marker=line[7],
        mec='none' if line[7]!='x' else None, lw=0, elinewidth=1, capsize=3, ms=5,
        yerr=numpy.array([[mminus], [mplus]]), label=name if name!=prev else None)
    if name != prev:
        for im in range(len(methods)):
            if methods[im] in line[6]:
                if '/' in line[6]:
                    plt.plot(0.24, 0.94-im*0.04, 's', color=line[8], mec='none', ms=5, transform=ax.transAxes)
                else:
                    plt.plot(0.24+methind[im]*0.02, 0.96-im*0.04, 's', color=line[8], mec='none', ms=5, transform=ax.transAxes)
                    methind[im]+=1
                break
    prev = name
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,600)
plt.ylim(1e11, 2e12)
ax.set_xticklabels(['','10','100'], minor=False)
ax.set_xticklabels(['','','','','','','','','20','','','50','','','','','200'], minor=True)
plt.xlabel('radius [kpc]', fontsize=10, labelpad=2)
plt.ylabel('enclosed mass [$M_\odot$]', fontsize=10, labelpad=0)
plt.legend(loc='lower right', frameon=False, numpoints=1, ncol=2, fontsize=8, handletextpad=0, columnspacing=1)
plt.plot(0.03, 0.96, '<', color='gray', transform=ax.transAxes, mec='none')
plt.plot(0.03, 0.92, '>', color='gray', transform=ax.transAxes, mec='none')
plt.plot(0.03, 0.88, 'D', color='gray', transform=ax.transAxes, mec='none')
plt.plot(0.03, 0.84, 'o', color='gray', transform=ax.transAxes, mec='none')
plt.plot(0.03, 0.80, 'x', color='gray', transform=ax.transAxes, mew=1.5)
plt.text(0.05, 0.96, 'globular clusters', transform=ax.transAxes, ha='left', va='center', fontsize=8)
plt.text(0.05, 0.92, 'satellites', transform=ax.transAxes, ha='left', va='center', fontsize=8)
plt.text(0.05, 0.88, 'both', transform=ax.transAxes, ha='left', va='center', fontsize=8)
plt.text(0.05, 0.84, 'halo stars', transform=ax.transAxes, ha='left', va='center', fontsize=8)
plt.text(0.05, 0.80, 'stream', transform=ax.transAxes, ha='left', va='center', fontsize=8)
for im in range(len(methods)):
    plt.text(0.39, 0.96-im*0.04, methnam[im], transform=ax.transAxes, ha='left', va='center', fontsize=8)

rrr = numpy.logspace(0, 3, 100)
fid = numpy.column_stack((rrr, p22.enclosedMass(rrr), (-rrr*p22.force(numpy.column_stack((rrr,rrr*0,rrr*0)))[:,0])**0.5))
plt.plot(fid[:,0], fid[:,1], '--', color='gray', zorder=-9)
plt.text(450, 1.28e12, 'fiducial model', ha='center', va='center', rotation=16, fontsize=7, color='gray')
if not True:
    plt.plot(rrr, b15.enclosedMass(rrr), color='gray', dashes=[5,2,1,2], zorder=-9)
    plt.plot(rrr, m17.enclosedMass(rrr), color='gray', dashes=[2,2], zorder=-9)
    plt.text(400, 1.6e12, 'McMillan 2017', ha='center', va='center', rotation=16.5, fontsize=7, color='gray')
    plt.text(400, 1.0e12, 'Bovy 2015', ha='center', va='center', rotation=16., fontsize=7, color='gray')
plt.savefig('fig16_mw_mass_profile.pdf')

plt.figure(figsize=(6.6, 3.7))
ax = plt.axes([0.08, 0.09, 0.9, 0.9])
data = (
    ('Mroz19', 'o', 'lightgray'),
    ('Eilers19', '>', 'orangered'),
    ('Ablimit20', '<', 'olivedrab'),
    ('Zhou23', '^', 'dodgerblue'),
    #('Wang23', 'v', 'mediumorchid'),
    ('Jiao23', 'v', 'mediumorchid'),
    ('Poder23', 'd', 'orange'),
    ('Ou24', 'x', 'slategray'),
    #('SylosLabini23', 'D', 'gold'),
)
for name, marker, colour in data:
    tab = numpy.loadtxt('rc_%s.txt' % name)
    if name == 'Mroz19':
        plt.plot(tab[:,0], tab[:,2], marker=marker, color=colour, mec='none', lw=0, label=name, ms=2.0, zorder=-1)
    else:
        plt.errorbar(tab[:,0], tab[:,1], yerr=tab[:,2:].T if tab.shape[1]>3 else tab[:,2], marker=marker, color=colour,
        mec='none' if name != 'Ou24' else colour, ms=5-1 if name == 'Ou24' else 4, lw=0,
        label=name, elinewidth=1, capsize=2, alpha=0.75)
plt.xlim(0, 30)
plt.gca().set_yticks(numpy.linspace(150, 270, 13))
plt.ylim(145, 275)
plt.xlabel('radius [kpc]', fontsize=10, labelpad=2)
plt.ylabel('circular velocity [km/s]', fontsize=10, labelpad=6)
plt.legend(loc='upper right', frameon=False, numpoints=1, fontsize=8, handletextpad=0, columnspacing=1)
plt.plot(fid[:,0], fid[:,2], '--', color='gray', zorder=-9)
plt.text(27.5, 202, 'fiducial model', ha='center', va='center', rotation=-5, fontsize=7, color='gray')
plt.savefig('fig17_mw_rotation_curve.pdf')

plt.figure(figsize=(3.3, 1.6))
plt.axes([0.01, 0.21, 0.98, 0.77])
data = numpy.loadtxt('vesc.txt', str)
for i, (name, vesc, minus, plus, sym, col) in enumerate(data):
    plt.errorbar(float(vesc), i, xerr=([float(minus)], [float(plus)]), marker=sym,
        mec='none' if sym=='o' else col, ms=3 if sym=='o' else 4, mew=1, c=col, capsize=2)
    if name=='Koppelman21':
        plt.errorbar(float(vesc)*1.1, i, xerr=([float(minus)*1.1], [float(plus)*1.1]), marker=sym,
            mec='none' if sym=='o' else col, ms=3 if sym=='o' else 4, mew=1, c='r', capsize=2)
    plt.text(380, i, name, ha='left', va='center', fontsize=8)
plt.ylim(i+0.6, -0.6)
plt.xlim(370, 650)
plt.gca().set_xticks([450,500,550,600])
plt.gca().set_yticks([])
plt.xlabel('escape velocity [km/s]', labelpad=3)
plt.savefig('fig19_mw_escape_velocity.pdf')
