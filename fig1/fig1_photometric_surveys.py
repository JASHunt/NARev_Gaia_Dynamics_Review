import numpy, agama, matplotlib, matplotlib.pyplot as plt

plt.figure(figsize=(3.2, 1.8))
ax = [
    plt.axes([0.01, 0.51, 0.45, 0.40]),
    plt.axes([0.54, 0.51, 0.45, 0.40]),
    plt.axes([0.01, 0.01, 0.45, 0.40]),
    plt.axes([0.54, 0.01, 0.45, 0.40])]

def pointInsidePolygon(x, y, polygon):
    ncorners = len(polygon)
    wind = numpy.zeros_like(x)
    for i in range(ncorners):
        v = polygon[i]
        w = polygon[(i+1)%ncorners]
        sign = (w[0]-v[0]) * (y-v[1]) - (w[1]-v[1]) * (x-v[0])
        wind[(v[1]<=y) * (w[1]> y) * (sign>0)] += 1
        wind[(v[1]> y) * (w[1]<=y) * (sign<0)] -= 1
    return wind!=0

def project(lon, lat):  # Hammer
    sinlat, coslat = numpy.sin(lat), numpy.cos(lat)
    sinlon2,coslon2= numpy.sin(lon/2), numpy.cos(lon/2)
    X = 2*sinlon2 * coslat / (1 + coslat*coslon2)**.5
    Y = sinlat / (1 + coslat*coslon2)**.5
    return X, Y

gridX, gridY = numpy.linspace(-2, 2, 201), numpy.linspace(-1, 1, 101)
centrX, centrY = numpy.repeat((gridX[1:]+gridX[:-1])/2, len(gridY)-1), numpy.tile((gridY[1:]+gridY[:-1])/2, len(gridX)-1)

data = numpy.load('sdss.npy')
img  = numpy.where(data, 0.4, 1.0).T
img  = numpy.repeat(img.reshape(-1), 3).reshape(img.shape+(3,))
ax[0].imshow(img, extent=[-2,2,-1,1], aspect='auto', origin='lower', interpolation='none')
ax[0].add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, color='k', lw=1.0, fill=False, clip_on=False))
ax[0].set_xlim(2.02,-2.02)
ax[0].set_ylim(-1.01,1.01)
ax[0].set_axis_off()
ax[0].text(0, 1.05, 'SDSS', ha='center', va='bottom', fontsize=8)

data = numpy.load('decals.npy')
img  = numpy.where(data, 0.4, 1.0).T
img  = numpy.repeat(img.reshape(-1), 3).reshape(img.shape+(3,))
ax[2].imshow(img, extent=[-2,2,-1,1], aspect='auto', origin='lower', interpolation='none')
ax[2].add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, color='k', lw=1.0, fill=False, clip_on=False))
ax[2].set_xlim(2.02,-2.02)
ax[2].set_ylim(-1.01,1.01)
ax[2].set_axis_off()
ax[2].text(0, 1.05, 'Legacy surveys', ha='center', va='bottom', fontsize=8)

l, b = numpy.loadtxt('panstarrs.txt').T * numpy.pi/180
X, Y = project(l, b)
img  = ~pointInsidePolygon(centrX, centrY, numpy.column_stack([X,Y]))
img  = numpy.where(img * ((centrX/2)**2 + centrY**2 <= 1.0), 0.4, 1.0).reshape(len(gridX)-1, len(gridY)-1).T
img  = numpy.repeat(img.reshape(-1), 3).reshape(img.shape+(3,))
ax[1].imshow(img, extent=[-2,2,-1,1], aspect='auto', origin='lower', interpolation='none')
ax[1].add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, color='k', lw=1.0, fill=False, clip_on=False))
ax[1].set_xlim(2.02,-2.02)
ax[1].set_ylim(-1.01,1.01)
ax[1].set_axis_off()
ax[1].text(0, 1.05, 'PanSTARRS', ha='center', va='bottom', fontsize=8)

l, b = numpy.loadtxt('decaps.txt').T * numpy.pi/180
X, Y = project(l, b)
img  = pointInsidePolygon(centrX, centrY, numpy.column_stack([X,Y]))
img  = numpy.where(img * ((centrX/2)**2 + centrY**2 <= 1.0), 0.4, 1.0).reshape(len(gridX)-1, len(gridY)-1).T
img  = numpy.repeat(img.reshape(-1), 3).reshape(img.shape+(3,))
ax[3].imshow(img, extent=[-2,2,-1,1], aspect='auto', origin='lower', interpolation='none')
ax[3].add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, color='k', lw=1.0, fill=False, clip_on=False))
ax[3].set_xlim(2.02,-2.02)
ax[3].set_ylim(-1.01,1.01)
ax[3].set_axis_off()
ax[3].text(0, 1.05, 'DECaPS', ha='center', va='bottom', fontsize=8)

vvv = numpy.array([[10.4, 5.1], [-10, 5.1], [-10, 2.25], [-65, 2.25], [-65, -2.25], [-10, -2.25], [-10, -10.3], [10.4, -10.3], [10.4, 5.1]])
vvvx= numpy.array([[10.4, 10], [-10, 10], [-10, 4.5], [-130, 4.5], [-130, -4.5], [-10, -4.5], [-10, -14.5], [10.4, -14.5], [10.4, -4.5], [20, -4.5], [20, 4.5], [10.4, 4.5], [10.4, 10]])

X, Y = project(vvvx[:,0] * numpy.pi/180, vvvx[:,1] * numpy.pi/180)
ax[3].add_artist(matplotlib.patches.Polygon(numpy.column_stack((X, Y)), fill=False, color='b', alpha=0.75))
ax[3].text(0.0, -0.25, 'VVVX', color='b', fontsize=8, ha='center', va='top')
X, Y = project(vvv [:,0] * numpy.pi/180, vvv [:,1] * numpy.pi/180)
ax[3].add_artist(matplotlib.patches.Polygon(numpy.column_stack((X, Y)), fill=False, color='r', alpha=0.75))
ax[3].text(0.0, 0.15, 'VVV', color='r', fontsize=8, ha='center', va='bottom')

plt.savefig('fig1_photometric_surveys.pdf')
