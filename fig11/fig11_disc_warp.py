import numpy, matplotlib, matplotlib.pyplot as plt, mpl_toolkits.mplot3d
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
plt.figure(figsize=(3.45,2.3), dpi=200)
ax=plt.axes([0.05,0.05,1,1], projection='3d')
ax.view_init(elev=30, azim=210)
ax.xaxis.set_pane_color((1.0,)*4)
ax.yaxis.set_pane_color((1.0,)*4)
ax.zaxis.set_pane_color((1.0,)*4)
ax.set_xlabel('\nX [kpc]', linespacing=1)
ax.set_ylabel('\nY [kpc]', linespacing=1)
ax.set_zlabel('\nZ [kpc]', linespacing=1)
f=[0,0.5,1]; r=[0.2,0.6,1.0]; g=[0.2,0.6,0.2]; b=[1.0,0.6,0.2]
cmap=matplotlib.colors.LinearSegmentedColormap('bluegrayred', dict(red=zip(f,r,r), green=zip(f,g,g), blue=zip(f,b,b)))
r=numpy.linspace(0, 16, 17)[1:]
phi=numpy.linspace(0, 2*numpy.pi, 65)
Phi0 = lambda r: (-12 * (((numpy.maximum(0, r-11.5) / 1.0)**2 + 1)**0.5 - 1) + 190) * numpy.pi/180
Z0   = lambda r: -1.0 * (((numpy.maximum(0, r-10.0) / 3.0)**2 + 1)**0.5 - 1)
R,Phi=numpy.meshgrid(r,phi)
X=R*numpy.cos(Phi); Y=R*numpy.sin(Phi)
Z=Z0(R) * numpy.sin(Phi - Phi0(R))
for i in range(len(phi)-1):
    for j in range(len(r)-1):
        c=numpy.array(cmap(numpy.clip(0.5*(Z[i,j]+Z[i,j+1]) / 1.0 + 0.5, 0, 1)))
        c[0:3] = numpy.clip(X[i,j]/5.0, 0, 1) * (0.8-c[0:3]) + c[0:3]
        ax.plot(X[i,j:j+2], Y[i,j:j+2], Z[i,j:j+2], c=c, lw=0.5)
    for j in range(len(r)):
        c=numpy.array(cmap(numpy.clip(0.5*(Z[i,j]+Z[i+1,j]) / 1.0 + 0.5, 0, 1)))
        c[0:3] = numpy.clip(X[i,j]/5.0, 0, 1) * (0.8-c[0:3]) + c[0:3]
        ax.plot(X[i:i+2,j], Y[i:i+2,j], Z[i:i+2,j], c=c, lw=0.5)
Rsun = 8.2
ax.plot([-Rsun], [0], [0], '*', c='k', zorder=12)
class Arrow3D(matplotlib.patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        matplotlib.patches.FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        matplotlib.patches.FancyArrowPatch.draw(self, renderer)

phi=numpy.linspace(-numpy.pi, -numpy.pi*4/3, 16)[1:]
xx,yy=Rsun*numpy.cos(phi), Rsun*numpy.sin(phi)
colrot = '#008000'
ax.plot(xx, yy, phi*0, c=colrot)
ax.add_artist(Arrow3D([xx[-1], xx[-1]+(xx[-1]-xx[-2])*5], [yy[-1], yy[-1]+(yy[-1]-yy[-2])*5], [0,0],
    color=colrot, mutation_scale=10, arrowstyle='-|>', zorder=10))
# line of nodes
rr=numpy.linspace(11.0, 16.0, 11)
ax.plot(rr * numpy.cos(Phi0(rr)), rr * numpy.sin(Phi0(rr)), rr*0, c='#202020')
ax.set_xlim(-16, 16)
ax.set_ylim(-16, 16)
ax.set_zlim(-2, 2)
plt.savefig('fig11_disc_warp.pdf')
