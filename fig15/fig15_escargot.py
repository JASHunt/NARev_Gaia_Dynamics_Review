import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib as mpl

dominant_mode,inferred_time=np.load('Escargot_times.npy')

cmap1 = cmr.sunburst_r
cmap2 = cmr.voltage_r

jpmin=1600.
jpmax=2400.
jtmin=2.85
jtmax=3.4

inds=0
columns = 80
rows = 22
fss=30

fig, ax_array = plt.subplots(rows, columns, figsize=(15,7),sharex='col',sharey='row')
for k,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        if dominant_mode[inds]==1:
            axes.set_facecolor(cmap1(inferred_time[inds]/1000.))
        else:
            axes.set_facecolor(cmap2(inferred_time[inds]/1000.))
        inds+=1
        axes.set_axis_off()
        axes.add_artist(axes.patch)
        axes.patch.set_zorder(-1)
        
plt.subplots_adjust(wspace=0.)
plt.subplots_adjust(hspace=0.)

ax2 = plt.axes([0.125,0.11,0.815,0.77])

ax2.patch.set_alpha(0.)
ax2.set_facecolor('none')
ax2.set_xlim(jpmin,jpmax)
ax2.set_ylim(jtmin,jtmax)
ax2.tick_params(axis='both', labelsize=fss)

ax2.set_xlabel(r'$-J_{\phi}\ (\mathrm{kpc\ km\ s^{-1}})$',fontsize=fss)
ax2.set_ylabel(r'$\theta_{\phi}\ (\mathrm{rad})$',fontsize=fss)

fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes([0.95, 0.11, 0.025, 0.77]) #from left,bottom,?,height
norm = mpl.colors.Normalize(vmin=0.,vmax=1)

sm = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
sm.set_array([])
cb=fig.colorbar(sm, cax=cbar_ax)
cbar_ax.tick_params(axis='y', labelsize=fss)
cbar_ax.set_yticks([])

cbar_ax = fig.add_axes([0.975, 0.11, 0.025, 0.77]) #from left,bottom,?,height
norm = mpl.colors.Normalize(vmin=0.,vmax=1)

sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
sm.set_array([])
cb=fig.colorbar(sm, cax=cbar_ax)
cbar_ax.tick_params(axis='y', labelsize=fss)
cb.set_label(r'$\mathrm{Inferred\ impact\ time\ (Gyr)}$',fontsize=fss,fontname="serif",style="normal")

plt.figtext(0.122, 0.92, 'm=1 phase spiral dominant', fontsize=fss,color=cmap1(0.5))
plt.figtext(0.547, 0.92, 'm=2 phase spiral dominant', fontsize=fss,color=cmap2(0.5))
plt.figtext(0.522, 0.92, '/', fontsize=fss,color='black')

plt.savefig('fig15_phase_spiral_impact_time.pdf',bbox_inches='tight')

plt.show()
