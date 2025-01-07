import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(7.,7.35))

flw=0
fcpsz=8
fsz=10
elw=2

qual='royalblue'
quant='crimson'
historic='lightsteelblue'

xoff=60
yoff=-0.25

plt.errorbar(y=-0.2,x=50.875,xerr=[3],marker='o',capsize=fcpsz,label='Dehnen (2000)',lw=flw,ms=fsz,elinewidth=elw,color=historic)
plt.text(xoff,-0.3-yoff,'Dehnen (2000)',color='k') 
plt.errorbar(y=0.8,x=39,xerr=[3.5],marker='x',capsize=fcpsz,label='Portail (2017)',lw=flw,ms=fsz,elinewidth=elw,color=historic)
plt.text(xoff,0.7-yoff,'Portail et al. (2017)',color='k') 

plt.plot([0,100],[1.5,1.5],ls=':',color='dimgrey')

plt.errorbar(y=2,x=51.7,xerr=[0],marker='+',capsize=0,label='Monari et al. (2017)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,2-yoff,'Monari et al. (2017)',color='k') 
plt.errorbar(y=3,x=39,xerr=[0],marker='o',capsize=0,label='Perez-Villegas et al. (2017)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,3-yoff,'Perez-Villegas et al. (2017)',color='k')
plt.errorbar(y=4,x=35.75,xerr=[0],marker='o',capsize=0,label='Hunt et al. (2018)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,4-yoff,'Hunt et al. (2018)',color='k')
plt.errorbar(y=5,x=41,xerr=[3],marker='o',capsize=fcpsz,label='Michtchenko et al. (2018b)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,5-yoff,'Michtchenko et al. (2018b)',color='k')
plt.errorbar(y=6,x=54,xerr=[0],marker='+',capsize=0,label='Ramos et al. (2018)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,6-yoff,'Ramos et al. (2018)',color='k')
plt.errorbar(y=7,x=41,xerr=[3],marker='x',capsize=fcpsz,label='Bovy et al. (2019)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,7-yoff,'Bovy et al. (2019)',color='k')
plt.errorbar(y=8,x=37.5,xerr=[3],marker='x',capsize=fcpsz,label='Clarke et al. (2019)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,8-yoff,'Clarke et al. (2019)',color='k')
plt.errorbar(y=9,x=39,xerr=[0],marker='+',capsize=0,label='Monari et al. (2019a)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,9-yoff,'Monari et al. (2019a)',color='k')
plt.errorbar(y=10,x=42.5,xerr=[2.5],marker='+',capsize=fcpsz,label='Asano et al. (2020)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,10-yoff,'Asano et al. (2020)',color='k')
plt.errorbar(y=11,x=36,xerr=[0],marker='o',capsize=0,label='Binney (2020)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,11-yoff,'Binney (2020)',color='k')
plt.errorbar(y=12,x=35.5,xerr=[0.8],marker='+',capsize=fcpsz,label='Chiba et al. (2021)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,12-yoff,'Chiba et al. (2021)',color='k')
plt.errorbar(y=[13,13,13],x=[46.5,34.5,51.5],xerr=[1.5,1.5,0.5],marker='s',capsize=fcpsz,label='Trick et al. (2021)',lw=flw,ms=fsz/2,elinewidth=elw,color=qual)
plt.text(xoff,13-yoff,'Trick et al. (2021)',color='k')
plt.errorbar(y=[14,14],x=[34,42],xerr=[0,0],marker='s',capsize=0,label='Kawata et al. (2021)',lw=flw,ms=fsz/2,elinewidth=elw,color=qual)
plt.text(xoff,14-yoff,'Kawata et al. (2021)',color='k') #2
plt.errorbar(y=[15],x=[38.5],xerr=[0],marker='s',capsize=0,label='Trick (2022)',lw=flw,ms=fsz/2,elinewidth=elw,color=qual)
plt.text(xoff,15-yoff,'Trick (2022)',color='k')
plt.errorbar(y=16,x=33.29,xerr=[1.81],marker='x',capsize=fcpsz,label='Clarke et al. (2022)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,16-yoff,'Clarke et al. (2022)',color='k')
plt.errorbar(y=17,x=38.1,xerr=2,marker='x',capsize=fcpsz,label='Gaia Collaboration et al (2023)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,17-yoff,'Gaia Collaboration et al. (2023)',color='k')
plt.errorbar(y=18,x=40.08,xerr=[1.78],marker='x',capsize=fcpsz,label='Leung et al. (2023)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,18-yoff,'Leung et al. (2023)',color='k')
plt.errorbar(y=19,x=41,xerr=[3],marker='x',capsize=fcpsz,label='Sanders et al. (2023)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,19-yoff,'Sanders et al. (2023)',color='k')
plt.errorbar(y=20,x=[37.5],xerr=[2.5],marker='^',capsize=fcpsz,label='Dillamore et al. (2023)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,20-yoff,'Dillamore et al. (2023)',color='k')
plt.errorbar(y=21,x=40,xerr=[0],marker='+',capsize=0,label='Lucchini et al. (2024)',lw=flw,ms=fsz*1.5,elinewidth=elw,color=qual)
plt.text(xoff,21-yoff,'Lucchini et al. (2024)',color='k')
plt.errorbar(y=[22],x=[35],xerr=0,marker='^',capsize=0,label='Dillamore et al. (2024)',lw=flw,ms=fsz,elinewidth=elw,color=qual)
plt.text(xoff,22-yoff,'Dillamore et al. (2024)',color='k')
plt.errorbar(y=23,x=24,xerr=[3],marker='x',capsize=fcpsz,label='Horta et al. (2024)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,23-yoff,'Horta et al. (2024)',color='k')
plt.errorbar(y=24,x=34.1,xerr=[2.4],marker='x',capsize=fcpsz,label='Zhang et al. (2024)',lw=flw,ms=fsz,elinewidth=elw,color=quant)
plt.text(xoff,24-yoff,'Zhang et al. (2024)',color='k')

plt.text(22,1,'"Pre $Gaia$"',color='dimgrey')
plt.text(22,2.3-yoff,'"Post $Gaia$"',color='k')

plt.plot([0,100],[-1,-1],ls='-',color='dimgrey')

plt.plot([23],[-2],marker='o',color='black')
plt.text(25,-1.75,'UV plane',color='black')
plt.plot([37],[-2],marker='+',color='black')
plt.text(39,-1.75,r'$R-v_{\phi}$',color='black')
plt.plot([48],[-2],marker='x',color='black')
plt.text(50,-1.75,'Inner MW',color='black')
plt.plot([62],[-2],marker='s',color='black')
plt.text(64,-1.75,'Action-Angle',color='black')
plt.plot([79],[-2],marker='^',color='black')
plt.text(81,-1.75,'Halo',color='black')

plt.figtext(0.57, 0.082, 'Type of estimation:', fontsize=13,color='k')
plt.figtext(0.57, 0.057, ' - Qualitative comparison', fontsize=13,color=qual)
plt.figtext(0.57, 0.027, ' - Quantitative fit', fontsize=13,color=quant)

plt.ylim(25,-3)
plt.xlim(20,90)
plt.ylabel('<- Newer              Older ->',fontsize=20)
plt.gca().set_yticks([])
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.gca().set_xticks([20,25,30,35,40,45,50,55])

plt.xlabel(r'$\ \ \ \Omega_{\mathrm{b}}\ (\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1})$',fontsize=20,loc='left')

plt.title(r'$\mathrm{Bar\ pattern\ speed\ in\ the}\ Gaia\ \mathrm{era}$',fontsize=15)

plt.savefig('fig10_bar_pattern_speed.pdf',bbox_inches='tight')

plt.show()
