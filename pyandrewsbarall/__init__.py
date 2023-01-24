from . import andrewsbarall_scenario
import matplotlib.pylab as plt
import numpy as np
import sys

class andrewsbarall:
	"""docstring for andrewsbarall"""
	def __init__(self,dx,cutratio):
		self.dx =dx
		self.cutratio = cutratio

	def scenario_from_mag(self,mag,model='Leonard2014_Interp-SS',custom_length=None,custom_width=None,rcrit=5e3,cut_h=0.01):

		self.modelname=model

		length,width = fault_size(mag,model)

		if custom_length is not None:
			if length>custom_length:
				length = custom_length
		if custom_width is not None:
			if width>custom_width:
				width = custom_width

		print('Rupture length is {0:5.0f} km\nRupture width is {1:5.0f} km'.format(length/1e3,width/1e3))
		
		self.tmpnx = np.ceil(length*2/self.dx)
		# self.tmpnz = np.ceil(min(width*2,30e3)/self.dx)
		self.tmpnz = np.ceil(max(width*2,25e3)/self.dx)
	
		print('Nx = {0}, Nz = {1}'.format(self.tmpnx,self.tmpnz))
		# scenario_generator(nx=nx,nz=nz,dx=dx,cutratio=cutratio,depthcut=width*2,half_lengthcut=length)
		self.ts,self.tn,self.mus,self.mud,self.ihypox,self.ihypoz=\
		scenario_generator_nonpower2(nx=self.tmpnx,nz=self.tmpnz,dx=self.dx,cutratio=self.cutratio,\
			                         depthcut=width*2,half_lengthcut=length*(1-cut_h),cut_h=cut_h,nucl=rcrit)
		self.nx = self.ts.shape[1]
		self.ny = self.ts.shape[0]
		return 

		

def scenario_generator_nonpower2(nx=2048,nz=2048/4,dx=50,cutratio=1,depthcut=24e3,half_lengthcut=60e3,nucl=5e3,asperity=False,\
	                             hetero=True,tn_eff=65e6,cut_h=0.1,cut_v=0.1):
	# if tn_eff > 0:
	# normal stress would be saturated to this level

	# nx = 2048 #power of 2
	# dx = 50.
	# depthcut = 24e3 #m
	# half_lengthcut = 60e3 #m
	#     Make this flag false to select the high-stress model.
	#     Make this flag true to select the low-stress asperity model.
	# asperity = False
	# hetero=True

	iseed=np.random.randint(1,1e10,4)
	# iseed=[121231223,212313,32323,32324345345]

	w,shear,normal,mus,mud,ihypox,ihypoz=\
		andrewsbarall_scenario.initiate_scenario_nonpower2(nx,nz,asperity,hetero,tn_eff,dx,cutratio,depthcut,half_lengthcut,nucl,iseed)
	

	#remove edge area for saving computational costs: 
	x_s,x_e = int(nx*cut_h),int(nx*(1-cut_h))+1
	y_s,y_e = 0,int(nz*(1-cut_v))+1
	print('Hypocentral location (x) {0} should be between {1} and {2}'.format(ihypox,x_s,x_e))
	print('Hypocentral location (y) {0} should be between {1} and {2}'.format(ihypoz,y_s,y_e))

	if ihypox <x_s or ihypox>x_e:
		sys.exit('Hypocentral location is too close to edges')

	ihypox -= x_s
	nx = x_e-x_s
	nz = y_e-y_s

	if nx < nz:
		xs2 = x_s + int((nz-nx)/2)
		xe2 = x_e + int((nz-nx)/2)
		ihypox = ihypox+int((nz-nx)/2)
		nx = nz

		total_nx = int(nx) + 20
		total_nz = int(nz) + 10
	
		final_shear =  np.zeros((total_nz,total_nx))
		final_normal = np.zeros((total_nz,total_nx))
		final_mus =    1e20*np.ones((total_nz,total_nx))
		final_mud =    1e20*np.ones((total_nz,total_nx))

		final_shear[0:-10,xs2:xe2] =   shear[x_s:x_e,y_s:y_e].T
		final_normal[0:-10,xs2:xe2] = -normal[x_s:x_e,y_s:y_e].T
		final_mus[0:-10,xs2:xe2] =     mus[x_s:x_e,y_s:y_e].T
		final_mud[0:-10,xs2:xe2] =     mud[x_s:x_e,y_s:y_e].T

		final_shear = extend_edge(final_shear,'left',xs2+1)
		final_shear = extend_edge(final_shear,'right',xe2-1)
		final_shear = extend_edge(final_shear,'down',total_nz-11)
		final_normal = extend_edge(final_normal,'left',xs2+1)
		final_normal = extend_edge(final_normal,'right',xe2-1)
		final_normal = extend_edge(final_normal,'down',total_nz-11)

	else:
		total_nx = int(nx) + 20
		total_nz = int(nz) + 10
	
		final_shear =  np.zeros((total_nz,total_nx))
		final_normal = np.zeros((total_nz,total_nx))
		final_mus =    1e20*np.ones((total_nz,total_nx))
		final_mud =    1e20*np.ones((total_nz,total_nx))
	
		final_shear[0:-10,10:-10] =   shear[x_s:x_e,y_s:y_e].T
		final_normal[0:-10,10:-10] = -normal[x_s:x_e,y_s:y_e].T
		final_mus[0:-10,10:-10] =     mus[x_s:x_e,y_s:y_e].T
		final_mud[0:-10,10:-10] =     mud[x_s:x_e,y_s:y_e].T
	
		final_shear = extend_edge(final_shear,'left',11)
		final_shear = extend_edge(final_shear,'right',total_nx-11)
		final_shear = extend_edge(final_shear,'down',total_nz-11)
		final_normal = extend_edge(final_normal,'left',11)
		final_normal = extend_edge(final_normal,'right',total_nx-11)
		final_normal = extend_edge(final_normal,'down',total_nz-11)

	# print('nx={0}   ny={1}\n'.format(int(nx),int(nz)))
	# print('ihypox={0}   ihypoy={1}\n'.format(ihypox+10,ihypoz))

	# ihypox,ihypoz = find_hypocenter(sd)
	# print(ihypox,ihypoz)

	stressdrop = final_shear + final_normal*final_mud
	ratiodrop = -final_shear/final_normal-final_mud

	# # stressdrop[stressdrop<1e9]=np.nan
	PlotFigure((w).T,(total_nz-1)*dx/1e3,(total_nx-1)*dx/1e3, -1,3,True,'gray',None,None,'pyandrewsbarall/w_nonpower2.pdf')
	PlotFigure(final_shear,(total_nz-1)*dx/1e3,(total_nx-1)*dx/1e3, None,None,True,'jet',ihypox+20,ihypoz,'pyandrewsbarall/shearstress_nonpower2.pdf')
	# PlotFigure(final_normal,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'normalstress_nonpower2.pdf')
	# PlotFigure(-final_shear/final_normal,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'ratio_nonpower2.pdf')
	# PlotFigure(final_mus,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'mus_nonpower2.pdf')
	# PlotFigure(ratiodrop,(total_nz-1)*dx/1e3,(total_nx-1)*dx/1e3, -1,1,True,ihypox+20,ihypoz,'pyandrewsbarall/ratiodrop_nonpower2.pdf')
	PlotFigure(stressdrop,(total_nz-1)*dx/1e3,(total_nx-1)*dx/1e3, -10e6,10e6,True,None,ihypox+20,ihypoz,'pyandrewsbarall/stressdrop_nonpower2.pdf')

	return final_shear,final_normal,final_mus,final_mud,ihypox+20,ihypoz


########## model prediction of rupture geometry
# only for strike-slip events
def fault_size(mag,model='Leonard2014_Interp-SS',printmodel=True):
	if printmodel:
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
		print('M-L-W scaling used is {0}\n'.format(model))
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
	if model == 'Wesnousky2008':
		A = 5.56
		B = 0.87
		length_inkm = 10**((mag-A)/B)
		width_inkm = length_inkm/(0.088*length_inkm**0.96)
	elif model == 'Wells1994':
		a = -3.55
		b = 0.74
		length_inkm = 10**(a + b * mag)
		a = -0.76
		b = 0.27
		width_inkm = 10**(a + b * mag)
	elif model == 'Leonard2014_Interp-SS':
		mag_1 = np.log10(45)+5.27
		if mag <= mag_1:
			length_inkm = 10.**((mag-4.17)/1.667)
		else:
			length_inkm = 10.**((mag-5.27))
		width_inkm = 10.**((mag-3.88)/2.5)
		if width_inkm > 19.:
			width_inkm = 19.
	elif model == 'Leonard2014_SCR-SS':
		mag_1 = np.log10(60)+5.44
		if mag <= mag_1:
			length_inkm = 10.**((mag-4.25)/1.667)
		else:
			length_inkm = 10.**((mag-5.44))
		width_inkm = 10.**((mag-4.22)/2.5)
		if width_inkm > 20.:
			width_inkm = 20.
	elif model == 'Thingbaijam2017':
		length_inkm = 10.**(-2.943+0.681*mag)
		width_inkm = 10.**(-0.543+0.261*mag)
		if length_inkm > 580:
			length_inkm = 580.
		if length_inkm < 6:
			length_inkm = 6.
		if width_inkm > 50.:
			width_inkm = 50.
		if width_inkm <6.5:
			width_inkm = 6.5


	return length_inkm * 1e3, width_inkm * 1e3

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()



####################################################


def extend_edge(f, opt, index):
	n = np.shape(f)
	if opt == 'left':
		for i in range(0,index-1):
			f[:,i] = f[:,index-1]
	elif opt== 'right':
		for i in range(index-1,n[1]):
			f[:,i] = f[:,index-2]
	elif opt=='down':
		for i in range(index-1,n[0]):
			f[i,:] = f[index-2,:]
	else:
		print('Only input allowable: right, left or down')
	return f 

def PlotFigure(SF,W,L,vmin=None,vmax=None,savefig=False,mymap=None,ihypox=None,ihypoz=None,fname=None):
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	import matplotlib.colors as mcolors

	# sample the colormaps that you want to use. Use 128 from each so we get 256
	# colors in total

	# colors1 = plt.cm.Reds(np.linspace(0., 1, 128))
	# colors2 = plt.cm.Blues_r(np.linspace(0, 1, 128))
	
	# # combine them and build a new colormap
	# colors = np.vstack((colors1, colors2))
	# mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

	if mymap is None:
		colors1 = plt.cm.Reds(np.linspace(0., 1, 128))
		colors2 = plt.cm.Blues_r(np.linspace(0, 1, 128))
	
		# combine them and build a new colormap
		colors = np.vstack((colors1, colors2))
		mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)



	lz,lx = np.shape(SF)
	dz = W/lz
	dx = L/lx

	if vmin is None:
		vmin = SF.min()
	if vmax is None:
		vmax = SF.max()
#	%%% set up axis
	zax = np.linspace(0,W,lz)
	xax = np.linspace(0,L,lx)

#	%%% plotting slip distribution
# OJO ponerlo por si en una se dice que NO	if nf == 'y':

	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'

	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['font.family'] = 'sans-serif'
	matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

	plt.figure(figsize=(10, 4))
	ax=plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

	X,Y=np.meshgrid(zax, xax)

#	im = plt.imshow(SF, interpolation='spline16', origin='upper',
	im = ax.imshow(SF, interpolation='bilinear', origin='upper',
#	im = plt.imshow(SF, interpolation='nearest', origin='upper', #Q
					cmap=mymap, extent=(-L/2,L/2,W,0),
					# cmap=mymap,
					# cmap='jet',
					vmax=vmax, vmin=vmin) #Q


	# ihypox,ihypoz = find_hypocenter(SF)
	if ihypox is not None:
		plt.scatter((ihypox)*dx-L/2,(ihypoz)*dz,100,'gold',marker='*',alpha=0.6,edgecolor='k')
	ax.set_ylabel('Depth [km]') #Q
	ax.set_xlabel('Along-strike Distance [km]') #Q
	# plt.title('Random Field Mean:{0:.2f} and std: {1:.2f}'.format(SF.mean(),np.std(SF,ddof=1))) #Q

# We can still add a colorbar for the image, too.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.1)
	CBI = plt.colorbar(im, orientation='vertical', shrink=1,cax=cax)
	CBI.set_label(fname.split('_')[0]) #Q

	# plt.show() #Q
	if savefig:
		plt.savefig(fname, dpi=300, transparent=False)  
	return

def trial():
	AB.example_driver()


def plot_model_geo():
	import matplotlib.pyplot as plt

	model = ['Wells1994','Wesnousky2008','Leonard2014_Interp-SS','Leonard2014_SCR-SS','Thingbaijam2017']
	c=['red','blue','gold','purple','green']
	plt.figure() #Q
	i=0
	for md in model:
		
		Lall=[]
		Wall=[]
		magall=np.arange(5.5,8.1,0.1)
		for mag in magall:
			L,W=fault_size(mag,model=md,printmodel=False)
			Lall.append(L)
			Wall.append(W)
	
		
		plt.plot(magall,np.array(Lall)/1e3,color=c[i],marker='',markersize=3,linestyle='-',label='L-'+md,alpha=0.8)
		plt.plot(magall,np.array(Wall)/1e3,color=c[i],marker='',markersize=3,linestyle=':',label='W-'+md,alpha=0.8)
		plt.gca().set_yscale('log')
		plt.gca().set_xlabel('Magnitude')
		plt.grid(True,which='both')

		i+=1

	plt.legend(loc='upper left',fontsize='small')
	plt.savefig('pyandrewsbarall/Geometry.pdf',dpi=300,transparent=False)
	return


# if __name__ == '__main__':

	# p=andrewsbarall(50,4)
	# p.scenario_from_mag(5.5) #mag is from 5.5 to 
	# plot_model_geo()










