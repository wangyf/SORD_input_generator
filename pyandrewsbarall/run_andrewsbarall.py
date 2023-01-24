import andrewsbarall_scenario as AB
import matplotlib.pylab as plt
import numpy as np

class andrewsbarall:
	"""docstring for andrewsbarall"""
	def __init__(self,dx,cutratio):
		self.dx =dx
		self.cutratio = cutratio

	def scenario_from_mag(self,mag,model='Wells1994'):

		self.modelname=model

		length,width = fault_size(mag,model)
		print('Rupture length is {0:5.0f} km\nRupture width is {1:5.0f} km'.format(length/1e3,width/1e3))
		
		self.tmpnx = np.ceil(length*2/self.dx)
		self.tmpnz = np.ceil(min(width*2,30e3)/self.dx)
	
		print('Nx = {0}, Nz = {1}'.format(self.tmpnx,self.tmpnz))
		# scenario_generator(nx=nx,nz=nz,dx=dx,cutratio=cutratio,depthcut=width*2,half_lengthcut=length)
		self.ts,self.tn,self.mus,self.mud=\
		scenario_generator_nonpower2(nx=self.tmpnx,nz=self.tmpnz,dx=self.dx,cutratio=self.cutratio,depthcut=width*2,half_lengthcut=length)
		self.nx = self.ts.shape[1]
		self.ny = self.ts.shape[0]
		return 

		

def scenario_generator_nonpower2(nx=2048,nz=2048/4,dx=50,cutratio=1,depthcut=24e3,half_lengthcut=60e3,asperity=False,hetero=True):
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

	w,shear,normal,mus,mud,ihypox,ihypoz=AB.initiate_scenario_nonpower2(nx,nz,asperity,hetero,dx,cutratio,depthcut,half_lengthcut,iseed)
	

	total_nx = int(nx) + 20
	total_nz = int(nz) + 10

	final_shear = np.zeros((total_nz,total_nx))
	final_normal = -1e20*np.ones((total_nz,total_nx))
	final_mus = 1e10*np.ones((total_nz,total_nx))
	final_mud = 1e10*np.ones((total_nz,total_nx))

	final_shear[0:-10,10:-10] = shear.T
	final_normal[0:-10,10:-10] = -normal.T
	final_mus[0:-10,10:-10] = mus.T
	final_mud[0:-10,10:-10] = mud.T

	print('nx={0}   ny={1}\n'.format(int(nx),int(nz)))
	print('ihypox={0}   ihypoy={1}\n'.format(ihypox,ihypoz))

	# ihypox,ihypoz = find_hypocenter(sd)
	# print(ihypox,ihypoz)

	# stressdrop = final_shear+final_normal*final_mud
	# # stressdrop[stressdrop<1e9]=np.nan
	# # PlotFigure((w).T,(w.shape[1]-1)*dx/1e3,  (w.shape[0]-1)*dx/1e3, -4,4,True,'w_nonpower2.pdf')
	# PlotFigure(final_shear,(final_shear.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'shearstress_nonpower2.pdf')
	# PlotFigure(final_normal,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'normalstress_nonpower2.pdf')
	# PlotFigure(-final_shear/final_normal,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'ratio_nonpower2.pdf')
	# PlotFigure(final_mus,(final_normal.shape[1]-1)*dx/1e3,  (final_shear.shape[0]-1)*dx/1e3, None,None,True,'mus_nonpower2.pdf')
	# PlotFigure(stressdrop,(w.shape[1]-1)*dx/1e3,(w.shape[0]-1)*dx/1e3, -10e6,10e6,True,'stressdrop_nonpower2.pdf')

	return final_shear,final_normal,final_mus,final_mud

def find_hypocenter(a):
	return np.unravel_index(a.argmax(), a.shape)
#################################################################

def scenario_generator(nx=2048,nz=2048/4,dx=50,cutratio=1,depthcut=24e3,half_lengthcut=60e3,asperity=False,hetero=True):
	# nx = 2048 #power of 2
	# dx = 50.
	# depthcut = 24e3 #m
	# half_lengthcut = 60e3 #m
	#     Make this flag false to select the high-stress model.
	#     Make this flag true to select the low-stress asperity model.
	# asperity = False
	# hetero=True

	# iseed=np.random.randint(1,1e10,4)
	iseed=[121231223,212313,32323,32324345345]

	w,s,sd,ihypox,ihypoz=AB.initiate_scenario(nx,nz,asperity,hetero,dx,cutratio,depthcut,half_lengthcut,iseed)

	dy = dx

# print(ihypox,ihypoz)
	PlotFigure((w).T,(w.shape[1]-1)*dy/1e3,  (w.shape[0]-1)*dx/1e3, -4,4,True,'w.pdf')
	# PlotFigure(s.T,(nz-1)*dx/1e3,  (nx-1)*dx/1e3, None,None,True,'shearstress.pdf')
	PlotFigure(sd.T,(w.shape[1]-1)*dy/1e3,(w.shape[0]-1)*dx/1e3, -10e6,10e6,True,'stressdrop.pdf')

	return

########## model prediction of rupture geometry
# only for strike-slip events
def fault_size(mag,model='Wesnousky2008'):
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
	else:
		print('Only input allowable: right or left')
	return f 

def PlotFigure(SF,W,L,vmin=None,vmax=None,savefig=False,fname=None):
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	import matplotlib.colors as mcolors

	# sample the colormaps that you want to use. Use 128 from each so we get 256
	# colors in total
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

	X,Y=np.meshgrid(zax, xax)
	plt.figure() #Q
#	im = plt.imshow(SF, interpolation='spline16', origin='upper',
	im = plt.imshow(SF, interpolation='bilinear', origin='upper',
#	im = plt.imshow(SF, interpolation='nearest', origin='upper', #Q
					# cmap=mymap, extent=(0,L,W,0),
					cmap=mymap,
					# cmap='jet',
					vmax=vmax, vmin=vmin) #Q


	ihypox,ihypoz = find_hypocenter(SF)
	plt.scatter((ihypoz-1),(ihypox-1),100,'gold',marker='*',alpha=0.6,edgecolor='k')
	plt.ylabel('Down Dip Distance [km]') #Q
	plt.title('Random Field Mean:{0:.2f} and std: {1:.2f}'.format(SF.mean(),np.std(SF,ddof=1))) #Q

# We can still add a colorbar for the image, too.
	CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q
	CBI.set_label('Amplitude') #Q

	# plt.show() #Q
	if savefig:
		plt.savefig(fname, dpi=300, transparent=False)  
	return

def trial():
	AB.example_driver()


if __name__ == '__main__':

	p=andrewsbarall(50,4)
	p.scenario_from_mag(5.5) #mag is from 5.5 to 










