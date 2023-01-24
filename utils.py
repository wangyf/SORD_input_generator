#!/usr/bin/env python
import sys
import numpy as np
from specrandom import *
import pyproj
from sordw3.extras import coord
import pandas
# from scipy import *
from scipy import interpolate
from numba import jit
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'


idtype = np.dtype( 'f8' ).str #'<f8'

def Shuoma_velstru(depth, SI=True):
	"""
	The following is a subroutine from Shuo Ma (written in Fortran, 
	but can easily be modified to the modeller’s preferred code) to 
	create a 1D layered velocity model to use in the simulations. 
	After extraction, set the minimum shear wave velocity (Vs) in 
	your simulation so that Vs30 = 760 m/s. Since Vs30 is calculated 
	from the time-averaged shear velocity from 0-30 meters depth and 
	since many modellers are using grid points spacing on the order 
	of ~25-50 m, the best way to achieve uniformity between modellers 
	is to clamp the top node shear velocity to 760 m/s. Do not adjust 
	Vp or rho.

	Input:
		depth: in km positive below surface
	Output (SI unit):
		rou0: density kg/m^3
		vp0:  vp      m/s
		vs0:  vs      m/s
	"""
	if SI:
		depth = depth/1e3

	zandrews = depth + 0.0073215   #in km
	if zandrews < 0.03:
		vs0 = 2.206 * zandrews ** 0.272
	elif zandrews < 0.19:
		vs0 = 3.542 * zandrews ** 0.407
	elif zandrews<4.:
		vs0 = 2.505 * zandrews ** 0.199
	elif zandrews<8.:
		vs0 = 2.927 * zandrews ** 0.086
	else:
		vs0 = 2.927 * 8. ** 0.086

	vp0=max(1.4+1.14*vs0,1.68*vs0)
	rou0=2.4405+0.10271*vs0

	rou0=rou0*1000. # convert g/cm^3 to kg/m^3
	vp0=vp0*1000.   # convert km/s to m/s
	vs0=vs0*1000.

	return rou0,vp0,vs0



def gammaH(rho,ddepth):

	g = 9.81
	num = rho.shape[0]
	gammaHint = np.empty(num)

	gammaHint[0] = g*ddepth*rho[0]/2
	for i in range(1,num):
		gammaHint[i] = gammaHint[i-1] + g*ddepth*(rho[i]+rho[i-1])/2
	return gammaHint

def plasticyieldsurface(stensor,phi,co):
	s11 = stensor[0]
	s22 = stensor[1]
	s33 = stensor[2]
	s23 = stensor[3]
	s31 = stensor[4]
	s12 = stensor[5]

	ms = (s11 + s22 + s33)/3
	ds11 = s11 - ms
	ds22 = s22 - ms
	ds33 = s33 - ms
	ds12 = s12
	ds23 = s23
	ds31 = s31

	sbar = np.sqrt((ds11*ds11 + ds22*ds22 + ds33*ds33 +\
		           2*ds12*ds12 + 2*ds23*ds23 + 2*ds31*ds31)/2)

	sy = co*np.cos(np.arctan(phi)) - np.sin(np.arctan(phi))*ms

	return sbar,sy


def cohesionfromcloseness(stensor,phi,closeness):
	s11 = stensor[0]
	s22 = stensor[1]
	s33 = stensor[2]
	s23 = stensor[3]
	s31 = stensor[4]
	s12 = stensor[5]

	ms = (s11 + s22 + s33)/3
	ds11 = s11 - ms
	ds22 = s22 - ms
	ds33 = s33 - ms
	ds12 = s12
	ds23 = s23
	ds31 = s31	

	sbar = np.sqrt((ds11*ds11 + ds22*ds22 + ds33*ds33 +\
		           2*ds12*ds12 + 2*ds23*ds23 + 2*ds31*ds31)/2)

	if sbar<1e-3:
		co = 1e10
	else:
		co = (sbar/closeness + np.sin(np.arctan(phi))*ms)/np.cos(np.arctan(phi))

	if co < 0:
		print('stress^bar/mean_stress={0}'.format(-sbar/ms))
		print('Cohesion cannot be negative. Try reduce closeness or phi')
		sys.exit()
	else:
		return co

def Init_coordinate(izone=11, irotate=None,iorigin=(-116.5, 34.5)):
	proj = pyproj.Proj(proj='utm', zone=izone, ellps='WGS84')
	if irotate is not None:
		proj = coord.Transform(proj, rotate=irotate, origin=iorigin)
	return proj

def geographic_angle_conversion(strike=None, proj=None, lon=-116.5,lat=34.5):
	rot = coord.rotation(lon, lat, proj)[1]
	strike = strike + rot
	strike=np.where(strike<0.,strike+360.,strike)
	return strike

def coordinate_conversion(lon=None,lat=None,leftcorner=None,proj=None,inverse=False):
	#leftcorner is origin in the new coordinate system after rotation
	if lon is not None and lat is not None:
		ux,uz=proj(lon,lat,inverse=inverse)
		ux = ux - leftcorner[0]
		uz = uz - leftcorner[1]
	return ux,uz

def staindexsearch(p1,p2,p3,p4,sta,method=None):

	if method == 'crude':
		m_i = 0.5
		n_i = 0.5
		return m_i,n_i
	elif method == 'fine':

		eps = 1e-3
		"""
		 y
		 ^
		 p4 (01)------ p3
		 |			|
		 |   sta	|
		 |			|
		 p1 ------ p2 (10)
					-> x
		"""
		# print(p1,p2,p3,p4)
	
		x00 = p1[0]
		x01 = p4[0]
		x10 = p2[0]
		x11 = p3[0]
		
		y00 = p1[1]
		y01 = p4[1]
		y10 = p2[1]
		y11 = p3[1]
	
		xmn = sta[0]
		ymn = sta[1]
	
		a1 = xmn-x00 
		a2 = x10-x00
		a3 = x01-x00
		a4 = x11-x01-x10+x00
		
		b1 = ymn-y00
		b2 = y10-y00
		b3 = y01-y00
		b4 = y11-y10-y01+y00
		
		# print('a1,a2,a3,a4:',a1,a2,a3,a4 )
		# print('a1,a2,a3,a4:',a1,a2,a3,a4 )
		if np.abs(a4)<eps:
			if np.abs(b4)<eps:
				n_i = (b2*a1-b1*a2)/(a3*b2-b3*a2)
				m_i = (a1-a3*n_i)/a2
			if np.abs(a3)<eps and np.abs(b4)>eps:
				n_i = (b1*a2-b2*a1)/(b3*a2+b4*a1)
				m_i = a1/a2
			if np.abs(b4)>eps and np.abs(a3)>eps:
				A = b4*a3
				B = a3*b2-b3*a2-b4*a1
				C = b1*a2-b2*a1
	
				d = (B**2) - (4*A*C)
				if d < 0:
					print('No real solution')
					exit()

				root1 = (-B - np.sqrt(d)) / (2 * A)
				root2 = (-B + np.sqrt(d)) / (2 * A)

				if root1 > 0 and root1 < 1:
					n_i = root1
				elif root2 > 0 and root2 < 1:
					n_i = root2
				else:
					print('No usable root found')
					exit()
				m_i = (a1-a3*n_i)/a2
	
		if np.abs(b4)<eps and np.abs(a4)>eps:
			if np.abs(b2)<eps:
				m_i = (a1*b3-a3*b1)/(a2*b3+a4*b1)
				n_i = (b1-b2*m_i)/b3
	
			if np.abs(b2)>eps:
				A = a4*b2
				B = a3*b2-b3*a2-a4*b1
				C = a1*b3-a3*b1
				
				d = (B**2) - (4*A*C)
				if d < 0:
					print('No real solution')
					exit()

				root1 = (-B - np.sqrt(d)) / (2 * A)
				root2 = (-B + np.sqrt(d)) / (2 * A)	

				if root1 > 0 and root1 < 1:
					m_i = root1
				elif root2 > 0 and root2 < 1:
					m_i = root2
				else:
					print('No usable root found')
					exit()	
				n_i = (b1-b2*m_i)/b3		
				
		if np.abs(a4)>eps and np.abs(b4)>eps:
			# a4!=0 
			A = (b2*a4-b4*a2)
			B = -(a2*b3-a3*b2+b1*a4-b4*a1)
			C = a1*b3-a3*b1
		
		
			d = (B**2) - (4*A*C)
			if d < 0:
				print('No real solution')
				exit()
	
			root1 = (-B - np.sqrt(d)) / (2 * A)
			root2 = (-B + np.sqrt(d)) / (2 * A)
			
			if root1 > 0 and root1 < 1:
				m_i = root1
			elif root2 > 0 and root2 < 1:
				m_i = root2
			else:
				print('No usable root found')
				exit()
			
			n_i = ((b1*a4-b4*a1)-(b2*a4-b4*a2)*m_i)/(b3*a4-b4*a3)
	
		return m_i, n_i #m_i,n_i between 0,1


## recommend to use this function
@jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside



def point_inside_polygon(x, y, poly, include_edges=True):
	'''
	Test if point (x,y) is inside polygon poly.

	poly is N-vertices polygon defined as 
	[(x1,y1),...,(xN,yN)] or [(x1,y1),...,(xN,yN),(x1,y1)]
	(function works fine in both cases)

	Geometrical idea: point is inside polygon if horisontal beam
	to the right from point crosses polygon even number of times. 
	Works fine for non-convex polygons.
	'''
	n = len(poly)
	inside = False

	p1x, p1y = poly[0]
	for i in range(1, n + 1):
		p2x, p2y = poly[i % n]
		if p1y == p2y:
			if y == p1y:
				if min(p1x, p2x) <= x <= max(p1x, p2x):
					# point is on horisontal edge
					inside = include_edges
					break
				elif x < min(p1x, p2x):  # point is to the left from current edge
					inside = not inside
		else:  # p1y!= p2y
			if min(p1y, p2y) <= y <= max(p1y, p2y):
				xinters = (y - p1y) * (p2x - p1x) / float(p2y - p1y) + p1x

				if x == xinters:  # point is right on the edge
					inside = include_edges
					break

				if x < xinters:  # point is to the left from current edge
					inside = not inside

		p1x, p1y = p2x, p2y

	return inside

def included_angle(angle1=None,angle2=None):
	if np.shape(angle1) != np.shape(angle2):
		angle2 = angle2.transpose()
	angle = angle1-angle2
	angle = np.where(angle>0,angle,-angle)
	angle = np.where(angle<90,angle,180.-angle)
	return angle

def plane_rotate(sig11,sig22,sig12,rot):
	# rot is include angle between axis-x and 
	# axis-1(max compressive component)
	rot = rot*np.pi/180.
	sig11p = (sig11 + sig22)/2. + \
			 (sig11 - sig22)/2.*np.cos(2*rot)+\
			 sig12*np.sin(2*rot)
	# sig11*np.cos(rot)**2+\
	#		  sig22*np.sin(rot)**2+\
	#		  2*np.sin(rot)*np.cos(rot)*sig12
	sig22p = (sig11 + sig22)/2. - \
			 (sig11 - sig22)/2.*np.cos(2*rot)-\
			 sig12*np.sin(2*rot)
			#  sig11*np.sin(rot)**2+\
	#		  sig22*np.cos(rot)**2-\
	#		  2*np.sin(rot)*np.cos(rot)*sig12
	sig12p = -(sig11-sig22)/2.*np.sin(2*rot) + \
			 sig12*np.cos(2*rot)
			 # -np.sin(rot)*np.cos(rot)*sig11 + \
	   #		np.sin(rot)*np.cos(rot)*sig22 + \
	   #		np.cos(2*rot)*sig12
	return sig11p,sig22p,sig12p

def fill_halo(f, opt, index, val):
	n = np.shape(f)
	if opt == 'left':
		for i in range(0,index-1):
			f[:,i] = val
	elif opt== 'right':
		for i in range(index-1,n[1]):
			f[:,i] = val
	elif opt == 'down':
		for j in range(index-1,n[0]):
			f[j,:] = val
	elif opt == 'up':
		for j in range(0,index-1):
			f[j,:] = val
	else:
		print('Only input allowable: right, left, up and down')
	return f 

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

def writebin(file,matrix,outdtype=idtype):
	fd = open( file ,'wb')
	# if dtype==None: dtype=np.dtype( 'f8' ).str #'<f4'
	print('Write size check',file,matrix.shape)
	matrix.astype(outdtype).tofile( fd )
	fd.close()
	return

def readbin(file,shape,inputdtype=None):
	fd = open( file ,'rb')
	if inputdtype == None:
		inputdtype = idtype
	matrix = np.fromfile(fd, inputdtype).reshape(shape).astype(idtype)
	print('Read size check',file,matrix.shape)
	fd.close()
	return matrix

def n2c(a):
	n0 = a.shape[0]
	n1 = a.shape[1]
	return (a[0:-1,0:-1] +   a[1:n0+1,0:-1] +\
			       a[0:-1,1:n1+1] + a[1:n0+1,1:n1+1])/4


def point2line(p1,p2,x_axis):
	x = [p1[0],p2[0]]
	y = [p1[1],p2[1]]

	coefficients = np.polyfit(x, y, 1)

	polynomial = np.poly1d(coefficients)

	y_axis = polynomial(x_axis)

	return y_axis

def gauss_kern(size, sizey=None):
	""" Returns a normalized 2D gauss kernel array for convolutions """
	size = int(size)
	if not sizey:
		sizey = size
	else:
		sizey = int(sizey)
	x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
	g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
	return g / g.sum()

def smooth2D_old(im, n, ny=None) :
	""" blurs the image by convolving with a gaussian kernel of typical
		size n. The optional keyword argument ny allows for a different
		size in the y direction.
	"""
	g = gauss_kern(n, sizey=ny)
	improc = signal.convolve(im,g, mode='valid')
	return(improc)

def blur2D(a,sigma=10):
	b = gaussian_filter(a, sigma=sigma)
	return b

def smooth1D(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.

	output:
		the smoothed signal
		
	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	see also: 
	
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
 
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")


	if window_len<3:
		return x


	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	# s = x
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')

	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[(int(window_len/2)-1):-(int(window_len/2))]

def taper2D(size,taper,samp,inverse=False, dtype=idtype):
	nx = int(size[1]/samp[1]+1.5)
	ny = int(size[0]/samp[0]+1.5)

	if len(taper) == 4:
		twdpt = taper[3]		# additional taper with depth
		twkm = taper[0:3]		# set taper windows to given values 
	elif len(taper) == 3:
		twkm = taper[0:3]		# set taper windows to given values 

	if len(samp) == 2: 
		ntx = round(twkm[0]/samp[1]) 
		ntt = round(twkm[1]/samp[0])
		ntb = round(twkm[2]/samp[0])
		print('Side   taper node ntx {0}'.format(ntx))
		print('Top	taper node ntx {0}'.format(ntt))
		print('Bottom taper node ntx {0}'.format(ntb))
		tapslp = [ntx, ntt, ntb] 	# defines the # of rows/cols of the
									## output slip distribution to be tapered
	G = np.ones((ny,nx),dtype=idtype)
	G = Taperfunction(G,tapslp) 

	if 'twdpt' in locals():
		print(' Additional depth taper applied: Sz**{0}'.format(twdpt))
		i1,j1=np.shape(G)
		w1 = np.linspace(1,i1*samp[0],i1)
		w = np.transpose(w1**twdpt)
		w = w/max(w)
		for i in range(0,j1):
			G[:,i]=G[:,i]*w	

	if inverse: G = 1 - G
	return G

def loadBBP(file=None, format=None):

	# all as string table
	# table = np.genfromtxt(file, delimiter=' ', dtype=idtype, skip_header=1)
	#table = np.loadtxt(file, dtype=idtype, skiprows=1)
	if format == 'fault':
		# table = np.genfromtxt(file,delimiter=',', dtype='f8',autostrip=True,comments="#")
		table = pandas.read_csv(file,sep=',',header=None,skiprows=1,\
			names=['index','length','width', 'strike','dip','hypo_along_stk','hypo_down_dip'],skipinitialspace=True)
	elif format == 'station':
		table = pandas.read_csv(file,sep=',',header=None,skiprows=1,\
			names=['index','name', 'lon', 'lat', 'rjb','vs30'],skipinitialspace=True)
		# table = np.genfromtxt(file,delimiter=',', dtype='U10',autostrip=True,comments="#")
	elif format == 'proj':
		table = pandas.read_csv(file,sep=',',header=None,skiprows=1,\
			names=['proj','zone', 'ellps', 'rotate', 'origin-lon','origin-lat','leftc1','leftc2'],\
			skipinitialspace=True)
	elif format == 'velmodel':
		table = pandas.read_csv(file,sep=' ',header=None,skiprows=1,\
			names=['thickness','vp', 'vs', 'rho', 'qp','qs'],\
			skipinitialspace=True)
		table['thickness'] = table['thickness'] * 1e3
		table['vp'] = table['vp'] * 1e3
		table['vs'] = table['vs'] * 1e3
		table['rho'] = table['rho'] * 1e3
		# print((table))
		# print(table['zone'][0])

		# table = np.genfromtxt(file,delimiter=', ', dtype="U10", autostrip=True, comments="#")
	else:
		print('Format is not included so far')
	#np.genfromtxt('data.txt', delimiter=',', dtype=None, \
	#	names=('sepal length', 'sepal width', 'petal length', 'petal width', 'label'))
	
	return table

def table2numpy(table,namelist):
	return	table[namelist].to_numpy()


def PlotFigure(*args):

	ndim=args[0]
	SF=args[1]

	if ndim == 2:
		W=args[2]
		L=args[3]
		name=args[4]
		if len(args)==6:
			linseg=args[5]
		if len(args)==8:
			linseg=args[5]
			vmin=args[6]
			vmax=args[7]

		lz,lx = np.shape(SF)
		dz = W/lz
		dx = L/lx

#	%%% set up axis
		zax = np.linspace(0,W,lz)
		xax = np.linspace(0,L,lx)

#	%%% plotting slip distribution
# OJO ponerlo por si en una se dice que NO	if nf == 'y':


		X,Y=np.meshgrid(zax, xax)
		plt.figure() #Q
		if len(args) < 8:
			im = plt.imshow(SF, interpolation=None, origin='upper',\
#	im = plt.imshow(SF, interpolation='spline16', origin='upper',
	# im = plt.imshow(SF, interpolation='bilinear', origin='upper',
#	im = plt.imshow(SF, interpolation='nearest', origin='upper', #Q
					cmap=plt.get_cmap('rainbow'), extent=(0,L,W,0),\
					vmax=(SF).max(), vmin=(SF).min()) #Q
			im.set_rasterized(True)
		else:
			import matplotlib.colors as mcolors

			# sample the colormaps that you want to use. Use 128 from each so we get 256
			# colors in total
			colors1 = plt.cm.Reds(np.linspace(0., 1, 128))
			colors2 = plt.cm.Blues_r(np.linspace(0, 1, 128))
	
			#	 combine them and build a new colormap
			colors = np.vstack((colors1, colors2))
			mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

			im = plt.imshow(SF, interpolation=None, origin='upper',\
					cmap=mymap, extent=(0,L,W,0),\
					vmax=vmax, vmin=vmin) #Q
			im.set_rasterized(True)
		plt.ylabel('Down Dip Distance [km]') #Q
		plt.title('Random Field Mean:{0:.5f} and std: {1:.5f}\nMax: {2:.5f} Min: {3:.5f}'.\
					  format(SF.mean(),np.std(SF,ddof=1),SF.max(),SF.min())) #Q

# We can still add a colorbar for the image, too.
		CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q
		CBI.set_label('Amplitude') #Q

		if len(args)>5:
			for i in range(len(linseg)):
				plt.plot([(linseg[i]-1)*dx,(linseg[i]-1)*dx],[0,W],'k--',lw=0.5)

	# plt.show() #Q
		plt.savefig(name, dpi=300, transparent=True,format='pdf')
		# plt.savefig(name.replace('.png','.pdf'),format='pdf', dpi=100, transparent=True) 
		plt.close() 
	elif ndim == 1:
		# print(SF.shape)
		W=args[2]
		name=args[3]

		if len(args)>4:
			y1 = args[4]
			y2 = args[5]
		lz = len(SF)
		dz = W/lz

		zax = np.linspace(0,W,lz)
		
		plt.figure() #Q

		if SF.ndim == 2:
			for i in range(SF.shape[1]):
				plt.plot(SF[:,i],zax)
		elif SF.ndim == 1:
			plt.plot(SF,zax)

		plt.gca().invert_yaxis()
		if len(args)>4:
			plt.gca().set_xlim([y1,y2])
		# plt.gca().set_xscale('log')
		plt.savefig(name, dpi=300, transparent=True,format='pdf')  
		# plt.savefig(name.replace('.png','.pdf'),format='pdf', dpi=100, transparent=True) 
		plt.close()
	return


def sord_sw_scenario(rundir='tmp',np3=(32,14,16),nn=(1728,551,1102),T=15,ihypo=(688,263,551),rcrit=5e3,_ot = 50,_ds=10):
	import numpy, sordw3, sys

	# computational discretization
	dx = 50, 50, 50
	dt = dx[0] / 12500.0
	nt = int( T / dt + 1.5 )
	npml = 50
	faultnormal = 3
	slipvector = (1.0, 0.0, 0.0)

	# boundary conditions
	bc1 = 10, 0,  10 
	bc2 = 10, 10, 10

	# material properties
	hourglass = 1.0, 2.0
	eplasticity = 'plastic'
	plmodel='DP1'
	tv = 10 * dt
	
	fieldio = [
	   ( '=R', 'rho', [ihypo[0]+0.5,(),ihypo[2]+0.5], 'rho.bin'  ),
	   ( '=R', 'vp',  [ihypo[0]+0.5,(),ihypo[2]+0.5], 'vp.bin'  ),
	   ( '=R', 'vs',  [ihypo[0]+0.5,(),ihypo[2]+0.5], 'vs.bin'  ),
	   ( '=', 'gam', [], 0.1    ),
   	   ( '=R', 'mco', [(),(),ihypo[2]+0.5], 'pco.bin'),
   	   ( '=R', 'phi', [(),(),ihypo[2]+0.5], 'phi.bin'),
	]

	# initial volume stress input
	ivols = 'yes'
	fieldio += [
	   ( '=R', 'a33', [(),(),ihypo[2]+0.5],          'sigma_zz.bin'),
	   ( '=R', 'a22', [(),(),ihypo[2]+0.5],          'sigma_yy.bin'),
	   ( '=R', 'a11', [(),(),ihypo[2]+0.5],          'sigma_xx.bin'),      
	   ( '=R', 'a31', [(),(),ihypo[2]+0.5],          'sigma_xz.bin' ),
	]

	friction = 'slipweakening'
	fieldio += [
	
	   ( '=r' ,'dc',  [],  'dc.bin' ),
	   ( '=r', 'mus', [],  'mus.bin'),
	   ( '=r', 'mud', [],  'mud.bin'),
	   ( '=r', 'co',  [],  'fco.bin'),
	
	]

	svtol = 0.1
	# Nucleation
	vrup = 2e3
	# rcrit = 5e3
	rrelax = 8e2

	fieldio += [
	  ( '=w', 'tsm',[(),(),ihypo[2],(1,-1, _ot)], 'tsm.out'),
	  ( '=w', 'svm',[(),(),ihypo[2],(1,-1, _ot)], 'svm.out'),
	  ( '=w', 'tsm',[(),(),ihypo[2], 2], 'tss.out'),
	  ( '=w', 'tsm',[(),(),ihypo[2],-1], 'tse.out'),
	  ( '=w', 'x1', [(),(),ihypo[2]], 'fx.out'),
	  ( '=w', 'x2', [(),(),ihypo[2]], 'fy.out'),
	  ( '=w', 'x3', [(),(),ihypo[2]], 'fz.out'),
	  ( '=w', 'nhat1',[(),(),ihypo[2]], 'nhat1.out'),
	  ( '=w', 'nhat2',[(),(),ihypo[2]], 'nhat2.out'),
	  ( '=w', 'nhat3',[(),(),ihypo[2]], 'nhat3.out'),
	  ( '=w', 'sum', [(),(),ihypo[2],-1], 'sum.out'),
	  ( '=w', 'trup',[(),(),ihypo[2],-1], 'trup.out'),
	  ( '=w', 'x1',  [(),1,(ihypo[2]-20,ihypo[2]+21)], 'fd_x1'  ),
	  ( '=w', 'x2',  [(),1,(ihypo[2]-20,ihypo[2]+21)], 'fd_x2'  ),
	  ( '=w', 'x3',  [(),1,(ihypo[2]-20,ihypo[2]+21)], 'fd_x3'  ),
	  ( '=w', 'u1',  [(),1,(ihypo[2]-20,ihypo[2]+21),-1], 'fd_u1'  ),
	  ( '=w', 'u2',  [(),1,(ihypo[2]-20,ihypo[2]+21),-1], 'fd_u2'  ),
	  ( '=w', 'u3',  [(),1,(ihypo[2]-20,ihypo[2]+21),-1], 'fd_u3'  ),
	  ( '=w', 'x1',  [(1,-1,_ds),1,(ihypo[2]-int(20e3/dx[2]),ihypo[2],_ds)], 'm_x1'  ),
	  ( '=w', 'x3',  [(1,-1,_ds),1,(ihypo[2]-int(20e3/dx[2]),ihypo[2],_ds)], 'm_x3'  ),
	  ( '=w', 'x1',  [(1,-1,_ds),1,(ihypo[2]+1,+int(20e3/dx[2])+ihypo[2]+1,_ds)], 'p_x1'  ),
	  ( '=w', 'x3',  [(1,-1,_ds),1,(ihypo[2]+1,+int(20e3/dx[2])+ihypo[2]+1,_ds)], 'p_x3'  ),
	  ( '=w', 'a1',  [(1,-1,_ds),1,(ihypo[2]-int(20e3/dx[2]),ihypo[2],_ds)], 'm_a1'  ),
	  ( '=w', 'a3',  [(1,-1,_ds),1,(ihypo[2]-int(20e3/dx[2]),ihypo[2],_ds)], 'm_a3'  ),
	  ( '=w', 'a1',  [(1,-1,_ds),1,(ihypo[2]+1,+int(20e3/dx[2])+ihypo[2]+1,_ds)], 'p_a1'  ),
	  ( '=w', 'a3',  [(1,-1,_ds),1,(ihypo[2]+1,+int(20e3/dx[2])+ihypo[2]+1,_ds)], 'p_a3'  ),
	]

	sordw3.run( locals() )
	return 

def copy_binary(rundir,bindir):
	import os
	cmd="cp "+ bindir+'/* '+rundir+'/in'
	print(cmd)
	return os.system(cmd)

def submit_queue(rundir):
	import os
	cmd="sh "+rundir+'/queue.sh'
	print(cmd)
	return os.system(cmd)

