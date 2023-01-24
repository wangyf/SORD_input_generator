#!/usr/bin/env python

import scipy.special
import datetime
import time
import math
import numpy as np
# from numpy.fft import fft2, ifft2
from pyfftw.interfaces.numpy_fft import fft2, ifft2
import multiprocessing
from numpy.linalg import lstsq

################# uitl ###################
def sgival(*args):
	B=args[0]
	val1=args[1]
	val2=args[2]
	r,c=np.shape(B)
	i=0
	while i < r:
		j=0
		while j < c:
			if B[i,j]<=val1:
				B[i,j]=val2
			else:
				B[i,j]=B[i,j]
			j+=1
		i+=1
	B=B.reshape(np.shape(B))


# Created by Hugo Cruz Jimenez, August 2011, KAUST
# Hanna Window function
def hann(n):
	n = int(n)
	han=np.zeros(n)
	if np.mod(n,2)==0:
		m=int(n/2)
	else:
		m=int((n+1)/2)
	i=1
	while i<=m:
		han[i-1]=0.5*(1-np.cos(2*np.pi*i/(n+1)))
		i+=1
	i=m+1
	while i<=n:
		han[i-1]=han[n-i]
		i+=1
	return han

def PlotFigure(*args):
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	SF=args[0]
	W=args[1]
	L=args[2]

	lz,lx = np.shape(SF)
	dz = W/lz
	dx = L/lx

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
					cmap=cm.jet, extent=(0,L,W,0),
					vmax=(SF).max(), vmin=(SF).min()) #Q


	plt.ylabel('Down Dip Distance [km]') #Q
	plt.title('Random Field Mean:{0:.2f} and std: {1:.2f}'.format(SF.mean(),np.std(SF,ddof=1))) #Q

# We can still add a colorbar for the image, too.
	CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q
	CBI.set_label('Amplitude') #Q

	# plt.show() #Q
	plt.savefig('field.pdf', dpi=300, transparent=False)  
	return

############################################








def Taperfunction(*args):

#function [S] = TaperSlip(S,N,window)
#
# function [S] = TaperSlip(S,N,'window')
# tapers the slip amplitudes at the fault boundaries 
# to avoid large slip at the edges. 
#	
# INPUT:
# S	- original grid
# N 	- vector of numbers of rows/cols to be tapered [left/right top bottom]
# window- 'hn' for Hanning window 
#	  'kw' for Kaiser window
#	  'tr' for triangular window
# 	  window can also be a vector [c1 c2 c3] with c(i) < 1; 
#	  in this case, the corresponding rows/cols will simply be 
#	  multiplied by that c(i)
#	  NOTE: this scheme only works if only one col/row is tapered
#
# Default values: N = [1 0 1], window = 'hn'
#
# OUTPUT: 
# S 	- tapered grid (same size as input grid)
#
# See also MATLAB function HANNING, and KAISER
	
# Written by Martin Mai (mmai@pangea.Stanford.EDU) 
# 05/08/98
# last change 08/26/99
# ------------------------------------------------
	
	if len(args) == 1:
	   S=args[0]
	   N=[1, 0, 1]
	   window = 'hn'
	elif len(args) == 2:	
	   S=args[0]
	   N=args[1]
	   window = 'hn'
	elif len(args) == 3:	
	   S=args[0]
	   N=args[1]
	   window = args[2]
	
	#%% create taper window, i.e. Kaiser window (NOTE: increasing beta in 
	#%% Kaiser-Window widens main lobe and attenuates amplitude in the side lobes)
	#%% or hanning (cosine) from 0 to 1 in N steps on EACH side
	if type(window) is str:
		if window == 'hn':
			taperS = hann(2*N[0]+1) # for left/right columns
			taperT = hann(2*N[1]+1) # for top row
			taperB = hann(2*N[2]+1) # for bottom rows
		elif window == 'kw':
			beta = 6
			taperS = np.kaiser(2*N[0]+1,beta)
			if N[1]==0:
	 			taperT = [1]
			else:
	 			taperT = np.kaiser(2*N[1]+1,beta)
			taperB = np.kaiser(2*N[2]+1,beta)
		elif window == 'tr':
			taperS = ss.triang(2*N[0]+1)	
			taperT = ss.triang(2*N[1]+1)	
			taperB = ss.triang(2*N[2]+1)	
		mm=len(taperB)
		winS=taperS[int(N[0]+1): int(2*N[0]+1) ]
#		print winS
		winT=taperT[0:int(N[1])]
#		print winT
		winB=taperB[mm-(int(N[2])-1)-1:mm]
#		print winB
	
	elif type(window) is not str:

#	  i,j = find(N == 0)  # to make sure that rows/cols with N = 0 are not
		j=0
		while j < len(N):
			if N[j]==0:
				window[j]=1
			j += 1 

#	  [i,j] = find(N == 0)  # to make sure that rows/cols with N = 0 are not
#	  window[j] = 1	   # tapered in case s contains entries other than 1	

		winS = window[0]
		winT = window[1]
		winB = window[2]
	
	#%% check in case no taper is applied at one of the boundaries
	if len(winS) == 0: winS = 1
	if len(winT) == 0: winT = 1
	if len(winB) == 0: winB = 1
	
	 
	#%% set-up bell-shaped tapering function 
	bell = np.ones((np.shape(S))) 
	j,k = np.shape(S)

#	print 'tipo de winS',type(winS)
	# print 'winS=',winS
	# print 'winT=',winT
	# print 'winB=',winB
	try:
		ls = len((winS)) 
	except:
		ls = 0
	try:	
		lt = len((winT))
	except:
		lt = 0
	try:
		lb = len((winB))
	except:
		lb = 0


	for zs in range(0,j):
		bell[zs,0:ls] = bell[zs,0:ls]*winS[::-1]
#		print 'abcd',zs,k-ls, k
#		print 'abcd',zs,k-ls,k
		bell[zs,k-ls:k] = bell[zs,k-ls:k]*winS

	for zt in range(0,k):
		bell[0:lt,zt-1] = bell[0:lt,zt-1]*winT
		bell[j-lb:j,zt-1] = bell[j-lb:j,zt-1]*winB
	
	#%% finally, multiply input slip with bell-function
	S = S*bell
	return S
###################################################################











def SpecSyn2(*args):
	#function [Y,spar,spec,ierr] = SpecSyn2(N,samp,corr,acf,Rseed)
	#  [Y,spar,spec,ierr] = SpecSyn2(N,samp,corr,'acf',Rseed) 
	#  generates a 2D-random field Y of size (Nz+1 x Nx+1) with
	#  possibly variable spatial sampling in z,x-direction. 
	#  This function simulates anisotropic random fields, i.e. 
	#  the correlation len in both directions can be different 
	#  rectangular grid dimensions are also possible.
	#  corr contains the corr. len ax, az and the Hurstnumber H 
	#  or the fractal dimension D the autocorrelation function to 
	#  be used has to be specified by acf
	#  NOTE: D = 3-H, and larger D (smaller H) yield "rougher" fields
	#
	#  The algorithm is based on the spectral synthesis method by 
	#  Pardo-Iguzquiza, E. and Chica-Olma, M. (1993)
	#  The Fourier integral method: and efficient spectral method for 
	#  simulation of random fields, Mathematical Geology, 25, p177-217.
	#  but extends their method to handle rectangular grids and variable
	#  sampling in the two directions.
	#
	#  INPUT:
	#  N	 - grid dimensions [Nz Nx]
	#  samp   - desired sampling, [dz dx] if scalar, then dz = dx
	#  corr  - corr = [az ax]   for 'gs' and 'ex' 
	#	 corr = [az ax H] for 'ak' note that 0 <= H <= 1
	#	 corr = [D kc]	for 'fr' D is the fractal dimension,
	#	  kc: corner wavenumber, spectrum decays linearly for k>kc
	#  acf   - autocorrelation function: 
	#	'gs' or 'GS' - Gaussian
	#	'ex' or 'EX' - Exponential
	#	'ak' or 'AK' - anisotropic vonKarman
	#	'fr' or 'FR' - fractal distribution
	#  Rseed - seeds for random number generators if omitted or empty
	#	 Rseed = sum(100*clock) is used (returned in structure spar)
	#	 [Rpseed Rsseed] for the phase and small random spectral part
	# 
	#  OUTPUT:
	#  Y	- Y = [z x] random field whose size is determined by the sample
	#	spacing in each direction, the number of points and whether
	#	the pow2-option is given or not. 
	#  spar  - structure with len vectors, sampling in both direction
	#	and other relevant parameters it also contains the random
	#	seed number, useful to reproduce realizations
	#  spec - structure containing the computed power spectrum as well
	#		 wavenumber vectors
	#  ierr - 0 when successfully executed 
	#		 1 when error in Z,X sampling
	#
	#  Written by Martin Mai (martin@seismo.ifg.ethz.ch) 
	#  originally from 07/16/98, based on SRB-toolbox (Ph. Rio)
	#  last changes 03/01/2000 Nov. 2002;
	# ------------------------------------------------
	
	ierr  = 0			  #% error variable
	check = 'n'  	#% set to 'y' if you want to create
	  		#% a simple out put plot to check the
	  		#% the spectra and the resulting field
	t = time.process_time()

	#%% check input variables
	if len(args) < 4: print('  Error *** Not enough input arguments ***')
	elif len(args) == 4:
		N=args[0] 
		samp=args[1] 
		corr=args[2] 
		acf=args[3] 
		Rseed = {} 
	elif len(args) == 5:
		N=args[0] 
		samp=args[1] 
		corr=args[2] 
		acf=args[3] 
		Rseed =args[4]
	
	if len(samp) == 1: samp = [samp, samp]
	if len(N) == 1: N = [N, N]
	

	#%% error checking on inpur array size and given sampling
	if np.abs(N[1]-int(N[1]/samp[1])*samp[1])>1e-6:
		ierr = 1
		print('** sampling in X does not yield an integer number **')
		print('   of grid points ==> abort!')
		print('==> BOOM OUT in SpecSyn2<==')
		return

	if np.abs(N[0]-int(N[0]/samp[0])*samp[0])>1e-6:
		ierr = 1
		print('** sampling in Z does not yield an integer number **')
		print('   of grid points ==> abort!')
		print('==> BOOM OUT in SpecSyn2<==')
		return
	
	
	#%% get data values on the correlation len/fractal dimension
	if  acf == 'fr' or acf == 'FR':
		if len(corr) == 2:
			D = corr[0]; kc = corr[1]
		else:
			D = corr[0]; kc = 0.1 
			print('** Corner wavenumber kc not given: set to 0.1 **')
	elif acf == 'ex' or acf == 'EX' or acf == 'gs' or acf == 'GS':
		ax = corr[1]; az = corr[0] 
	elif acf == 'ak' or acf == 'AK':
		ax = corr[1]; az = corr[0]; H = corr[2]
	
	
	#%% set size for spectral synthesis tool that generates
	#%% fields of size (2*rmz+1, 2*rmx+1), i.e. the method 
	#%% requires an ODD number of points in each direction
	nptsX = round(N[1]/samp[1])  #% number of grid-points in X
	nptsZ = round(N[0]/samp[0])  #% number of grid-points in Z
   

	if np.mod(nptsX,2) == 0: rmx = int(nptsX/2)
	elif np.mod(nptsX,2) != 0: rmx = int((nptsX-1)/2)
	
	if np.mod(nptsZ,2) == 0: rmz = int(nptsZ/2)
	elif np.mod(nptsZ,2) != 0: rmz = int((nptsZ-1)/2)
	
	
	#%% compose power spectrum for two of the four quadrants
	#%% wavenumber vector in [-pi,pi]
	kx = (rmx/((2*rmx+1)*samp[1]))*np.linspace(-2*np.pi,2*np.pi,2*rmx+1)
	kz = (rmz/((2*rmz+1)*samp[0]))*np.linspace(-2*np.pi,2*np.pi,2*rmz+1)
	rk = [int(rmz+1),int(2*rmx+1)]
	kr = np.zeros(rk)
	k1 = np.zeros(rk)


	for j in range(0,int(2*rmx+1)):		
		for i in range(0,int(rmz+1)):		
			if acf == 'fr' or acf == 'FR':
				kr[i,j] = np.sqrt((kz[i]**2) + (kx[j]**2))
			else:
				kr[i,j] = np.sqrt((az**2)*(kz[i]**2) + (ax**2)*(kx[j]**2))
	
	 #%% calculate power spectral density, depending on selected ACF
	if acf == 'gs' or acf == 'GS':
		PS = 0.5*ax*az*np.exp(-0.25*kr**2)
	elif acf == 'ex' or acf == 'EX':
		PS = (ax * az)/(1 + (kr**2))**1.5
	elif acf == 'ak' or acf == 'AK':
		# for j in range(0,int(2*rmx+1)):		
		# 	for i in range(0,int(rmz+1)):		
		# 		k1[i,j]=kr[i,j]
		k1=kr[0:int(rmz+1),0:int(2*rmx+1)]
		k3=k1.conj().transpose()
		a1,b1=np.shape(k1)
		k2=k3.reshape(a1*b1,1)
		ka = k2.compress((k2>0).flat)
		# In [14]: print k2.compress.__doc__
		# k2.compress(condition, axis=None, out=None)
		#   .flat  also appears if k2 is np.array 
		c1=np.size(ka)
		ka2=ka.reshape(c1,1)
		coef = 4*np.pi*H*ax*az/scipy.special.kv(H,min(ka))
		PS = coef/(1 + (kr**2))**(H+1)
	#	coef = 4*pi*H*ax*az./besselk(H,min(k))
	#	 PS = coef./(1 + (kr.**2)).**(H+1)
	elif acf == 'fr' or acf == 'FR':
		decay = 0.5*(8-2*D)
		#% to ensure proper scaling of the power spectrum 
		#% in k-space we cannot allow kr == 0 
		# (just make sure, not important if corner wavenumber is applied)
		if kr.min() == 0:

			a,b=np.shape(kr)
			for i in range(0,a):
				for j in range(0,b):
					if kr[i,j]==0:
						p,q= i,j
						kr[p,q] = np.mean(kr[p-1:p+1,q-1:q+1])
		#% set values below k< kc to constant pi*kc
		sgival(kr,np.pi*kc,np.pi*kc)
		PS = 1/((kr**2)**decay)		 	
	# 
	# #%% the IFFT needs the spectrum normalized to max.amp unity
	PS = PS/PS.max()
	AM = np.sqrt(PS)
	# 
	# 
	# #%% compose the random phase
	# #%% initialize random number generator
	if Rseed is None:
		Rseed = int(time.time())

	np.random.seed(Rseed)

    
	# print('\n Time Stamp {0:.6e} s \n'.format(time.process_time()-t))
	# #%% random phase in [0,pi]
	m,n=np.shape(kr)
	PH=np.zeros((m,n))
	PH = np.random.random((m,n))*2*np.pi
	# for i in range(0,m):
	# 	for j in range(0,n):
	# 		PH[i,j] = 2*np.pi*random.random()
	  
	# #%% assemble the random field in FT-domain  
	# # add small random high-wavenumber components
	# #x = (1 + 0.5*randn(size(kr)))
	x = 1
	RAD = AM*x
	  
	# #%% set DC-component to different value, if desired
	# #%% NOTE that this only changes the overall 'level' of
	# #%% the field, not its appearance, but has significant
	# #%% impact on the Fourier transform which reflects the
	# #%% DC-value at the smallest wavenumber ("Nugget Effect")
	Neff = 0							   #% "Nugget" alue
	RAD[rmz,rmx] = Neff  	#% "Nugget" effect, zero-mean field
	AM[rmz,rmx]  = RAD[rmz,rmx]
	m1,n1=np.shape(PH)
	Y=np.zeros((m1,n1))
	# for i in range(0,m1):
	# 	for j in range(0,n1):
	Y = RAD*np.cos(PH)+1j*RAD*np.sin(PH)
# Matlab and Python need *np.pi/180.0, but here is not used. Why?

	# #%% the following takes care of the conjugate symmetry condition
	# #%% in order to ensure the final random field is purely real
	aa=[2*rmz+1,2*rmx+1]
	U = np.zeros(aa)+1j*np.zeros(aa)		  #% will be conj. sym. field
	Y = np.concatenate((Y , np.conj(np.fliplr(np.flipud(Y[0:rmz,:]  )))))

	for i in range(0,int(rmx)):
		Y[rmz,-i+2*rmx+2-2] = np.conj(Y[rmz,i])
	
	for i in range(0,int(rmz)+1):
		for j in range(0,int(rmx)+1): U[i,j] = Y[i+rmz,j+rmx]

	for i in range(int(rmz)+1,2*int(rmz)+1):
		for j in range(int(rmx)+1,2*int(rmx)+1): U[i,j] = Y[i-rmz-1,j-rmx-1]
   
	for i in range(0,int(rmz)+1):
		for j in range(int(rmx)+1,2*int(rmx)+1): U[i,j] = Y[i+rmz,j-rmx-1]
	
	for i in range(int(rmz)+1,2*int(rmz)+1):
		for j in range(0,int(rmx)+1):
			U[i,j] = Y[i-1-rmz,j+rmx]

#			return U, Y
#	return U[i,j], Y[i-1-rmz,j+rmx]
#			UU=U[i,j]

#		return U[i,j]
	
	# print('\n Time used before {0:.6e} s \n'.format(time.process_time()-t))
	# #%% take 2D-inverse FFT to obtain spatial field imaginary parts
	# #%% of order 1e-13 due to machine precision are removed 
	# #%% also, remove mean and scale to unit variance
	# t = time.process_time()
	Y = np.real(ifft2(U, threads=multiprocessing.cpu_count()))
	# print('\nifft2 uses {0:.6e} s by {1} threads\n'.format(time.process_time()-t,multiprocessing.cpu_count()))
	Y = Y/np.std(Y,ddof=1)  			# standard deviation of unity
	if np.mean(Y) < 0: Y = (-1)*Y  # positive mean (though small)

	   
	# #%% due to the requirement of [2*rmz+1,2*rmx+1] grid points, the
	# #%% simulated field may be too large by one point in either direction.
	# #%% This is compensated by resampling
	# #Y = resampgrid(Y,[nptsZ nptsX])
	# 
	# 
	# #%% final output structure with len vectors,sample spacing
	# #%% and model parameters
	spar={}
	spar['dim']   = N
	spar['samp']  = samp 
	spar['size']  = np.shape(Y)
	spar['corr']  = corr
	spar['acf']   = acf
	spar['Rseed'] = Rseed
	a1,a2=np.shape(Y)
#	spar['lx'] = np.arange(0,samp[1]*(a2-1)+1,samp[1])
	spar['lx'] = np.arange(0,samp[1]*(a2),samp[1])
#	spar['lx'] = np.arange(0,samp[1]*(a2-1),samp[1])
	spar['lz'] = np.arange(0,samp[0]*(a1),samp[0])
#	spar['lz'] = np.arange(0,samp[0]*(a1-1)+1,samp[0])

	# #% assemble structure with spectra in pos. quadrant
   
	px=[]
	for i in range(0,np.size(kx)):
		if kx[i]>=0:
			px.append(i)

	pz=[]
	for i in range(0,np.size(kz)):
		if kz[i]<=0:
			pz.append(i)

	spec={}
	spec['PD']  = PS[:,px]
	spec['kpx'] = kx[px]
	spec['kpz'] = kz[pz]
	m,n=np.shape(spec['PD'])
	spec['PDx'] = spec['PD'][m-1,]
	spec['PDz'] = spec['PD'][:,0]

	return Y, spar, spec
#######################################################
	

def randomfieldspecdistr(*args):
	#    srcpar - Array-structure with ALL source parameters OR 
    #             cell-array with source parameters and string to identify scaling;
    #             OR simple vector with source parameters
    #             For the last two options, srcpar can be of the form:
    #                  {Mw 'rel'} --  source dimensions computed from scaling laws
    #                         rel == 'MB' uses Mai & Beroza (2000)
    #                         rel == 'WC' uses Wells & Coppersmith (1994)
    #                         rel == 'WG' uses USGS WorkingGroup99 M vs. A (2000)
    #                  {A 'rel'}  --  area A (in km**2), rel is 'MB' or 'WC'
    #                  [W L]      --  Mw estimated from Wells & Coppersmith (1994),
    #  
    #  
    #    acf    - string to denote autocorrelation function
    #                  'ak' or 'AK' for anisotropic von Karman ACF
    #                  'ex' or 'EX' for exponential ACF
    #                  'fr' or 'FR' for the fractal case (power law decay)
    #                  'gs' or 'GS' for Gaussian ACF
    #                      for this option YOU have to SPECIFY the correlation length
    #                      if {}, default 'ak' is used
    #  
    #    corr   - correlation length and/or spectral decay parameters
    #                  [az ax] for Gauss or exponential ACF
    #                  [az ax H] for von Karman ACF H = Hurst number)
    #                  [D kc] for fractal slip where D is the fractal dimension;
    #                      kc: corner wavenumber beyond which the spectrum decays;
    #                      kc is related to the source dimensions, and is computed
    #                      as kc = 2*pi/(sqrt(L*W)) if it is not given
    #                  {} if corr is an empty matrix, the relevant parameters for the
    #                      given autocorrelation function will be computed
    #                      (NOT true for the Gaussian since no relations have been
    #                      established between corr length and source parameters)
    #  
    #    seed   - structural array of seed values for the random-number generator, 
    #                  called at various locations in the code; if seed == {}, then 
    #                  the code uses an expression like 
    #                          'Rseed = sum(100*clock)' 
    #                          'randn('seed', Rseed)' <-- uses MATLAB4 generators!!
    #                  to generate the random numbers; the value Rseed is stored in
    #                  the output structure par.
    #                  The sequence is as follows (also returned by the code):
    #                          seed.SS = SSseed; 1x2-array, used in SpecSyn2
    #                          seed.WL = WLseed; 1x1-array, used in WaterLevel
    #                          seed.CS = CSseed; 2x2-array, used in CalcSigfromD
    #                          seed.RC = RCseed; 3x2-array, used in CalcDimfromM
    #                          seed.RWC = RWCseed; 3x2-array, used in CalcDimWC
    #                          seed.CR = CRseed; 1x1, 4x2, 5x2, used in CalcCorrfromLW
    #                  Hence, you can run the code once, get the corresponding array 
    #                  and use it again to create the EXACT SAME random slip model.
    #  
    #    samp   - sampling of dislocation model in z,x direction [dz dx]
    #                  NOTE: the final sampling may be finer in case the option 'pow2' 
    #                      is given as 'y' (see SPECSYN2) or sampling must be adjusted for
    #                      the source dimensions
    #  
    #    grd	  - slip function to be defined on grid-nodes or subfaults 
    #                  'nod' for grid-node definition [grid-size (L/dx+1)*(W/dz+1)] 
    #                  'sub' for sub-fault definition [grid-size (L/dx) * (W/dz)]  
    #  
    #    nexp   - non-linear scaling exponent for the random field (i.e S = S**nexp) 
    #                  nexp < 1 smoothens the fields (steepens the spectrum) 
    #                  nexp == 1 doesn't change anything (default)
    #                  nexp > 1 roughens the field (flattens the spectrum)
    #                  the purpose of this variable is to allow for simulation of
    #                  slip distributions with large peak-slip values;
    #                  a good choice in this case is usually nexp = 1.25 - 1.5;
    #  
    #  
    #    taper  - tapering the edges of the slip model
    #                  'y' for default tapering of 2.5 km (i.e [2.5 2.5 2.5])
    #                  [left/right top bottom] (in km) for customized tapering
    #                  [left/right top bottom P] to apply an additional
    #                   depth-dependent tapering of the form z**P; 
    #                   P > 1 to obtain less slip in shallow part of the fault
    #  
    #    depth  - max. depth of rupture plane;
    #             option with depth range [zmin zmax] not implemented
    #                      zmin is the depth to the top of the fault plane
    #                      zmax is the maximum depth of fault plane (in km)  
    #                           (default: zmin = 0, zmax =15)  # I changed to 30 km
    #  
    #    dip	  - dip angle of fault plane (in degrees) (default = 90 deg)  
    #  
    #    fig	  - optional: 'y' to view the slip realization; this will open
    #                  a figure window for each realization (set to 'n' if called 
    #                  in a loop) (default: 'y')
    #  
    #    outfile- optional: string for a filename to which the code writes 
    #                  the slip realization as a simple ascii 2D-array, where rows
    #                  correspond to depth (default: 'n')
    #  
    #    OUTPUT:	
    #    S 	  - 2D distribution
    #    par 	  - structure with all relevant source parameters 
    #  


	# constant parameters:
	tapwin = 'hn'

	if len(args)==12:
		srcpar=args[0]
		acf=args[1]
		corr=args[2]
		seed=args[3]
		samp=args[4]
		grd=args[5]
		nexp=args[6]
		taper=args[7]
		depth=args[8]
		dip=args[9]
		fig=args[10]
		outfile=args[11]
	else:
		print('Error in numbers of auguments')
		return

	if seed is None:
		# SSseed = []
		SSseed = None
		print('   new SEED-values used ')
	else:
		SSseed = seed
		print('   SEED-values from previous simulation used')
		print('	  Used seed: {0} \n'.format(SSseed))

	if len(srcpar)==2:
		W = float(srcpar[0])
		L = float(srcpar[1])
	else:
		print('   Error in srcpar input ')

	h = depth - W*math.sin(np.pi*dip/180.)	  # top of fault plane

	#### check whether values of L and W, together with the selected
	#### spatial sampling, yield an integer number of point; otherwise
	#### the spectral synthesis will return an error message. For now
	#### we consider only one significant digit. The program will then
	#### adjust L and W in order to maintain the chosen sampling
	L = 0.1*round(L*10)		
	W = 0.1*round(W*10)

	if np.abs(L-int(L/samp[1])*samp[1])>1e-6:
		print('   --> Need to adjust length L in order to be compatible')
		print('	   with the chosen spatial sampling')
		nnx = L/samp[1]
		L  = round(nnx)*samp[1]
	if np.abs(W-int(W/samp[0])*samp[0])>1e-6:
		print('   --> Need to adjust width W in order to be compatible')
		print('	   with the chosen spatial sampling')
		nnz = W/samp[0] 
		W  = round(nnz)*samp[0]


	print('   ** Final Source Parameters: ')
	print('  Fault   L = {0:.2f} m, W = {1:.2f} m'.format(L,W) )

	#### -----------------------------------------------------------------
	#### TAPERING OF SLIP FUNCTION, [left/right top bot wfct]
	#### to avoid large slip at the fault boundaries, we apply a taper 
	#### function that may vary for the left/right, top and bottom extent
	#### of the taper window; the default is set to 2.5 km. Additionally,
	#### a depth-dependent "weighting function" can be applied such as to 
	#### have larger slip values at depth
	#### -----------------------------------------------------------------

	print('Taper type: {0}'.format(type(taper)))
	print('Taper parameter numbers: {0}'.format(len(taper)))

	if len(taper) == 4:
		twdpt = taper[3]		# additional taper with depth
	elif len(taper) == 0:
		taper = [0, 0, 0]		# array of zeros in case of NO taper

	if taper == 'y':				
		twkm = [2.5, 2.5, 2.5]			# set taper windows to default values 
	elif taper == 'd':
		if 'twfr' not in locals():
			twfr = 0.25
			twL = twfr*L; twW = twfr*W	# set taper window length to 25# of 
			twkm = [twL, twW, twW]		# the source size in each direction

	elif type(taper) is list and len(taper)>=3:			
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

	elif len(samp) == 1: 
		ntx = round(twkm[0]/samp[0])  # Esta raro, deberia ser tambien 0 porque solo tiene 1, pero asi viene del de Martin
		print('##*$ ntx {0}'.format(ntx))
		ntt = round(twkm[1]/samp[0])
		ntb = round(twkm[2]/samp[0])
		print('Side   taper node ntx {0}'.format(ntx))
		print('Top	  taper node ntx {0}'.format(ntt))
		print('Bottom taper node ntx {0}'.format(ntb))
		tapslp = [ntx, ntt, ntb] 	# defines the # of rows/cols of the
									## output slip distribution to be tapered	

	if acf == 'ak' or acf == 'ex' or acf == 'gs':
			if W > L and corr[0] < corr[1]: 
				corr[0:1] = [corr[1], corr[0]]
	   
	elif len(corr) == 1 and acf == 'fr' or acf == 'FR': 
	#### need to compute corner wavenumber, based on source dimensions
		kc = 2*pi/(sqrt(L*W))
	#kc = 1/sqrt(L*W)
		corr = [corr[0], kc]
	else:
		corrmeth = 'specified'
		print('   Spectral decay parameters used as specified')



	if acf == 'ak' or acf == 'AK':
		H = corr[2]  #Ver pk esta mal
		if H < 0.5:
			print('   ++ Hurst exponent H < 0.5 theoretically NOT allowed (for slip generating) ')
		if H >= 1:
			###corr(3) = 0.99;
			print('   -- Hurstnumber computed/given: H > 1')
			print('	  accepted, but outside the range of H [0;1]')
			##print '	  corrected to max. allowed H = 0.99 '
		elif H <= 0: 
			corr[2] = 0.01
			print('   -- Hurstnumber computed: H < 0')
			print('	  corrected to min. allowed H = 0.01')
			print('	  NOTE: spectrum will be EXTREMELY flat, ')
			print('	  generating a very heterogeneous slip model')

	#### print ' final spectral decay parameters to screen')
	if acf == 'ex' or acf == 'EX' or acf == 'gs' or acf == 'GS':
		print('   Final Decay Parameters: ')
		print('	   az = {0:.2f} m, ax = {1:.2f} m'.format(corr[0], corr[1]))
	elif acf == 'ak' or acf == 'AK':
		print('   Final Decay Parameters: ')
		print('	   az = {0} m, ax = {1} m, H = {2:.2f}'.format(corr[0], corr[1], corr[2]))
#		print('	   az = {0:.2f} km, ax = {1:.2f} km, H = {2:.2f}'.format(corr[0], corr[1], corr[2])
	elif acf == 'fr' or acf == 'FR':
		print('   Final Decay Parameters: ')
		print('	   D = {0:.2f}, kc = {1:.2f}'.format(corr[0], corr[1]))


	## this is the standard option to use
	t = time.process_time()
	G,spar,spec = SpecSyn2([W, L],samp,corr,acf,SSseed)
	print('SpecSyn2 runs in {0:.5e} s'.format(time.process_time()-t))

	SSseed = spar['Rseed']		# save the seed value
	print('Seed number %d : ' %(SSseed))

	G = G - np.mean(G)		# to ensure that the simulated random field
	G = G/np.std(G,ddof=1)			# has zero mean and unit variance


	#### crude check that positive values are concentrated in the interior
	#### of the field; this should avoid large slip at the boundaries which
	#### will result in large stresses and unrealistically "patchy" slip models
	lz,lx = np.shape(G) 
	px = round(lx/np.sqrt(2))		# dimensions for 'interior' area
	pz = round(lz/np.sqrt(2))	
	qx = int(np.floor(0.5*(lx-px)))	# indices for 'interior' area
	qz = int(np.floor(0.5*(lz-pz)))		
	m,n= np.shape(G)
	GI = G[qz:m-qz,qx:n-qx]
	if np.mean(GI) < np.mean(G): G = -1*G


	#### ---------------------------------------------------------
	#### RESIZE AND SCALE FIELD TO MATCH DESIRED SOURCE PARAMETERS
	#### ---------------------------------------------------------
	   
	#### resize grid (by bilinear interpolation) in case of subfault definition
	#### Note that spectral synthesis tool works on grid-dimensions of ODD size!
	print('Node or Cell: {0}'.format(grd))

	if grd == 'sub':
		print('   Defined on SUBFAULTS selected')
		G,spar['lx'],spar['lz'] = interpgrid2(G,spar['samp'],spar['dim'],spar['samp']) #bug: there is no interpgrid2 function
	else:
		print('   Defined on GRID-NODES selected ')


	#### perform a non-linear transformation, if desired, to create models
	#### with higher peak-slip values
	if nexp != 1:
		print('  Non-linear scaling of field function: S=S**{0}'.format(nexp))
		G = G - np.min(G)
		G = np.power(G,nexp)
		G = G - np.mean(G)

	if type(taper)==list and len(taper) and max(taper)>0 >= 3:
		print('   Tapered at the boundaries:')
		print('   left/right, top, bottom: [ {0:.1f}, {1:.1f}, {2:.1f}] km'.format(twkm[0], twkm[1], twkm[2]))
		G = Taperfunction(G,tapslp,tapwin) 
	 
		if 'twdpt' in locals():
			print(' Additional depth taper applied: Sz**{0}'.format(twdpt))
			i1,j1=np.shape(G)
			w1 = np.linspace(1,i1*samp[0],i1)
			w = np.transpose(w1**twdpt)
			w = w/max(w)
			for i in range(0,j1):
				G[:,i]=G[:,i]*w

		G = G - np.mean(G)
		G = G/np.std(G,ddof=1)			# has zero mean and unit variance

   
	#### simple plot of resulting slip realization
	if fig == 'y': 
		print('  Realization will be graphically displayed')
		PlotFigure(G,W,L)

	return G,spar


##########################################





if __name__=='__main__':

	W=25.0			  # Fault width
	L=80.0			 # Fault length
	Mw=6.5			 # Moment Magnitude
	srcpar = [W, L]
	acf='ak'			 # autocorrelation function
	c1=2			   # corr=[c1, c2, Hn]=  array of correlation length [ax az H]
	c2=8			   #
	Hn=0.5			 # Hn= Hurst number
	corr = [c1,c2,Hn]
	seed=1585020116
	sa1=0.5			# Along the strike sampling
	sa2=0.5			# Down dip sampling
	samp=[sa1, sa2]
	grd='nod'			#
	nexp=1.		   # non-linear scaling exponent for the random field (i.e S = S^nexp)
	tp1=5			  # taper=[tp1, tp2, tp3]= array of taper in km for [left/right top bottom]
	tp2=0			  #
	tp3=5			  #
	tp4=0.5			  # depth dependent tapering of the form z^tp4
	taper = [tp1, tp2, tp3, tp4]
	depth=25.0		 # max. depth of rupture plane
	dip=90			 # Dip (Default for "ss"=90; for "ds"=50.0, OR CHOOSE YOUR VALUE
	fig='y'			  # Show figure
	outfile='n'
	
	G,spar=randomfieldspecdistr(srcpar,acf,corr,seed,samp,grd,nexp,taper,depth,dip,fig,outfile)

	PlotFigure(G,W,L)
	