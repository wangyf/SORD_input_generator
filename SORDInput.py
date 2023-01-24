#!/usr/bin/env python
import sys, os
import numpy as np
from specrandom import *
import pyproj
from sordw3.extras import coord
import pandas
# from scipy import *
from scipy import interpolate
from utils import *

# global variable
debug=False
idtype = np.dtype( 'f8' ).str #'<f8'
slipvector=np.array([1,0,0])  #compatible with SORD
#idtype = np.dtype( 'f' ).str #'<f4'



faultflip = True #flip fault from left to right or the other
leftnode = 735   #left node distance to fault (Landers)
rightnode = 1024 #right node distance to fault(Landers)

# flat fault of landers
# nx = 3081
# ny = 351
# nz = 1715
# hypo = [2501,1,1044]

# rough fault of landers
nx = 3438
ny = 351
nz = 1810
hypo = [2501,1,1044]

# nx = 3081
# ny = 4
# nz = 1715
# hypo = [2501,1,1044]

dx = [50, 50, 50]
ifault = 3
fin='in/'
fname='goodbin/'
ftmp='tmp/'
fig='fig/'
rough='../Fault_roughness/rough_profile/'

faultzone_vs='../Fault_zone_generator/bin/fault_zone_vs.bin'
faultzone_GSI='../Fault_zone_generator/bin/fault_zone_GSI.bin'
faultzone_range= (1014, 1074),(1, 149),(725, 2424) # (iz1,iz2),(iy1,iy2),(ix1,ix2)
use_faultzone = False
try:
	os.mkdir(fname) 
except:
	pass

############### 
fromBBP = True
readmesh = False
readroughfault = False
searchstation = False


# fault stress and friction by segment
seglist=[2,3,4] #index list from N to S (left to right)
Ratio = [0.55, 0.55, 0.55] #prefered model: [0.5, 0.5, 0.5]
fco = [0, 0, 0]
mus = [0.5, 0.5, 0.5] #prefered model: [0.4, 0.4, 0.4]
musf =[0.55,0.5, 0.5] #final mus after computing ts tn 
mud = [0.3, 0.3, 0.3] #prefered model: [0.3, 0.3, 0.3]
dc = [0.6, 0.6, 0.6]
taper_layer = 4e3
nbdl = 10 # left  boundary pad before taper for coseismic area
nbdr = 100 # right boundary pad before taper for coseismic area
taperlist = [taper_layer,0,taper_layer] #used for taper mus and mud distribution
# taperlist = [taper_layer,0,0]

# offset vertical stress at surface
offset_stress = 0.0

# frictional cohesion at top xx km from fcotop linearly decay to zero
fcdepth= 3e3 #m
fcotop = 1e6 #Pa
fcrandom = True

# mud at top xx km from mudtop linearly decay to mud
muddepth = 0 #m
mudtop = [None,None,None]

# plasticity model
plmodel = 'HoekBrown' #'Horsrud2001' 'Chang2006', 'Roten2014', 'HoekBrown','closeness'
make='good'        #very good, good, average poor only for HoekBrown
# closeness = 0.6
plvar = make
weight  = [1,1,1]


# random field
randomfield=False
interp_ratio = 1
# ## random field
acf='ak'			 # autocorrelation function
c1=3e3			   # corr=[c1, c2, Hn]=  array of correlation length [ay ax H] [vertical and horizontal]
c2=3e3			   #
Hn=0.75			 # Hn= Hurst number
corr = [c1,c2,Hn]
seed = None
# seed = 1588487981
sa1=dx[0]			# Along the strike sampling
sa2=dx[1]		# Down dip sampling
samp=[sa1*interp_ratio, sa2*interp_ratio]
grd='nod'			#
nexp=1.		   # non-linear scaling exponent for the random field (i.e S = S^nexp)
tp1=2e3			  # taper=[tp1, tp2, tp3]= array of taper in km for [left/right top bottom]
tp2=0			  #
tp3=2e3			  #
tp4=0			  # depth dependent tapering of the form z^tp4
# taper = [tp1, tp2, tp3, tp4]
taper = []		#for ramdom field
depth=25.0e3		 # max. depth of rupture plane
dip=90			 # Dip (Default for "ss"=90; for "ds"=50.0, OR CHOOSE YOUR VALUE
nfig='n'			  # Show figure
outfile='n'

hgv = 0.4         #max value of ramdom field (1 means low mus and high stress drop:  mus = ts/tn mud=mud_bar+mus-ts/tn)
              						    #(0 means original mus and mud:    mus = mus_bar mud=mud_bar)
lov = -0.6       #min value of random field (-1 means high mus and low stress drop: mud = ts/tn)

mudoffset = 0.0 #perhaps used to genereate negative stress drop (suggested: 0.1)
musoffset = 0.0
	

kinkoff = False
# kink high friction
# kink index starts from segment 1 and 2
kinklist=[2,3]
nktaper=500 #meter
kinktaper=[nktaper,0,nktaper]
kdmus= 1e3  #	#kdmus is difference of mus at kink (relative to background mus)
kdmud= 1e3  #	#kdmud is difference of mud at kink (relative to background mud)
klov = -1.5  #	only apply for randomfield==true (set min value of randomfield F)
dep=0/dx[1]
kinkl = 2e3/dx[0]
kinkw = 3e3/dx[1]
				 #on each kinks
				 #kinkl (length along strike)
				 #kinkw (length along depth)
				 #dep (depth of kink patch top, 0 means from free surface)
				 #khgv max value of random field (1 means low mus and high stress drop)
				 #klov min value of random field (-1 means high mus and low stress drop)
kextent=np.array([[kinkl,kinkw,dep,kdmus,kdmud],\
				  [kinkl,kinkw,dep,kdmus,kdmud]])






























########################################
class SORDinput:
	"""
	SORDinput module
	"""
	def __init__(self,nx,ny,nz,hypo,ifault,dx):
		"""
		ifault is fault normal direction
		ihypo is hypocenter index
		"""
		self.nx = nx
		self.ny = ny
		self.nz = nz
		self.ifault=ifault
		self.hypo = hypo
		self.dx = dx
		print('Initiate SORDinput Module\n')
		print('Nx = %5d' %self.nx)
		print('Ny = %5d' %self.ny)
		print('Nz = %5d' %self.nz)
		print('Fault is normal to %d' %self.ifault)
		print('Hypocenter is at %5d %5d %5d' %(self.hypo[0],\
											   self.hypo[1],\
											   self.hypo[2]))
		print('Dx is %5f %5f %5f\n' %(dx[0],dx[1],dx[2]))

		return

	# fault segment along strike read in 
	# args is input of segment junctions (nseg-1)
	def fault_segment(self, nseg, *args):
		ierr = 1
		self.nseg = nseg
		self.segb = (1,)+args[0]
		self.sege = args[0] + (self.nx,)

		if len(self.segb) != nseg:
			print('Error input junction index')
			return ierr

		print('Number of fault segments: %3d' %self.nseg)
		print('Segment starting index: ', self.segb)
		print('Segment ending index: ',   self.sege)

		ierr = 0
		return ierr


	def init_depth(self,dtype=idtype):
		self.cdepth=np.zeros(self.ny-1,dtype=dtype)
		self.ndepth=np.zeros(self.ny,dtype=dtype)
		for i in range(self.ny-1):
			self.cdepth[i] = (i + 0.5) * self.dx[2] #at cell center
			self.ndepth[i+1] = (i + 1) * self.dx[2] #at node center
		return

	def customvel(self,dtype=idtype,method=None):
		self.vp=np.empty(self.ny-1,dtype=dtype)
		self.vs=np.empty(self.ny-1,dtype=dtype)
		self.rho=np.empty(self.ny-1,dtype=dtype)
		for i in range(self.ny-1):
			depth = self.cdepth[i]
			if method is None:
				self.rho[i],self.vp[i],self.vs[i]=Shuoma_velstru(depth, SI=True)
			else:
				print('Other velocity model has not been implemented!')
				sys.exit()
		return True
			

	def BBPvel(self,veltable,dtype=idtype):
		self.vp=np.empty(self.ny-1,dtype=dtype)
		self.vs=np.empty(self.ny-1,dtype=dtype)
		self.rho=np.empty(self.ny-1,dtype=dtype)

		depu = 0
		depd = veltable['thickness'][0]
		k = 0
		for i in range(self.ny-1):
			dep = self.cdepth[i]
			if dep >= depu and dep<= depd:
				self.vp[i] = veltable['vp'][k]
				self.vs[i] = veltable['vs'][k]
				self.rho[i] = veltable['rho'][k]
			else:
				while dep > depd:
					k = k + 1
					depu = depd
					depd = veltable['thickness'][k] + depu
				self.vp[i] = veltable['vp'][k]
				self.vs[i] = veltable['vs'][k]
				self.rho[i] = veltable['rho'][k]
		return

	def assign_stress(self,sigma_xx,sigma_zz,sigma_xz,ts, tn):

		self.sigma_xx = sigma_xx
		self.sigma_zz = sigma_zz
		self.sigma_xz = sigma_xz
		self.sigma_yy = np.repeat(self.sigma_y[:,np.newaxis],self.nx-1,axis=1)
		self.ts = ts
		self.tn = tn 

		return

	def rough_ts_tn(self,nhat_x,nhat_y,nhat_z):
		n0 = nhat_x.shape[0]
		n1 = nhat_x.shape[1]
		cnhat_x = (nhat_x[0:-1,0:-1] +   nhat_x[1:n0+1,0:-1] +\
			       nhat_x[0:-1,1:n1+1] + nhat_x[1:n0+1,1:n1+1])/4
		cnhat_y = (nhat_y[0:-1,0:-1] +   nhat_y[1:n0+1,0:-1] +\
			       nhat_y[0:-1,1:n1+1] + nhat_y[1:n0+1,1:n1+1])/4
		cnhat_z = (nhat_z[0:-1,0:-1] +   nhat_z[1:n0+1,0:-1] +\
			       nhat_z[0:-1,1:n1+1] + nhat_z[1:n0+1,1:n1+1])/4	

		start=domain.segb[1]
		end = domain.sege[-2]
		# pre-traction
		t0_x = self.sigma_xx[:,start+1:end+1] * cnhat_x
		t0_y = self.sigma_yy[:,start+1:end+1] * cnhat_y + self.sigma_xz[:,start+1:end+1] * cnhat_z
		t0_z = self.sigma_xz[:,start+1:end+1] * cnhat_y + self.sigma_zz[:,start+1:end+1] * cnhat_z

		# normal traction
		self.tn[:,start+1:end+1] = t0_x * cnhat_x + t0_y * cnhat_y + t0_z * cnhat_z

		self.ts[:,start+1:end+1] = np.sqrt((t0_x-self.tn[:,start+1:end+1] * cnhat_x)**2 +\
			                               (t0_y-self.tn[:,start+1:end+1] * cnhat_y)**2 +\
			                               (t0_z-self.tn[:,start+1:end+1] * cnhat_z)**2)
		self.tn[:,start+1:end+1] = -self.tn[:,start+1:end+1] #convert to postive tn

	def vertical_stress(self, rho=None, rhow=1000, g=9.81, offset=None):
		# offset is stress offset at zero depth
		if rho is None:
			rho = 2700
		if offset is None:
			offset = 0.

		# customized define
		# sigma_y = -offset - self.cdepth * g * (rho - rhow)

		# integral of density
		gh=gammaH(self.rho,self.dx[1])
		sigma_y = -offset + self.cdepth * g * rhow - gh

		self.sigma_y = sigma_y
		return sigma_y

	def horizontal_stress_seg(self, *args):
		# postive for compressive stress 

		# Ratio: Stress ratio (S = 1/Ratio - 1)
		try:
			fault_orientation = args[0]
			angle_sigma_1 = args[1]
			seglist = args[2]
			sigma_2 = args[3]
			Ratio = args[4]
			co = args[5]
			mus = args[6]
			mud = args[7]
		except:
			print('Input wrong in horizontal_stress_seg')
			return

		nseg = len(fault_orientation)

		sigma_1 = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		sigma_3 = np.zeros((self.ny-1,self.nx-1),dtype=idtype)


		sigma_xx = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		sigma_zz = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		sigma_xz = np.zeros((self.ny-1,self.nx-1),dtype=idtype)

		ts = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		tn = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		ts2 = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		tn2 = np.zeros((self.ny-1,self.nx-1),dtype=idtype)

		for ifault in range(nseg):
			leftindex=self.segb[seglist[ifault]-1]-1
			rightindex=self.sege[seglist[ifault]-1]-2
			upindex = 0
			downindex = self.ny-2

			print('Seg %d L %d - R %d (by node)' \
				%(seglist[ifault]-1,leftindex+1,rightindex+1))
			print('Seg %d U %d - D %d (by node)' \
				%(seglist[ifault]-1,upindex+1,downindex+1))

			R=Ratio[ifault]
			c=co[ifault]
			s=mus[ifault]
			d=mud[ifault]
			rot = angle_sigma_1[ifault] -270.
			print('Phi angle:', rot) # see attached notes
			rot2 = 270-fault_orientation[ifault]
			print('Alpha angle:', rot2)
			for i in range(leftindex,rightindex+1):
				term1 = (R*c-(R*s+(1-R)*d)*sigma_2[:])
				term2 = np.sin(2*(rot+rot2)*np.pi/180.)+\
					(R*s+(1-R)*d)*np.cos(2*(rot+rot2)*np.pi/180.)
				deltasigma = -term1/term2

				sigma_1[:,i] = sigma_2[:] + deltasigma
				sigma_3[:,i] = sigma_2[:] - deltasigma

				# computate stress tensor (sigma_xx sigma_yy)

				sigma_xx[:,i],sigma_zz[:,i],sigma_xz[:,i] = \
				plane_rotate(sigma_1[:,i],sigma_3[:,i],0, rot)
				ts[:,i] = -deltasigma*np.sin(2*(rot+rot2)*np.pi/180.)
				tn[:,i] = -sigma_2+deltasigma*np.cos(2*(rot+rot2)*np.pi/180.)

				# verify: compute from sigma_1 and sigma_3
				# ts2[:,i] = (sigma_3[:,i]-sigma_1[:,i])*(\
				# 			np.cos((rot+rot2)/180.*np.pi)*\
				# 			np.sin((rot+rot2)/180.*np.pi))
				# tn2[:,i] = -np.sin((rot+rot2)/180.*np.pi)**2*sigma_1[:,i]-\
				#			 np.cos((rot+rot2)/180.*np.pi)**2*sigma_3[:,i]

				# verify: compute from sigma_xx sigma_zz sigma_xz
				# tn2[:,i] = -np.sin(rot2*np.pi/180.)**2*sigma_xx[:,i] - \
				#			np.cos(rot2*np.pi/180.)**2*sigma_zz[:,i] + \
				#			2*np.sin(rot2*np.pi/180.)*np.cos(rot2*np.pi/180.)*\
				#			sigma_xz[:,i]
				# ts2[:,i] = -np.sin(rot2*np.pi/180.)*np.cos(rot2*np.pi/180.)*\
				# 		   (sigma_xx[:,i]-sigma_zz[:,i]) + \
				# 		   np.cos(2*rot2*np.pi/180.)*sigma_xz[:,i]

				## both passed

		return sigma_1,sigma_3,sigma_xx,sigma_zz,sigma_xz, ts, tn



	def friction_seg(self, *args):
		try:
			seglist = args[0]
			fco = args[1]
			mus = args[2]
			mud = args[3]
			dc = args[4]
		except:
			print('Input wrong in friction')

		self.fco = np.zeros((self.ny,self.nx),dtype=idtype)
		self.mus = np.zeros((self.ny,self.nx),dtype=idtype)
		self.mud = np.zeros((self.ny,self.nx),dtype=idtype)
		self.dc  = np.zeros((self.ny,self.nx),dtype=idtype)

		nseg = len(seglist)

		for ifault in range(nseg):
			leftindex=self.segb[seglist[ifault]-1]-1
			rightindex=self.sege[seglist[ifault]-1]
			upindex = 0
			downindex = self.ny

			self.fco[upindex:downindex,leftindex:rightindex] = fco[ifault]
			self.mus[upindex:downindex,leftindex:rightindex] = mus[ifault]
			self.mud[upindex:downindex,leftindex:rightindex] = mud[ifault]
			self.dc[upindex:downindex,leftindex:rightindex]  =  dc[ifault]


			if muddepth > 0 and mudtop is not None:
				for j in range(upindex,downindex):
					if self.ndepth[j]>muddepth:
						break
					else:
						self.mud[j,leftindex:rightindex] = \
						 		mudtop[ifault] - self.ndepth[j]/muddepth*(mudtop[ifault]-mud[ifault])



			if fcdepth > 0 and fcotop > 0:
				for j in range(upindex,downindex):
					if self.ndepth[j]>fcdepth:
						break
					else:
						if not fcrandom:
							self.fco[j,leftindex:rightindex] = \
						 		fcotop - self.ndepth[j]/fcdepth*fcotop
						else: #thus a weight function taper from the surface (1) to 0 at fcdepth
						 	self.fco[j,leftindex:rightindex] = \
						 		1 - self.ndepth[j]/fcdepth
		return

	def stationindex(self,*args):
		try:
			meshx = args[0]
			meshz = args[1]
			stax  = args[2]
			staz  = args[3]
		except:
			print('Input wrong in mesh and station list')

		nmesh=meshx.shape
		nsta=len(stax)

		flag = [False,]* nsta

		self.stationidx=np.empty((nsta,2))


		# j = 2862
		# i = 1017
		# ist=20
		# poly1=[meshx[i,j],meshz[i,j]]
		# poly2=[meshx[i+1,j],meshz[i+1,j]]
		# poly3=[meshx[i+1,j+1],meshz[i+1,j+1]]
		# poly4=[meshx[i,j+1],meshz[i,j+1]]
		# print(point_inside_polygon(stax[ist], staz[ist], \
		# 					np.array([poly1,poly2,poly3,poly4]), include_edges=True))

		# exit()
		for j in range(nmesh[1]-1):
			for i in range(nmesh[0]-1):
				
				if debug:
					print('Searching cell of {0} along z {1} along x'.format(i,j))

				poly1=[meshx[i,j],meshz[i,j]]
				poly4=[meshx[i+1,j],meshz[i+1,j]]
				poly3=[meshx[i+1,j+1],meshz[i+1,j+1]]
				poly2=[meshx[i,j+1],meshz[i,j+1]]
				# print('{0},{1}'.format(i,j))
				# print(flag)
				if not all(flag):
					for ist in range(nsta):
						
						if debug:
							print('{0}th station is at x:{1} z:{2}'.format(ist+1,stax[ist], staz[ist]))
						if i != hypo[2]-1 and \
						   flag[ist]==False and point_inside_polygon(stax[ist], staz[ist], \
							np.array([poly1,poly2,poly3,poly4]), include_edges=True):
							print('Station found {0}'.format(ist+1))
							flag[ist]=True
							idx,idz=staindexsearch(poly1,poly2,poly3,poly4,[stax[ist],staz[ist]],method='fine')
							self.stationidx[ist,:]=np.array([idx+j+1,idz+i+1])

		print('Station index searching completed')
		return 


	def vs_fault_zone3D(self,*args):
		try:
			vsfile = args[0]
			iz = args[1][0] # starting index along each axis
			iy = args[1][1] # ending   index along each axis
			ix = args[1][2] # ending   index along each axis
		except:
			print('Input wrong in vs_fault_zone3D')

		self.faultzone_vs=readbin(vsfile,(iz[1]-iz[0]+1,iy[1]-iy[0]+1,ix[1]-ix[0]+1))
		print('Reading vs in fault zone completed')
		return


	def GSI_fault_zone3D(self,*args):
		try:
			GSIfile = args[0]
			iz = args[1][0] # starting index along each axis
			iy = args[1][1] # ending   index along each axis
			ix = args[1][2] # ending   index along each axis
		except:
			print('Input wrong in GSI_fault_zone3D')

		GSI=readbin(GSIfile,(iz[1]-iz[0]+1,iy[1]-iy[0]+1,ix[1]-ix[0]+1))
		

		mb = self.tmi * np.exp((GSI-100)/(28 - 14*0))
		s = np.exp((GSI-100)/(9-3*0))
		a = 0.5 + 1/6*(np.exp(-GSI/15) - np.exp(-20/3))

		sigma_cm = self.tdci*((mb+4*s-a*(mb-8*s))*(mb/4+s)**(a-1))/(2*(1+a)*(2+a))

		sigma_3max = sigma_cm*0.47*(sigma_cm/- \
			np.repeat(self.sigma_yy[np.newaxis,iy[0]:iy[1]+1,ix[0]:ix[1]+1],iz[1]-iz[0]+1,axis=0))**(-0.94)

		sigma_3n = sigma_3max/self.tdci

		self.faultzone_pco = (self.tdci*((1+2*a)*s+(1-a)*mb*sigma_3n)*(s+mb*sigma_3n)**(a-1))/\
						          ((1+a)*(2+a)*np.sqrt(1+(6*a*mb*(s+mb*sigma_3n)**(a-1))/((1+a)*(2+a))))
		self.faultzone_phi = np.tan(np.arcsin((6*a*mb*(s+mb*sigma_3n)**(a-1))/\
							       (2*(1+a)*(2+a)+6*a*mb*(s+mb*sigma_3n)**(a-1))))


		# for k in range(iz[0],iz[1]+1):
		# 	for j in range(iy[0],iy[1]+1):
		# 		for i in range(ix[0],ix[1]+1):

		# 			s11 = self.sigma_xx[j,i]
		# 			s22 = self.sigma_yy[j,i]
		# 			s33 = self.sigma_zz[j,i]
		# 			s13 = self.sigma_xz[j,i]

		# 			sbar,sy = plasticyieldsurface([s11,s22,s33,0,s13,0],\
		# 				self.faultzone_phi[k-iz[0],j-iy[0],i-ix[0]],self.faultzone_pco[k-iz[0],j-iy[0],i-ix[0]])
		# 			if sbar > sy:
		# 				print('{0} should be < {1} at ix={2} iy={3} iz={4} in fault zone'.\
		# 					format(sbar,sy,i-ix[0],j-iy[0],k-iz[0]))

		print('computed pco and phi in fault zone completed')
		return




	def plasticity2D(self, *args):
		try:
			seglist = args[0]
			weight = args[1]
			method=args[2] #''
		except:
			print('Input wrong in plasticity 2D')

		self.method = method
		self.pco = np.zeros((self.ny-1,self.nx-1),dtype=idtype)
		self.phi = np.zeros((self.ny-1,self.nx-1),dtype=idtype)

		if method == 'HoekBrown':
			self.GSI   =    np.zeros((self.ny-1,self.nx-1),dtype=idtype)
			self.D     =    np.zeros((self.ny-1,self.nx-1),dtype=idtype)
			self.mi    =    np.zeros((self.ny-1,self.nx-1),dtype=idtype)
			self.dci   =    np.zeros((self.ny-1,self.nx-1),dtype=idtype)

			rocktype = args[3] #'' #follow Roten et al., 2017 GRL
			print('\n\nRock type is {0}\n'.format(rocktype))

			if rocktype == 'verygood':
				#Granite, diorite, very blocky, well interlocked and undisturbed
				self.tGSI = 75
				self.tmi = 25
				self.tdci = 150e6 
			elif rocktype == 'good':
				#Sandstone, basalt, blocky, interlocked and partly disturbed
				self.tGSI = 62.5
				self.tmi = 18.5
				self.tdci = 115e6
			elif rocktype == 'average':
				#Limestone, disintegrated, poorly interlocked
				self.tGSI = 50
				self.tmi = 12
				self.tdci = 80e6
			elif rocktype == 'poor':
				#Limestone, disintegrated, poorly interlocked
				self.tGSI = 30
				self.tmi = 8
				self.tdci = 40e6
			else:
				print('Input wrong in HoekBrown Rock type')
				sys.exit()

		if method == 'closeness':
			closeness = args[3]

		nseg = len(seglist)

		if method == 'HoekBrown':
			gh=gammaH(self.rho,self.dx[1])

		for ifault in range(nseg):
			leftindex=self.segb[seglist[ifault]-1]-1
			rightindex=self.sege[seglist[ifault]-1]-2
			upindex = 0
			downindex = self.ny-2

			for i in range(leftindex,rightindex+1):
				for j in range(upindex,downindex+1):

					if method=='Horsrud2001':
						if self.vs[j] < 2500: # alluvium and soft rock
							self.phi[j,i] = np.tan(35.*np.pi/180.)
						else: #hard rock
							self.phi[j,i] = np.tan(45.*np.pi/180.)
						mu  = self.rho[j] * self.vs[j] * self.vs[j]
						lam = self.rho[j] * self.vp[j] * self.vp[j] - 2 * mu
						BM  = lam + (2.*mu)/3. #bulk modulus 
						E   = (9.*BM*mu)/(3.*BM+mu) #Young's modulus
						PR  = lam/(lam+mu)/2. #Possion Ratio
						self.pco[j,i] = 3.61*(E/1e9)**0.712*1e6 * weight[ifault]#in Pa


					if method=='Chang2006':
						if self.vs[j] < 2500: # alluvium and soft rock
							self.phi[j,i] = np.tan(35.*np.pi/180.)
						else: #hard rock
							self.phi[j,i] = np.tan(45.*np.pi/180.)

						delta = 1/self.vp[j]*1e6/3.28084	
						self.pco[j,i] = 0.7069*1e7*np.power(delta,-3)*1e6 * weight[ifault]#in Pa


					if method == 'Roten2014':
						if self.vs[j] < 2500: # alluvium and soft rock
							self.phi[j,i] = np.tan(35.*np.pi/180.)
						else: #hard rock
							self.phi[j,i] = np.tan(45.*np.pi/180.)

						mu = self.rho[j] * self.vs[j] * self.vs[j]
						self.pco[j,i] = mu * 1e-4 * weight[ifault]#in Pa

					if method == 'closeness':
						self.phi[j,i] = 0.75

						s11 = self.sigma_xx[j,i]
						s22 = self.sigma_yy[j,i]
						s33 = self.sigma_zz[j,i]
						s13 = self.sigma_xz[j,i]
						self.pco[j,i]=cohesionfromcloseness([s11,s22,s33,0,s13,0],self.phi[j,i],closeness)

					if method == 'HoekBrown':
						#apply depth-dependent GSI
						if self.cdepth[j] < 1e3:
							self.GSI[j,i] = (100-self.tGSI)/1e3*(self.cdepth[j])+self.tGSI
						else:
							self.GSI[j,i] = 100

						self.mi[j,i] = self.tmi
						self.dci[j,i] = self.tdci

						mb = self.mi[j,i] * np.exp((self.GSI[j,i]-100)/(28 - 14*self.D[j,i]))
						s = np.exp((self.GSI[j,i]-100)/(9-3*self.D[j,i]))
						a = 0.5 + 1/6*(np.exp(-self.GSI[j,i]/15) - np.exp(-20/3))

						sigma_cm = self.dci[j,i]*((mb+4*s-a*(mb-8*s))*(mb/4+s)**(a-1))/(2*(1+a)*(2+a))
						
						# method - 1: use local vertical stress
						sigma_3max = sigma_cm*0.47*(sigma_cm/-self.sigma_yy[j,i])**(-0.94)

						# method - 2: use integral of above density
						# sigma_3max = sigma_cm*0.47*(sigma_cm/gh[j])**(-0.94)
						# print(sigma_3max)

						sigma_3n = sigma_3max/self.dci[j,i]

						self.pco[j,i] = (self.dci[j,i]*((1+2*a)*s+(1-a)*mb*sigma_3n)*(s+mb*sigma_3n)**(a-1))/\
						          ((1+a)*(2+a)*np.sqrt(1+(6*a*mb*(s+mb*sigma_3n)**(a-1))/((1+a)*(2+a))))
						self.phi[j,i] = np.tan(np.arcsin((6*a*mb*(s+mb*sigma_3n)**(a-1))/\
							       (2*(1+a)*(2+a)+6*a*mb*(s+mb*sigma_3n)**(a-1))))

						if self.cdepth[j] > 16e3:
							self.phi[j,i] = np.tan(89*np.pi/180)

						s11 = self.sigma_xx[j,i]
						s22 = self.sigma_yy[j,i]
						s33 = self.sigma_zz[j,i]
						s13 = self.sigma_xz[j,i]

						sbar,sy = plasticyieldsurface([s11,s22,s33,0,s13,0],self.phi[j,i],self.pco[j,i])
						if sbar > sy:
							print('{0} should be < {1} at {2} m'.format(sbar,sy,self.cdepth[j]))
		return


###################################################################
if __name__ == '__main__' and fromBBP:
	# try:
	# 	nx = sys.argv[1]
	# 	ny = sys.argv[2]
	# 	nz = sys.argv[3]

	# except:

################### initiate input module #####################
	# read in BBP module
	# load BBP2SORD data

	projd=loadBBP(file=fin+'proj.dat',format='proj');print(projd)
	irotate=float(projd['rotate'][0])
	iorigin = (float(projd['origin-lon'][0]), float(projd['origin-lat'][0]))

	proj = Init_coordinate(izone=projd['zone'].iloc[0],irotate=irotate,iorigin=iorigin)

	faultd=loadBBP(file=fin+'faultseg.dat',format='fault');print(faultd)
	stationd=loadBBP(file=fin+'station.dat',format='station');print(stationd)

	# convert station coordiante from lon/lat to x,z
	sta_x,sta_z = coordinate_conversion(stationd['lon'].to_numpy(),stationd['lat'].to_numpy(),\
				   (projd['leftc1'].iloc[0],projd['leftc2'].iloc[0]),proj)
	print('\n Conversion Station from Lon/Lat to x/z')
	for ist in range(len(sta_x)):
		print('{0} at x: {1} z: {2}'.format(stationd['name'].iloc[ist],sta_x[ist],sta_z[ist]))

	# fault segment length
	fault_length = table2numpy(faultd,['length',])
	fault_nnode = np.floor(fault_length/dx[0])+1

	# set up the whole domain
	# read nx ny nz hypo from Pointwise
	# now hypo is not accurate. only index normal to fault plane is accurate
	domain=SORDinput(nx,ny,nz,hypo,ifault,dx) 

	# strike of fault (geographic angle)
	fault_orientation = table2numpy(faultd,['strike',])
	print('Fault geographic strike: ', np.flip(fault_orientation.T))
	# proj = Init_coordinate(irotate=irotate,iorigin=iorigin) redundant

	# strike angle in transformed coordinate
	fault_orientation = geographic_angle_conversion(fault_orientation, proj)

	angle_sigma_1 = [33, 20, 11] #from south to north consistent with BBP
	angle_sigma_1 = geographic_angle_conversion(angle_sigma_1, proj)
	angle_sigma_1 = angle_sigma_1.reshape((np.shape(angle_sigma_1)[0],1))

	## fault flip (if different order between BBP and Pointwise mesh)
	# flip to from N to S (or from left to right)
	if faultflip:	
		fault_orientation = np.flip(fault_orientation)
		print('Fault orientation in new coordinate: %f %f %f'%tuple(fault_orientation))

		fault_length = np.flip(fault_length)
		print('Fault length in meter in new coordinate: %f %f %f'%tuple(fault_length))

		fault_nnode = np.flip(fault_nnode)
		print('Fault length in node in new coordinate: %f %f %f'%tuple(fault_nnode))

		angle_sigma_1 = np.flip(angle_sigma_1)
		print('Sigma_1 angle in new coordinate: %f %f %f\n'%tuple(angle_sigma_1))

	# print(fault_orientation,fault_length,fault_nnode)


##################################################################

	# set up fault segments
	domain.fault_segment(5,(leftnode,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(1)]))-1,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(2)]))-2,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(3)]))-3))
	print('True fault is from %d to %d (in total %d) \n' %\
		(domain.sege[0],domain.segb[-1],domain.segb[-1]-domain.sege[0]+1))
	if domain.sege[-1]-domain.sege[-2]+1 != rightnode:
		print('Error in setting up fault segments')
		exit()

	# compute hypocentral index in the computational domain
	length = faultd['length'][0]/2-faultd['hypo_along_stk'][0]
	hypo[0]= int(length/dx[0])+domain.segb[-2]
	hypo[1] = int(faultd['hypo_down_dip'][0]/dx[2])+1
	print('Hypocentral index is %d %d %d' %(hypo[0],hypo[1],hypo[2]))
	domain.hypo = hypo


##################################################################

	# set up initial depth profile
	domain.init_depth()#; print(domain.depth)


	## load velocity model of Mojave
	veld=loadBBP(file=fin+'Mojave.dat',format='velmodel');print(veld)
	domain.BBPvel(veld)
	PlotFigure(1,np.array([domain.vp,domain.vs,domain.rho]).T,(domain.ny-1)*dx[2]/1e3,fig+'velmodel.png')

	# set up vertical stress (sigma_zz or simga_2)
	domain.vertical_stress(offset=offset_stress)#; print(domain.sigma_yy)
	sigma_2 = domain.sigma_y
	# print(np.shape(simga_2))

	# set up horizontal stress (sigma_1 and sigma_3)
	# 


	print('fault_orientation',fault_orientation.transpose())
	print('angle_sigma_1',angle_sigma_1.transpose())

	# compute principal stress (sigma_1 and sigma_3)
	sigma_1,sigma_3,sigma_xx,sigma_zz,sigma_xz, ts, tn = \
			  domain.horizontal_stress_seg(fault_orientation,angle_sigma_1,seglist,\
								 sigma_2, Ratio, fco,\
								 mus, mud)
	sigma_xx = extend_edge(sigma_xx,'left',domain.segb[1])
	sigma_xx = extend_edge(sigma_xx,'right',domain.segb[-1])

	sigma_zz = extend_edge(sigma_zz,'left',domain.segb[1])
	sigma_zz = extend_edge(sigma_zz,'right',domain.segb[-1])

	sigma_xz = extend_edge(sigma_xz,'left',domain.segb[1])
	sigma_xz = extend_edge(sigma_xz,'right',domain.segb[-1])

	ts = extend_edge(ts,'left',domain.segb[1])
	ts = extend_edge(ts,'right',domain.segb[-1])
	tn = extend_edge(tn,'left',domain.segb[1])
	tn = extend_edge(tn,'right',domain.segb[-1])

	domain.assign_stress(sigma_xx,sigma_zz,sigma_xz, ts, tn)

	if readroughfault:
		print('\n Rough fault module starts\n')

		start=domain.segb[1]
		end = domain.sege[-2]
		faultnx = end-start+1
		rnhat_x=readbin(rough+'rnhat_x',(domain.ny,faultnx),inputdtype="float64")
		rnhat_y=readbin(rough+'rnhat_y',(domain.ny,faultnx),inputdtype="float64")
		rnhat_z=readbin(rough+'rnhat_z',(domain.ny,faultnx),inputdtype="float64")

		# this will update domain.ts and domain.tn
		domain.rough_ts_tn(rnhat_x,rnhat_y,rnhat_z) 

		# find max value
		tstnratio = (domain.ts[:,start+1:end+1]/domain.tn[:,start+1:end+1])

		for im in range(3):
			start=domain.segb[1+im]-domain.segb[1]
			end = domain.sege[1+im]-domain.segb[1]

			maxtstnratio = tstnratio[:,start:end].max()
			# print(maxtstnratio)
			if maxtstnratio > musf[im]:
				print('\nmusf has been changed on # {0} segment of the fault\n'.format(im))
				musf[im] = maxtstnratio*1.01


	PlotFigure(2,domain.sigma_zz,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'zz.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_xx,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'xx.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_xz,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'xz.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_yy,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'yy.png',domain.segb[1:])

	# writebin(ftmp+'ts',domain.ts) #2D
	# writebin(ftmp+'tn',domain.tn) #2D
	PlotFigure(2,domain.ts,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'ts.png',domain.segb[1:])
	PlotFigure(2,domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'tn.png',domain.segb[1:])


	# print(domain.vs)

	# write binary files
	writebin(fname+'sigma_xx.bin',domain.sigma_xx) #2D
	writebin(fname+'sigma_zz.bin',domain.sigma_zz) #2D
	writebin(fname+'sigma_xz.bin',domain.sigma_xz) #2D
	writebin(fname+'sigma_yy.bin',domain.sigma_yy) #2D
	writebin(fname+'vp.bin',domain.vp)   #1D
	writebin(fname+'vs.bin',domain.vs)   #1D
	writebin(fname+'rho.bin',domain.rho) #1D


	## smooth the sharp boundary
	# domain.ts=smooth2D(domain.ts,sigma=1)
	# domain.tn=smooth2D(domain.tn,sigma=1)

	# extend to 2D plane
	# pyy = domain.sigma_yy.reshape((domain.ny-1,1)).repeat(domain.nx-1,axis=1)	
	# pvp = domain.vp.reshape((domain.ny-1,1)).repeat(domain.nx-1,axis=1)
	# pvs = domain.vp.reshape((domain.ny-1,1)).repeat(domain.nx-1,axis=1)
	# prho = domain.rho.reshape((domain.ny-1,1)).repeat(domain.nx-1,axis=1)
	# writebin(ftmp+'fault_sigma_yy',pyy)
	# writebin(ftmp+'fault_vp',pvp)
	# writebin(ftmp+'fault_vs',pvs)
	# writebin(ftmp+'fault_rho',prho)
	# PlotFigure(2,pyy,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pyy.png')
	# PlotFigure(2,pvp,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pvp.png')
	# PlotFigure(2,pvs,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pvs.png')
	# PlotFigure(2,prho,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'prho.png')


	# read mesh from Pointwise
	if readmesh:
		if readroughfault:
			xx=readbin(rough+'surface_x1',(domain.nz,domain.nx),inputdtype="float64")
			zz=readbin(rough+'surface_x3',(domain.nz,domain.nx),inputdtype="float64") #in our coordinate should be zz
		else:
			xx=readbin(fin+'surface_x1',(domain.nz,domain.nx),inputdtype="float64")
			zz=readbin(fin+'surface_x3',(domain.nz,domain.nx),inputdtype="float64") #in our coordinate should be zz

		#request station index in computational domain
		if searchstation:
			domain.stationindex(xx,zz,sta_x,sta_z)
			for i in range(len(stationd)):
				print('{0} [{1:.3f}, 1, {2:.3f}]'.format(stationd['name'].iloc[i],domain.stationidx[i,0],domain.stationidx[i,1]))



	#verify fault plane
		fx = xx[hypo[2]-1,:]
		fz = zz[hypo[2]-1,:]
		fx2 = xx[hypo[2],:]
		fz2 = zz[hypo[2],:]

		print('Fault plane error check: {0:10.8e} (shoul be <1e-3)'.format( np.sum(abs(fx-fx2)+abs(fz-fz2) )))

	#smooth fault geometry 
	# fx = smooth1D(fx,window_len=10,window='flat')
	# fz = smooth1D(fz,window_len=10,window='flat')

	#seg 2
	# b=domain.segb[1]-1
	# e=domain.sege[1]
	# zz[hypo[2],b:e] = point2line([xx[hypo[2],b],zz[hypo[2],b]],\
	# 							 [xx[hypo[2],e],zz[hypo[2],e]],\
	# 							  xx[hypo[2],b:e])
	# zz[hypo[2]-1,b:e] = zz[hypo[2],b:e]

	# #seg 3
	# b=domain.segb[2]-1
	# e=domain.sege[2]
	# zz[hypo[2],b:e] = point2line([xx[hypo[2],b],zz[hypo[2],b]],\
	# 							 [xx[hypo[2],e],zz[hypo[2],e]],\
	# 							  xx[hypo[2],b:e])
	# zz[hypo[2]-1,b:e] = zz[hypo[2],b:e]

	# #seg 4
	# b=domain.segb[3]-1
	# e=domain.sege[3]
	# zz[hypo[2],b:e] = point2line([xx[hypo[2],b],zz[hypo[2],b]],\
	# 							 [xx[hypo[2],e],zz[hypo[2],e]],\
	# 							  xx[hypo[2],b:e])
	# zz[hypo[2]-1,b:e] = zz[hypo[2],b:e]


		writebin(fname+'xx.bin',xx) #2D
		writebin(fname+'zz.bin',zz) #2D
	# fault plane geometry
		# pfx = xx[domain.hypo[2]-1,:].reshape((1,domain.nx)).repeat(domain.ny,axis=0)
		# pfx = pfx.reshape((1,ny,nx)).repeat(2,axis=0)
		# pfz = zz[domain.hypo[2]-1,:].reshape((1,domain.nx)).repeat(domain.ny,axis=0)
		# pfz = pfz.reshape((1,ny,nx)).repeat(2,axis=0)
		# pfy = domain.ndepth.reshape((domain.ny,1)).repeat(domain.nx,axis=1)
		# pfy = pfy.reshape((1,ny,nx)).repeat(2,axis=0)
		# writebin(ftmp+'fault_x',pfx,'<f4')
		# writebin(ftmp+'fault_z',pfz,'<f4')
		# writebin(ftmp+'fault_y',pfy,'<f4')
	# PlotFigure(2,pfx,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pfx.png')
	# PlotFigure(2,pfz,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pfz.png')
	# PlotFigure(2,pfy,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,'pfy.png')




	## friction parameters
	#
	if musf is not None:
		domain.friction_seg(seglist,fco,musf,mud,dc)
	else:
		domain.friction_seg(seglist,fco,mus,mud,dc)
	domain.fco = extend_edge(domain.fco,'left',domain.segb[1]+1)
	domain.fco = extend_edge(domain.fco,'right',domain.segb[-1])
	domain.mus = extend_edge(domain.mus,'left',domain.segb[1]+1)
	domain.mus = extend_edge(domain.mus,'right',domain.segb[-1])
	domain.mud = extend_edge(domain.mud,'left',domain.segb[1]+1)
	domain.mud = extend_edge(domain.mud,'right',domain.segb[-1])
	domain.dc = extend_edge(domain.dc,'left',domain.segb[1]+1)
	domain.dc = extend_edge(domain.dc,'right',domain.segb[-1])



	#add kink high friction patch
	if kinkoff:
		for ik in range(len(kinklist)):
			print(kextent[ik,:])
			kl=int(domain.segb[kinklist[ik]]-1-kextent[ik,0]/2) #left 
			kr=int(domain.segb[kinklist[ik]]-1+kextent[ik,0]/2) #right
			kt=int(kextent[ik,2]   )							#top
			kb=int(kt+kextent[ik,1])							#bottom

			nkl = (kr - kl + 1)
			nkw = (kb - kt + 1)

			kmus = np.ones((nkw,nkl)) * kextent[ik,3] 
			kmud = np.ones((nkw,nkl)) * kextent[ik,4]

			kmus = kmus * taper2D([(nkw-1)*dx[1],(nkl-1)*dx[0]],\
				   kinktaper,[dx[0],dx[1]],inverse=False) #inverse=True means large at boundary
			kmud = kmud * taper2D([(nkw-1)*dx[1],(nkl-1)*dx[0]],\
				   kinktaper,[dx[0],dx[1]],inverse=False)

			domain.mus[kt:kb+1,kl:kr+1] = domain.mus[kt:kb+1,kl:kr+1] + kmus
			domain.mud[kt:kb+1,kl:kr+1] = domain.mud[kt:kb+1,kl:kr+1] + kmud

			print('\nAdding kink-related friction asperity\n')


	# coseismic segments
	upindex = 0
	downindex = domain.ny-1
	leftindex = domain.segb[seglist[0]-1]-1+nbdl
	rightindex = domain.sege[seglist[-1]-1]-nbdr-1


	print('\ncoseismic segments index range: {0}-{1}\n'.format(leftindex,rightindex))



	#random function
	# mus_a = domain.mus.mean()
	# mud_a = domain.mud.mean()
	if randomfield:
		W = (downindex - upindex )*dx[1]
		L = (rightindex - leftindex )*dx[0]
		srcpar = [W, L]

		if round(W/samp[1]) %2 == 1:
			print('Index number is coseismic depth {0} should be even, +-dx'.format(W))
			exit()
		if round(L/samp[0]) %2 == 1:
			print('Index number is coseismic length {0} should be even, +-dx'.format(L))
			exit()



	#coarse random field
		G,spar=randomfieldspecdistr(srcpar,acf,corr,seed,samp,grd,nexp,[],depth,dip,nfig,outfile)
		G = G-G.min()
		Fc = G/G.max()*(hgv-lov)+lov

		xc = np.arange(0,L+samp[1],samp[1])
		yc = np.arange(0,W+samp[0],samp[0])

		xf = np.arange(0,L+dx[0],dx[0])
		yf = np.arange(0,W+dx[0],dx[1])

		# print(xc.shape,yc.shape,Fc.shape)

		f = interpolate.interp2d(xc, yc, Fc, kind='linear')
		Ff = f(xf,yf)

		if kinkoff:
			for ik in range(len(kinklist)):
				print(kextent[ik,:])
				kl=int(domain.segb[kinklist[ik]]-1-kextent[ik,0]/2)-leftindex #left 
				kr=int(domain.segb[kinklist[ik]]-1+kextent[ik,0]/2)-leftindex #right
				kt=int(kextent[ik,2]   ) - upindex							#top
				kb=int(kt+kextent[ik,1]) - upindex
				temp=Ff[kt:kb+1,kl:kr+1]
				temp=np.where(temp>0, -temp, temp) 
				temp = temp + (klov-temp.min())
				Ff[kt:kb+1,kl:kr+1] = temp

		musbar = domain.mus[upindex:downindex,leftindex:rightindex+1].copy()
		tstn = domain.ts[upindex:downindex,leftindex:rightindex+1]/\
		   	domain.tn[upindex:downindex,leftindex:rightindex+1]
		domain.mus[upindex:downindex,leftindex:rightindex+1] = \
			musbar +(tstn-musbar)* Ff[:-1,:]+musoffset

		domain.mud[upindex:downindex,leftindex:rightindex+1] = \
			domain.mud[upindex:downindex,leftindex:rightindex+1].copy()+\
			domain.mus[upindex:downindex,leftindex:rightindex+1] - musbar + mudoffset




	nL = rightindex - leftindex 
	nW = downindex - upindex
	
	print('Plot fault range: x({0}-{1}),y({2}-{3})'.format(leftindex+1,rightindex,upindex+1,downindex))

	ftaper=taper2D([nW*dx[1],nL*dx[0]],\
		# [taper_layer,0,taper_layer],[dx[0],dx[1]],inverse=True)
		# change ny also change here
		taperlist,[dx[0],dx[1]],inverse=True) #inverse=True means large at boundary

	ftaper = ftaper * 1e5
	domain.mus[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.mus[upindex:downindex+1,leftindex:rightindex+1] + ftaper 
	domain.mud[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.mud[upindex:downindex+1,leftindex:rightindex+1] + ftaper 
	domain.dc[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.dc[upindex:downindex+1,leftindex:rightindex+1] + ftaper 

	domain.mus = extend_edge(domain.mus,'left',leftindex+1)
	domain.mus = extend_edge(domain.mus,'right',rightindex)
	domain.mud = extend_edge(domain.mud,'left',leftindex+1)
	domain.mud = extend_edge(domain.mud,'right',rightindex)
	domain.dc = extend_edge(domain.dc,'left',leftindex+1)
	domain.dc = extend_edge(domain.dc,'right',rightindex)


	# generate ramdom fco
	if fcrandom:
		W = (downindex - upindex )*dx[1]
		L = (rightindex - leftindex )*dx[0]
		srcpar = [W, L]

		c1=fcdepth		  # corr=[c1, c2, Hn]=  array of correlation length [ay ax H] [vertical and horizontal]
		c2=3e3			   #
		n=0.5			 # Hn= Hurst number
		corr = [c1,c2,Hn]

		seed = None
		# seed = 1588487981
		sa1=dx[0]			# Along the strike sampling
		sa2=dx[1]		# Down dip sampling
		samp=[sa1*interp_ratio, sa2*interp_ratio]

		G,spar=randomfieldspecdistr(srcpar,'ak',corr,seed,samp,'nod',1,[],25e3,90,'n','n')
		G = G-G.min()
		Fc = G/G.max()*(fcotop)

		xc = np.arange(0,L+samp[1],samp[1])
		yc = np.arange(0,W+samp[0],samp[0])

		xf = np.arange(0,L+dx[0],dx[0])
		yf = np.arange(0,W+dx[0],dx[1])

		# print(xc.shape,yc.shape,Fc.shape)

		f = interpolate.interp2d(xc, yc, Fc, kind='linear')
		Ff = f(xf,yf)
		domain.fco[upindex:downindex,leftindex:rightindex+1] = \
			domain.fco[upindex:downindex,leftindex:rightindex+1]* Ff[:-1,:]






	PlotFigure(2,domain.fco,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'fco.png',domain.segb[1:])
	PlotFigure(2,domain.mus,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'mus.png',domain.segb[1:],0.2,1)
	PlotFigure(2,domain.mud,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'mud.png',domain.segb[1:],0.2,1)
	PlotFigure(2,domain.dc,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'dc.png',domain.segb[1:],0,1)


	writebin(fname+'fco.bin',domain.fco) #2D
	writebin(fname+'mus.bin',domain.mus) #2D
	writebin(fname+'mud.bin',domain.mud) #2D
	writebin(fname+'dc.bin',domain.dc) #2D

	## plot potential stress drop
	PlotFigure(2,domain.ts/domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'tstn.png',domain.segb[1:],0,1)
	PlotFigure(2,domain.mus[:-1,:-1]*domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'strength.png',domain.segb[1:])
	stressdrop = domain.ts-domain.mud[:-1,:-1]*domain.tn
	stressdrop = np.where(stressdrop<0,np.nan,stressdrop)
	PlotFigure(2,stressdrop,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'stressdrop.png',domain.segb[1:],0,10e6)

	# generate depth-variable plasticity model
	domain.plasticity2D(seglist,[1,1,1],plmodel,plvar) #'Horsrud2001','Chang2006', 'Roten2014', 'HoekBrown'
	if use_faultzone:
		print('\nFault Zone index range z,y,x',faultzone_range)
		domain.vs_fault_zone3D(faultzone_vs,faultzone_range)
		domain.GSI_fault_zone3D(faultzone_GSI,faultzone_range)
		writebin(fname+'fz_vs.bin', domain.faultzone_vs) #3D
		writebin(fname+'fz_phi.bin',domain.faultzone_phi) #3D
		writebin(fname+'fz_pco.bin',domain.faultzone_pco) #3D

		print('Range of fz_vs',domain.faultzone_vs.min(),domain.faultzone_vs.max())
		print('Range of fz_phi',domain.faultzone_phi.min(),domain.faultzone_phi.max())
		print('Range of fz_pco',domain.faultzone_pco.min(),domain.faultzone_pco.max())

	domain.pco = extend_edge(domain.pco,'left',leftindex+1)
	domain.pco = extend_edge(domain.pco,'right',rightindex-1)
	domain.phi = extend_edge(domain.phi,'left',leftindex+1)
	domain.phi = extend_edge(domain.phi,'right',rightindex-1)
	PlotFigure(2,domain.pco,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'pco.png',domain.segb[1:])
	PlotFigure(2,domain.phi,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'phi.png',domain.segb[1:])

	writebin(fname+'phi.bin',domain.phi) #2D
	writebin(fname+'pco.bin',domain.pco) #2D

	print('Range of pco',domain.pco.min(),domain.pco.max())
	print('Range of phi',domain.phi.min(),domain.phi.max())

	PlotFigure(1,domain.pco[:,0],(domain.ny-1)*dx[2]/1e3,fig+'pco.png')
	PlotFigure(1,np.arctan(domain.phi[:,0])*180/np.pi,(domain.ny-1)*dx[2]/1e3,fig+'phi.png')
	# PlotFigure(1,\
	# 	np.array([domain.mus[:,leftindex+100],domain.mud[:,leftindex+100],domain.dc[:,leftindex+100]]).T,\
	# 	(domain.ny-1)*dx[2]/1e3,fig+'friction.png')


















































elif __name__ == '__main__' and not fromBBP:

	## initiate model parameters

	domain=SORDinput(nx,ny,nz,hypo,ifault,dx) 

	fault_length=[33e3,23e3,23e3] #if faultflip is true, this order is from right side to the left.
	fault_length=np.array(fault_length)
	fault_nnode = np.floor(fault_length/dx[0])+1	

	fault_orientation = [270,270,270] #segment consistent with BBP
	angle_sigma_1 = [315,315,315]   #segment


	if faultflip:	
		fault_orientation = np.flip(fault_orientation)
		print('Fault orientation in new coordinate: %f %f %f'%tuple(fault_orientation))

		fault_length = np.flip(fault_length)
		print('Fault length in meter in new coordinate: %f %f %f'%tuple(fault_length))

		fault_nnode = np.flip(fault_nnode)
		print('Fault length in node in new coordinate: %f %f %f'%tuple(fault_nnode))

		angle_sigma_1 = np.flip(angle_sigma_1)
		print('Sigma_1 angle in new coordinate: %f %f %f\n'%tuple(angle_sigma_1))

	# set up fault segments
	domain.fault_segment(5,(leftnode,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(1)]))-1,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(2)]))-2,\
						   leftnode+int(np.sum([fault_nnode[i] for i in range(3)]))-3))
	print('True fault is from %d to %d (in total %d) \n' %\
		(domain.sege[0],domain.segb[-1],domain.segb[-1]-domain.sege[0]+1))

	print('Hypocentral index is %d %d %d' %(hypo[0],hypo[1],hypo[2]))
	domain.hypo = hypo

	# set up initial depth profile
	domain.init_depth()#; print(domain.depth)

	#generate customized velocity model
	if domain.customvel():
		print('Customized velocity model has been enabled')

	# or we can use velocity table for it
	#velocity model
	# veld=pandas.DataFrame({'thickness': np.array([10,1e10]),\
	# 					   'vp': np.array([6000,6000]),\
	# 					   'vs': np.array([3464,3464]),\
	# 					   'rho': np.array([2600,2600]),\
	# 					   'qp': np.array([500,800]),\
	# 					   'qs': np.array([500,800]),\
	# 					  })
	# print(veld)
	# domain.BBPvel(veld)


	# set up vertical stress (sigma_zz or simga_2)
	domain.vertical_stress(offset=offset_stress,rhow=1000)#; print(domain.sigma_yy)
	sigma_2 = domain.sigma_y


	# compute principal stress (sigma_1 and sigma_3)
	sigma_1,sigma_3,sigma_xx,sigma_zz,sigma_xz, ts, tn = \
			  domain.horizontal_stress_seg(fault_orientation,angle_sigma_1,seglist,\
								 sigma_2, Ratio, fco,\
								 mus, mud)
	sigma_xx = extend_edge(sigma_xx,'left',domain.segb[1])
	sigma_xx = extend_edge(sigma_xx,'right',domain.segb[-1])

	sigma_zz = extend_edge(sigma_zz,'left',domain.segb[1])
	sigma_zz = extend_edge(sigma_zz,'right',domain.segb[-1])

	sigma_xz = extend_edge(sigma_xz,'left',domain.segb[1])
	sigma_xz = extend_edge(sigma_xz,'right',domain.segb[-1])

	ts = extend_edge(ts,'left',domain.segb[1])
	ts = extend_edge(ts,'right',domain.segb[-1])
	tn = extend_edge(tn,'left',domain.segb[1])
	tn = extend_edge(tn,'right',domain.segb[-1])

	domain.assign_stress(sigma_xx,sigma_zz,sigma_xz, ts, tn)


	# write binary files
	writebin(fname+'sigma_xx.bin',domain.sigma_xx) #2D
	writebin(fname+'sigma_zz.bin',domain.sigma_zz) #2D
	writebin(fname+'sigma_xz.bin',domain.sigma_xz) #2D
	writebin(fname+'sigma_yy.bin',domain.sigma_yy) #1D
	writebin(fname+'vp.bin',domain.vp)   #1D
	writebin(fname+'vs.bin',domain.vs)   #1D
	writebin(fname+'rho.bin',domain.rho) #1D



	## friction parameters
	#
	domain.friction_seg(seglist,fco,mus,mud,dc)
	domain.fco = extend_edge(domain.fco,'left',domain.segb[1]+1)
	domain.fco = extend_edge(domain.fco,'right',domain.segb[-1])
	domain.mus = extend_edge(domain.mus,'left',domain.segb[1]+1)
	domain.mus = extend_edge(domain.mus,'right',domain.segb[-1])
	domain.mud = extend_edge(domain.mud,'left',domain.segb[1]+1)
	domain.mud = extend_edge(domain.mud,'right',domain.segb[-1])
	domain.dc = extend_edge(domain.dc,'left',domain.segb[1]+1)
	domain.dc = extend_edge(domain.dc,'right',domain.segb[-1])

	# coseismic segments
	upindex = 0
	downindex = domain.ny-1
	leftindex = domain.segb[seglist[0]-1]-1+nbdl
	rightindex = domain.sege[seglist[-1]-1]-nbdr-1

	nL = rightindex - leftindex 
	nW = downindex - upindex
	
	print('Plot fault range: x({0}-{1}),y({2}-{3})'.format(leftindex+1,rightindex,upindex+1,downindex))



	#random function
	if randomfield:
		W = (downindex - upindex )*dx[1]
		L = (rightindex - leftindex )*dx[0]
		srcpar = [W, L]
	#coarse random field
		G,spar=randomfieldspecdistr(srcpar,acf,corr,seed,samp,grd,nexp,[],depth,dip,nfig,outfile)
		G = G-G.min()
		Fc = G/G.max()*(hgv-lov)+lov


		xc = np.arange(0,L+samp[1],samp[1])
		yc = np.arange(0,W+samp[0],samp[0])

		xf = np.arange(0,L+dx[0],dx[0])
		yf = np.arange(0,W+dx[0],dx[1])


		f = interpolate.interp2d(xc, yc, Fc, kind='linear')
		Ff = f(xf,yf)

		print('\n\n Ff:   MAX={0} MIN={1}\n\n'.format(Ff.max(),Ff.min()))

		musbar = domain.mus[upindex:downindex,leftindex:rightindex+1].copy()
		tstn = domain.ts[upindex:downindex,leftindex:rightindex+1]/\
		   	domain.tn[upindex:downindex,leftindex:rightindex+1]
		domain.mus[upindex:downindex,leftindex:rightindex+1] = \
			musbar +(tstn-musbar)* Ff[:-1,:]+musoffset

		domain.mud[upindex:downindex,leftindex:rightindex+1] = \
			domain.mud[upindex:downindex,leftindex:rightindex+1].copy()+\
			domain.mus[upindex:downindex,leftindex:rightindex+1] - musbar + mudoffset


		print(' mus:   MAX={0} MIN={1}\n'.format(domain.mus.max(),domain.mus.min()))
		print(' mud:   MAX={0} MIN={1}\n\n'.format(domain.mud.max(),domain.mud.min()))


	ftaper=taper2D([nW*dx[1],nL*dx[0]],\
		# [taper_layer,0,taper_layer],[dx[0],dx[1]],inverse=True)
		# change ny also change here
		taperlist,[dx[0],dx[1]],inverse=True) #inverse=True means large at boundary

	ftaper = ftaper * 1e5
	domain.mus[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.mus[upindex:downindex+1,leftindex:rightindex+1] + ftaper 
	domain.mud[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.mud[upindex:downindex+1,leftindex:rightindex+1] + ftaper 
	domain.dc[upindex:downindex+1,leftindex:rightindex+1] = \
		 domain.dc[upindex:downindex+1,leftindex:rightindex+1] + ftaper 

	domain.mus = extend_edge(domain.mus,'left',leftindex+1)
	domain.mus = extend_edge(domain.mus,'right',rightindex)
	domain.mud = extend_edge(domain.mud,'left',leftindex+1)
	domain.mud = extend_edge(domain.mud,'right',rightindex)
	domain.dc = extend_edge(domain.dc,'left',leftindex+1)
	domain.dc = extend_edge(domain.dc,'right',rightindex)

	writebin(fname+'fco.bin',domain.fco) #2D
	writebin(fname+'mus.bin',domain.mus) #2D
	writebin(fname+'mud.bin',domain.mud) #2D
	writebin(fname+'dc.bin',domain.dc) #2D

	# generate depth-variable plasticity model
	domain.plasticity2D(seglist,[1,1,1],plmodel,plvar) #'Horsrud2001','Chang2006', 'Roten2014', 'HoekBrown'
	domain.pco = extend_edge(domain.pco,'left',leftindex+1)
	domain.pco = extend_edge(domain.pco,'right',rightindex-1)
	domain.phi = extend_edge(domain.phi,'left',leftindex+1)
	domain.phi = extend_edge(domain.phi,'right',rightindex-1)

	writebin(fname+'phi.bin',domain.phi) #2D
	writebin(fname+'pco.bin',domain.pco) #2D

	print('Range of pco',domain.pco.min(),domain.pco.max())
	print('Range of phi',domain.phi.min(),domain.phi.max())




	#plot module
	PlotFigure(2,domain.sigma_zz,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'zz.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_xx,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'xx.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_xz,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'xz.png',domain.segb[1:])
	PlotFigure(2,domain.sigma_yy,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'yy.png',domain.segb[1:])

	PlotFigure(2,domain.ts,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'ts.png',domain.segb[1:])
	PlotFigure(2,domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'tn.png',domain.segb[1:])
	PlotFigure(1,np.array([-domain.sigma_xx[:,800],-domain.sigma_yy[:,800],-domain.sigma_zz[:,800],domain.sigma_xz[:,800]]).T,\
		(domain.ny-1)*dx[2]/1e3,fig+'stressall.png')
	PlotFigure(1,np.array([-domain.sigma_zz[:,800]*0.1,-domain.sigma_zz[:,800]*0.2,-domain.sigma_zz[:,800]*0.3,-domain.sigma_zz[:,800]*0.4,\
		-domain.sigma_zz[:,800]*0.5,-domain.sigma_zz[:,800]*0.6,-domain.sigma_zz[:,800]*0.7,-domain.sigma_zz[:,800]*0.8,-domain.sigma_zz[:,800]*0.9,\
		domain.sigma_xz[:,800],-domain.sigma_zz[:,800]*domain.mud[:-1,800],\
		-domain.sigma_zz[:,800]*domain.mus[:-1,800]]).T,\
		(domain.ny-1)*dx[2]/1e3,fig+'initialstress.png',0,200e6)
	PlotFigure(2,domain.fco,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'fco.png',domain.segb[1:])
	PlotFigure(2,domain.mus,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'mus.png',domain.segb[1:],0,1)
	PlotFigure(2,domain.mud,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'mud.png',domain.segb[1:],0,1)
	PlotFigure(2,domain.dc,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'dc.png',domain.segb[1:],0,1)
	PlotFigure(2,domain.ts/domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'tstn.png',domain.segb[1:],0,1)
	PlotFigure(2,domain.mus[:-1,:-1]*domain.tn,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'strength.png',domain.segb[1:])
	stressdrop = domain.ts-domain.mud[:-1,:-1]*domain.tn
	stressdrop = np.where(stressdrop<0,np.nan,stressdrop)
	PlotFigure(2,stressdrop,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'stressdrop.png',domain.segb[1:],-5,25e6)
	PlotFigure(1,np.array([domain.vp,domain.vs,domain.rho]).T,(domain.ny-1)*dx[2]/1e3,fig+'velmodel.png')
	PlotFigure(2,domain.pco,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'pco.png',domain.segb[1:])
	PlotFigure(2,domain.phi,(domain.ny-1)*dx[2]/1e3,(domain.nx-1)*dx[0]/1e3,fig+'phi.png',domain.segb[1:])
	PlotFigure(1,domain.pco[:,0],(domain.ny-1)*dx[2]/1e3,fig+'pco.png')
	PlotFigure(1,np.arctan(domain.phi[:,0])*180/np.pi,(domain.ny-1)*dx[2]/1e3,fig+'phi.png')
	#end


