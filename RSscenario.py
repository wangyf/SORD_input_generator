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
import pyandrewsbarall as pyab

debug=False
idtype = np.dtype( 'f8' ).str #'<f8'
slipvector=np.array([1,0,0])  #compatible with SORD

###################################################################
class param:
	def __init__(self):
		self.generatedir=True
		self.rundir=os.path.join(os.getcwd(),'tmp')
		# self.rundir='/home1/04584/tg838839/scratch/GenericStrikeSlip_SW3/'
		# self.rundir='/home1/04584/tg838839/scratch/GenericStrikeSlip_SW/' #this is for Frontera
		self.T=15
		self.np3=2,2,1
		self.custom_length=None
		self.custom_width=None
		self.scenario_module = 'andrews_barall' #generic, andrews_barall
		self.mag = 7.4
		self.cutratio = 4

		# global variable
		self.writebin=True
		self.figplot=True
		self.sordfolder_only=True
		self.fbin=os.path.join(os.getcwd(),'bin/')
		self.fig=os.path.join(os.getcwd(),'fig/')

		self.leftnode = 100   #left node distance to fault
		self.rightnode = 100 #right node distance to fault

		self.nx = 2001
		self.ny = 401
		self.nz = 802
		self.hypo = [601,251,401]		
		self.dx = [50, 50, 50]
		self.ifault = 3

		#velocity cap in the shallow depth
		self.velcap=False
		self.velcapdep=1e3

		# offset vertical stress at surface
		self.constant_gradient = True #constant depth gradient of normal stress (17MPa/km)
		self.offset_stress = 1.0e6
		self.depsat = 5.e3
		self.psi=45.
		# self.ratio=0.36 #constant
		self.ratio = ['depvar',[0,0.33],[1e3,0.33],[5e3,0.33],[1000e3,0.33]]
		#self.ratio=['ak',[3e3,10e3,0.8],0.1,0.4,None] #[vertical,horizontal correlation length]

		try:
			os.mkdir(self.fbin) 
		except:
			pass	

		#friction set up
		self.friction='RS'
		# rate and state friction
		# constants
		self.b=0.014
		self.v0 = 1e-6
		# self.f0 = 0.6
		# self.L = 0.2
		# self.fw = 0.3
		# self.vw = 0.1
		# dep variable (defined by turning point [dep,value])
		self.fw=[[0,1],     [1e3,0.22],    [5e3,0.22], [20e3,0.22]] 
		self.f0=[[0,1],     [1e3,0.6],    [5e3,0.6], [20e3,0.6]] 
		self.a= [[0,0.02],  [1e3,0.01],   [5e3,0.01],[20e3,0.01]] 
		self.vw=[[0,2],     [1e3,0.1],      [5e3,0.1], [20e3,0.1]]
		self.L =[[0,2],     [1e3,0.2],      [5e3,0.2], [20e3,0.2]]
		# self.v0=[[0,1e-10], [1e3,1e-6],  [5e3,1e-6], [20e3,1e-6]]

		self.friction='SW'
		# slip weakening friction
		self.mus = 0.5
		self.mud = 0.25
		self.dc =  [[0,0.8],       [3e3,0.4], [200e3,0.4]]
		self.fco = [[0,1e6],     [3e3,0], [200e3,0]]
		self.checkroughness = False, 'in/'

		
		self.taper_layer = 5e3
		self.taperlist = [self.taper_layer,0,self.taper_layer] #used for friction layer


		#random field for b parameter
		self.randomfield=False
		self.interp_ratio = 1
		# ## random field
		self.acf='ak'			 # autocorrelation function
		self.c1=3e3			   # corr=[c1, c2, Hn]=  array of correlation length [ay ax H] [vertical and horizontal]
		self.c2=3e3			   #
		self.Hn=0.6			 # Hn= Hurst number
		self.corr = [self.c1,self.c2,self.Hn]
		self.seed = None
		# self.seed = 1588487981
		self.seed=None
		self.sa1=self.dx[0]			# Along the strike sampling
		self.sa2=self.dx[1]		# Down dip sampling
		self.samp=[self.sa1*self.interp_ratio, self.sa2*self.interp_ratio]
		self.lov=-0.005
		self.hgv=0.005


		#plasticity
		# self.plmodel = 'closeness' #'Horsrud2001' 'Chang2006', 'Roten2014', 'HoekBrown','closeness'
		# make='average'        #very good, good, average poor only for HoekBrown # 'local' or 'fixed' for closeness
		# self.closeness = [[0,0.50], [3e3,0.20],[5e3,0.10], [10e3,0.1], [11e3,0.01],[100e3,0.01]]
		# self.close_phi = [[0,0.8],  [3e3,0.75], [5e3,0.6], [10e3,0.6], [100e3,0.6]]
		# self.plvar = make   #if self.closeness is not none, this is the value
		# self.weight  = [1,1,1]
		# self.mis=self.closeness,self.close_phi
		self.plmodel = 'HoekBrown' #'Horsrud2001' 'Chang2006', 'Roten2014', 'HoekBrown','closeness'
		make='good'        #very good, good, average poor only for HoekBrown	
		self.plvar = make   #if self.closeness is not none, this is the value
		self.weight  = [1,1,1]
		self.mis=None,None
		self.check_yieldsurface=True
########################################

def single_run():
	#initiate parameter and domain
	p=param()

	passrun=0
	while True:

		p,domain = module_init(p)
		if p.sordfolder_only:
			break

	
		domain=module_vel(p,domain)
	
		# set up stress field
		domain=module_stress(p,domain)
	
		# set up RS friction
		if p.friction == 'RS':
			domain=module_RSF(p,domain)
		elif p.friction == 'SW':
			domain=module_SWF(p,domain)
	
		# set up plasticity parameters
		domain=module_plastic(p,domain)

		if domain.plasticitypass:
			break
		passrun+=1

		if passrun>10:
			sys.exit('Plasticity module may be wrong ')

	if p.generatedir:
		copy_binary(p.rundir,p.fbin)
		# submit_queue(p.rundir)
	return

def multi_run():
	for iim in range(55,66):
		im=iim/10.0000
		# for realization 
		for ireal in range(1,6):
			
			#initiate parameter and domain
			p=param()

			p.mag = im
			ll,_ = pyab.fault_size(p.mag)
			p.T = np.ceil(ll/3000.)*2
			p.rundir=os.path.join(p.rundir,'M{0:3.1f}_{1:03d}'.format(im,ireal))
			p.np3 = 8,8,7

			passrun=0
			while True:
				p,domain = module_init(p)
				if p.sordfolder_only:
					break

				domain=module_vel(p,domain)
		
				# set up stress field
				domain=module_stress(p,domain)
		
				# set up RS friction
				if p.friction == 'RS':
					domain=module_RSF(p,domain)
				elif p.friction == 'SW':
					domain=module_SWF(p,domain)
		
				# set up plasticity parameters
				domain=module_plastic(p,domain)

				if domain.plasticitypass:
					break

				passrun+=1

				if passrun>10:
					sys.exit('Plasticity module may be wrong ')

		
			if p.generatedir:
				copy_binary(p.rundir,p.fbin)
				#submit_queue(p.rundir)
	return


#######################################
def module_init(p):

	if p.scenario_module == 'andrews_barall':
		print('\n\n--------------------------------------------------------------\n')
		print('          Set up Andrews and Barall (2011) module\n\n')
		AB=pyab.andrewsbarall(p.dx[0],p.cutratio)
		pyab.plot_model_geo()
		p.nuclsize=5e3
		if p.mag<6.5:
			p.nuclsize=(5e3-1e3)/(6.5-5.5)*(p.mag-5.5)+1e3
		print('Nucleation size is {0:5.2e}'.format(p.nuclsize))
		AB.scenario_from_mag(mag=p.mag,custom_length=p.custom_length,custom_width=p.custom_width,rcrit=p.nuclsize) #mag is from 5.5 to 7.5
		
		p.nx = int(AB.nx)+1
		p.ny = int(AB.ny)+1
		p.nz = int(AB.ny) * 2 + 2
		p.leftnode = 10 
		p.rightnode = 10
		p.hypo = [AB.ihypox,AB.ihypoz,p.ny]
		p.AB = AB
	else:
		print('\n\n--------------------------------------------------------------\n')
		print('         Set up Generic Strike-slip earthquake module\n\n')

	domain=SORDinput(p.nx,p.ny,p.nz,p.hypo,p.ifault,p.dx) 
	
	fault_length=(p.nx-p.leftnode-p.rightnode-1)*p.dx[0]
	fault_nnode = np.floor(fault_length/p.dx[0])+1	

	# set up fault segments including two for pml
	domain.fault_segment(3,(p.leftnode+1,\
						    p.leftnode+int(fault_nnode)))

	print('True fault is from %d to %d (in total %d) \n' %\
		(domain.sege[0],domain.segb[-1],domain.segb[-1]-domain.sege[0]+1))

	print('Hypocentral index is %d %d %d' %(p.hypo[0],p.hypo[1],p.hypo[2]))
	domain.hypo = p.hypo

	# generate sordw3 running folder
	if p.generatedir:
		if p.friction == 'SW':
			print('Generating sordw3 running folder ... ...')
			sord_sw_scenario(p.rundir,np3=p.np3,nn=(p.nx,p.ny,p.nz),T=p.T,ihypo=p.hypo,rcrit=p.nuclsize)

	if p.writebin:
		os.system("rm "+p.fbin+"/*")
		os.system("rm "+p.fig+"/*")
	return p, domain

def module_vel(p,domain):
	print('\n\n--------------------------------------------------------------\n')
	print('                       Set up velocity module\n\n')
		# set up initial depth profile
	domain.init_depth()#; print(domain.depth)

	#generate customized velocity model
	if domain.customvel(velcap=p.velcap,velcapdep=p.velcapdep):
		print('Customized velocity model has been enabled')
	if p.writebin:
		writebin(p.fbin+'vp.bin',domain.vp)   #1D
		writebin(p.fbin+'vs.bin',domain.vs)   #1D
		writebin(p.fbin+'rho.bin',domain.rho) #1D
	if p.figplot:
		PlotFigure(1,np.array([domain.vp,domain.vs,domain.rho]).T,(domain.ny-1)*p.dx[2]/1e3,p.fig+'velmodel.pdf')
	return domain

def module_stress(p,domain):
	print('\n\n--------------------------------------------------------------\n')
	print('                       Set up stress module\n\n')
	
	if p.scenario_module=='andrews_barall':
		domain.AB_stress_scenario(p.psi,p.AB.ts,p.AB.tn)
	else:
		domain.stress_scenario(ratio=p.ratio,psi=p.psi,depsat=p.depsat,constant_gradient=p.constant_gradient,offset=p.offset_stress,p=p)
	if p.writebin:
		# write binary files
		writebin(p.fbin+'sigma_xx.bin',domain.sigma_xx) #2D
		writebin(p.fbin+'sigma_zz.bin',domain.sigma_zz) #2D
		writebin(p.fbin+'sigma_xz.bin',domain.sigma_xz) #2D
		writebin(p.fbin+'sigma_yy.bin',domain.sigma_yy) #2D
	if p.figplot:
		PlotFigure(1,np.array([-domain.sigma_xx[:,int(p.nx/2)],-domain.sigma_yy[:,int(p.nx/2)],-domain.sigma_zz[:,int(p.nx/2)],domain.sigma_xz[:,int(p.nx/2)]]).T,\
			(domain.ny-1)*p.dx[2]/1e3,p.fig+'stressall.pdf')
		PlotFigure(2,domain.sigma_xz,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'sigma_xz.pdf',domain.segb[1:])
		PlotFigure(2,domain.sigma_xx,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'sigma_xx.pdf',domain.segb[1:])
		PlotFigure(2,domain.sigma_yy,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'sigma_yy.pdf',domain.segb[1:])
		PlotFigure(2,domain.sigma_zz,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'sigma_zz.pdf',domain.segb[1:])
		PlotFigure(2,-domain.sigma_xz/domain.sigma_zz,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'ratio.pdf',domain.segb[1:])
	return domain

def module_SWF(p,domain):
	print('\n\n--------------------------------------------------------------\n')
	print('                Set up Slip Weakening Friction module\n\n')


	domain.slipweakening_seg(p.fco,p.mus,p.mud,p.dc)
	if p.scenario_module == 'andrews_barall':
		domain.mus[0:-1,0:-1] = p.AB.mus #cell2node
		domain.mud[0:-1,0:-1] = p.AB.mud
		if PlotFigure:
			PlotFigure(2,p.AB.ts+p.AB.tn*p.AB.mud,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWSD.pdf',domain.segb[1:],-10e6,10e6)


	# coseismic segments
	upindex = 0
	downindex = domain.ny-1
	leftindex = domain.segb[1]-1
	rightindex = domain.sege[1]-1

	nL = rightindex - leftindex 
	nW = downindex - upindex
	print('Plot fault range: x({0}-{1}),y({2}-{3})'.format(leftindex+1,rightindex+1,upindex+1,downindex+1))

	# apply transition boundary
	ftaper=taper2D([nW*p.dx[1],nL*p.dx[0]],p.taperlist,[p.dx[0],p.dx[1]],inverse=True)
	
	domain.fco[upindex:downindex+1,leftindex:rightindex+1] = domain.fco[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 1e6
	domain.mus[upindex:downindex+1,leftindex:rightindex+1] = domain.mus[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 1e6
	domain.mud[upindex:downindex+1,leftindex:rightindex+1] = domain.mud[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 1e6
	domain.dc[upindex:downindex+1,leftindex:rightindex+1] = domain.dc[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 1e6
	
	domain.fco = extend_edge(domain.fco,'left',leftindex+1)
	domain.fco = extend_edge(domain.fco,'right',rightindex)
	domain.mus = extend_edge(domain.mus,'left',leftindex+1)
	domain.mus = extend_edge(domain.mus,'right',rightindex)
	domain.mud = extend_edge(domain.mud,'left',leftindex+1)
	domain.mud = extend_edge(domain.mud,'right',rightindex)
	domain.dc = extend_edge(domain.dc,'left',leftindex+1)
	domain.dc = extend_edge(domain.dc,'right',rightindex)

	if p.checkroughness[0] and p.scenario_module != 'andrews_barall':
		fname=p.checkroughness[1]
		print('\n\n--------------------------------------------------------------\n')
		print('                Checking if ts > co+mus*tn \n\n')

		if (os.path.isfile(fname +'rnhat_x') and \
			os.path.isfile(fname +'rnhat_y') and \
			os.path.isfile(fname +'rnhat_z') ):

			print('Reading fault normal vectors\n')
			domain.nhat_x = readbin(fname +'rnhat_x',(domain.ny,domain.nx))
			domain.nhat_y = readbin(fname +'rnhat_y',(domain.ny,domain.nx))
			domain.nhat_z = readbin(fname +'rnhat_z',(domain.ny,domain.nx))

		else:
			import fault_normal as fn
			fault=fn.Plot3d()
			fault.read_plot3d_ascii(fname+'faultnew.xyz')
			i1 = (1, 1) #not include ghost nodes
			i2 = fault.block_shapes[0][::-1][1:][::-1]
			nhat,area=fn.fault_normal_vector(fault.x[0][0,:,:].T,\
								 			 			   fault.y[0][0,:,:].T,\
								 			 			   fault.z[0][0,:,:].T,\
								 			 			   i1,i2)
			domain.nhat_x = nhat[:,:,0].T
			domain.nhat_y = nhat[:,:,1].T
			domain.nhat_z = nhat[:,:,2].T
			domain.area   = area.T
			
			writebin(fname +'rnhat_x',nhat[:,:,0].T)
			writebin(fname +'rnhat_y',nhat[:,:,1].T)
			writebin(fname +'rnhat_z',nhat[:,:,2].T)
	
		# this will update domain.ts and domain.tn
		domain.rough_ts_tn(domain.nhat_x,domain.nhat_y,domain.nhat_z)
		domain.strength = domain.tn * n2c(domain.mus) + n2c(domain.fco)

		res=domain.strength<domain.ts
		if res.any():
			sys.exit('(SW) Fault Strength is lower than initial shear stress')

		if p.figplot:
			PlotFigure(2,domain.strength-domain.ts,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWRes.pdf',domain.segb[1:])
			PlotFigure(2,domain.ts-domain.tn*n2c(domain.mud),(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWSD.pdf',domain.segb[1:],0,10e6)
			PlotFigure(2,domain.ts/domain.tn,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWtstnratio.pdf',domain.segb[1:])


	if p.writebin:
		writebin(p.fbin+'fco.bin',domain.fco) #2D
		writebin(p.fbin+'mus.bin',domain.mus) #2D
		writebin(p.fbin+'mud.bin',domain.mud) #2D
		writebin(p.fbin+'dc.bin', domain.dc) #2D

	if p.figplot:
		PlotFigure(2,domain.fco,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWfco.pdf',domain.segb[1:])
		PlotFigure(2,domain.mus,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWmus.pdf',domain.segb[1:])
		PlotFigure(2,domain.mud,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWmud.pdf',domain.segb[1:])
		PlotFigure(2,domain.dc,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'SWdc.pdf',domain.segb[1:])
	
	return domain




def module_RSF(p,domain):
	print('\n\n--------------------------------------------------------------\n')
	print('                Set up Rate and State Friction module\n\n')
	domain.rateandstate_seg(p.a,p.b,p.v0,p.f0,p.L,p.fw,p.vw)

	# coseismic segments
	upindex = 0
	downindex = domain.ny-1
	leftindex = domain.segb[1]-1
	rightindex = domain.sege[1]-1

	nL = rightindex - leftindex 
	nW = downindex - upindex
	print('Plot fault range: x({0}-{1}),y({2}-{3})'.format(leftindex+1,rightindex+1,upindex+1,downindex+1))
	if p.randomfield:
		W = (downindex - upindex )*p.dx[1]
		L = (rightindex - leftindex )*p.dx[0]
		srcpar = [W, L]

		G,spar=randomfieldspecdistr(srcpar,p.acf,p.corr,p.seed,p.samp,'nod',1,[],25e3,90,'n','n')
		G = G-G.min()
		Fc = G/G.max()*(p.hgv-p.lov)+p.lov

		xc = np.arange(0,L+p.samp[1],p.samp[1])
		yc = np.arange(0,W+p.samp[0],p.samp[0])

		xf = np.arange(0,L+p.dx[0],p.dx[0])
		yf = np.arange(0,W+p.dx[0],p.dx[1])

		# print(xc.shape,yc.shape,Fc.shape)

		f = interpolate.interp2d(xc, yc, Fc, kind='linear')
		Ff = f(xf,yf)
		# print('\nR_min={0} Rmax={1}'.format(Ff.min(),Ff.max()))
		# make b as randomized
		ftaper=taper2D([nW*p.dx[1],nL*p.dx[0]],p.taperlist,[p.dx[0],p.dx[1]])
		domain.b[upindex:downindex+1,leftindex:rightindex+1] = \
			domain.b[upindex:downindex+1,leftindex:rightindex+1] + Ff * ftaper


	# apply transition boundary
	ftaper=taper2D([nW*p.dx[1],nL*p.dx[0]],p.taperlist,[p.dx[0],p.dx[1]],inverse=True)
	domain.a[upindex:downindex+1,leftindex:rightindex+1] = domain.a[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 0.1
	domain.vw[upindex:downindex+1,leftindex:rightindex+1] = domain.vw[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 10
	domain.fw[upindex:downindex+1,leftindex:rightindex+1] = domain.fw[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 10
	domain.f0[upindex:downindex+1,leftindex:rightindex+1] = domain.f0[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 10
	domain.L[upindex:downindex+1,leftindex:rightindex+1] = domain.L[upindex:downindex+1,leftindex:rightindex+1]+ftaper * 10

	domain.a = extend_edge(domain.a,'left',leftindex+1)
	domain.a = extend_edge(domain.a,'right',rightindex)
	domain.b = extend_edge(domain.b,'left',leftindex+1)
	domain.b = extend_edge(domain.b,'right',rightindex)
	domain.vw = extend_edge(domain.vw,'left',leftindex+1)
	domain.vw = extend_edge(domain.vw,'right',rightindex)
	domain.f0 = extend_edge(domain.f0,'left',leftindex+1)
	domain.f0 = extend_edge(domain.f0,'right',rightindex)
	domain.fw = extend_edge(domain.fw,'left',leftindex+1)
	domain.fw = extend_edge(domain.fw,'right',rightindex)
	domain.v0 = extend_edge(domain.v0,'left',leftindex+1)
	domain.v0 = extend_edge(domain.v0,'right',rightindex)
	domain.L  = extend_edge(domain.L ,'left',leftindex+1)
	domain.L  = extend_edge(domain.L ,'right',rightindex)

    ###################################### use blur2D function ###########################################
	# domain.a = fill_halo(domain.a,'left',leftindex+1, 0.1)
	# domain.a = fill_halo(domain.a,'right',rightindex, 0.1)
	# domain.a = fill_halo(domain.a,'down',downindex-41,0.1)
	# domain.a = blur2D(   domain.a)

	# domain.vw = fill_halo(domain.vw,'left',leftindex+1, 10)
	# domain.vw = fill_halo(domain.vw,'right',rightindex, 10)
	# domain.vw = fill_halo(domain.vw,'down',downindex-41,10)
	# domain.vw = blur2D(   domain.vw)

	# domain.v0 = fill_halo(domain.v0,'left',leftindex+1, 1e-6)
	# domain.v0 = fill_halo(domain.v0,'right',rightindex, 1e-6)
	# domain.v0 = fill_halo(domain.v0,'down',downindex-41,1e-6)
	# domain.v0 = blur2D(   domain.v0)

	# domain.fw = fill_halo(domain.fw,'left',leftindex+1, 10)
	# domain.fw = fill_halo(domain.fw,'right',rightindex, 10)
	# domain.fw = fill_halo(domain.fw,'down',downindex-41,10)
	# domain.fw = blur2D(   domain.fw)

	# domain.L = fill_halo(domain.L,'left',leftindex+1, 10)
	# domain.L = fill_halo(domain.L,'right',rightindex, 10)
	# domain.L = fill_halo(domain.L,'down',downindex-41,10)
	# domain.L = blur2D(   domain.L)

	# domain.f0 = fill_halo(domain.f0,'left',leftindex+1, 10)
	# domain.f0 = fill_halo(domain.f0,'right',rightindex, 10)
	# domain.f0 = fill_halo(domain.f0,'down',downindex-41,10)
	# domain.f0 = blur2D(   domain.f0)
	#################################################################################



	if p.writebin:
		writebin(p.fbin+'a.bin',domain.a) #2D
		writebin(p.fbin+'b.bin',domain.b) #2D
		writebin(p.fbin+'v0.bin',domain.v0) #2D
		writebin(p.fbin+'vw.bin', domain.vw) #2D
		writebin(p.fbin+'f0.bin', domain.f0) #2D
		writebin(p.fbin+'fw.bin', domain.fw) #2D
		writebin(p.fbin+'L.bin', domain.L) #2D
	if p.figplot:
		PlotFigure(2,domain.a,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSa.pdf',domain.segb[1:])
		PlotFigure(2,domain.b,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSb.pdf',domain.segb[1:])
		PlotFigure(2,domain.f0,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSf0.pdf',domain.segb[1:])
		PlotFigure(2,domain.fw,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSfw.pdf',domain.segb[1:])
		PlotFigure(2,domain.v0,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSv0.pdf',domain.segb[1:])
		PlotFigure(2,domain.vw,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSvw.pdf',domain.segb[1:])
		PlotFigure(2,domain.L,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSL.pdf',domain.segb[1:])
		PlotFigure(2,domain.b-domain.a,(domain.ny-1)*p.dx[2]/1e3,(domain.nx-1)*p.dx[0]/1e3,p.fig+'RSb-a.pdf',domain.segb[1:],0,0.01)
	return domain


def module_plastic(p,domain):
	print('\n\n--------------------------------------------------------------\n')
	print('                       Set up Plasticity module\n\n')
	domain.plasticity2D([2,],p.weight,p.plmodel,p.plvar,p.mis) #'Horsrud2001','Chang2006', 'Roten2014', 'HoekBrown'
	leftindex = domain.segb[1]-1
	rightindex = domain.sege[1]-1
	domain.pco = extend_edge(domain.pco,'left',leftindex+1)
	domain.pco = extend_edge(domain.pco,'right',rightindex-1)
	domain.phi = extend_edge(domain.phi,'left',leftindex+1)
	domain.phi = extend_edge(domain.phi,'right',rightindex-1)

	if p.check_yieldsurface:
		domain.check_yieldsurface()
	else:
		domain.plasticitypass=True
		
	if p.writebin:
		writebin(p.fbin+'phi.bin',domain.phi) #2D
		writebin(p.fbin+'pco.bin',domain.pco) #2D
	if p.figplot:
		PlotFigure(1,domain.pco[:,0],(domain.ny-1)*p.dx[2]/1e3,p.fig+'pco.pdf')
		PlotFigure(1,np.arctan(domain.phi[:,0])*180/np.pi,(domain.ny-1)*p.dx[2]/1e3,p.fig+'phi.pdf')

	return domain

###############################################################################
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

	def customvel(self,velcap=False,velcapdep=2e3,dtype=idtype,method=None):
		self.vp=np.empty(self.ny-1,dtype=dtype)
		self.vs=np.empty(self.ny-1,dtype=dtype)
		self.rho=np.empty(self.ny-1,dtype=dtype)

		if velcap:
			rho_cap,vp_cap,vs_cap = Shuoma_velstru(velcapdep, SI=True)
		for i in range(self.ny-1):
			depth = self.cdepth[i]
			if method is None:
				if velcap and depth<velcapdep:
					self.rho[i],self.vp[i],self.vs[i] = rho_cap, vp_cap, vs_cap
				else:
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

	def AB_stress_scenario(self,psi,ts,tn):
		self.sigma_yy = np.zeros((self.ny-1,self.nx-1))
		self.sigma_xx = np.zeros((self.ny-1,self.nx-1))
		self.sigma_zz = np.zeros((self.ny-1,self.nx-1))
		self.sigma_xz = np.zeros((self.ny-1,self.nx-1))

		self.sigma_yy = tn


		if isinstance(psi,float):
			self.sigma_zz = 2*self.sigma_yy/(2+2*(ts/tn)/(np.tan(2*psi/180*np.pi)))
			self.sigma_xx = (1+2*ts/tn/(np.tan(2*psi/180*np.pi)))*self.sigma_zz
			self.sigma_xz = ts/tn*self.sigma_zz
		else:
			sys.exit('Heterogeneous psi in Andrews and Barall module has not been implemented yet!')

		return





	def stress_scenario(self,ratio=0.33,psi=45.,depsat=5e3,constant_gradient=True,rho=None, rhow=1000, g=9.81, offset=None,p=None):
		# offset is stress offset at zero depth
		if rho is None:
			rho = 2700
		if offset is None:
			offset = 0.

		self.sigma_yy = np.zeros((self.ny-1,self.nx-1))
		self.sigma_xx = np.zeros((self.ny-1,self.nx-1))
		self.sigma_zz = np.zeros((self.ny-1,self.nx-1))
		self.sigma_xz = np.zeros((self.ny-1,self.nx-1))

		# customized define
		if constant_gradient:
			sigma_y = -offset - np.minimum(self.cdepth,depsat) * g * (rho - rhow)

		# integral of density
		else:
			gh=gammaH(self.rho,self.dx[1])
			sigma_y = -offset + np.minimum(self.cdepth,depsat) * g * rhow - gh

		for i in range(self.nx-1):
				self.sigma_yy[:,i] = sigma_y


		if isinstance(ratio,float) and isinstance(psi,float):
			for i in range(self.nx-1):
				self.sigma_zz[:,i] = 2*sigma_y/(2-2*ratio/(np.tan(2*psi/180*np.pi)))
				self.sigma_xx[:,i] = (1-2*ratio/(np.tan(2*psi/180*np.pi)))*self.sigma_zz[:,i]
				self.sigma_xz[:,i] = -ratio*self.sigma_zz[:,i]
		elif isinstance(psi,float) and not isinstance(ratio,float):

			if ratio[0]=='depvar':
				upindex = 0
				downindex = self.ny-1
				leftindex = self.segb[1]-1
				rightindex = self.sege[1]-1

				tmp=np.zeros((self.ny-1,))

				idep=1
				for j in range(upindex,downindex):
					dep=self.cdepth[j]
	
					if dep>ratio[idep+1][0]:
						idep+=1
	
					tmp[j] = ratio[idep][1]+(dep-ratio[idep][0])*(ratio[idep+1][1]-ratio[idep][1])/ \
					                                                             (ratio[idep+1][0]-ratio[idep][0])
				for i in range(self.nx-1):
					self.sigma_zz[:,i] = 2*sigma_y/(2-2*tmp/(np.tan(2*psi/180*np.pi)))
					self.sigma_xx[:,i] = (1-2*tmp/(np.tan(2*psi/180*np.pi)))*self.sigma_zz[:,i]
					self.sigma_xz[:,i] = -tmp*self.sigma_zz[:,i]	

				return                                                             


			for i in range(self.nx-1):
				self.sigma_zz[:,i] = 2*sigma_y/(2-(ratio[2]+ratio[3])/(np.tan(2*psi/180*np.pi)))
				self.sigma_xx[:,i] = (1-(ratio[2]+ratio[3])/(np.tan(2*psi/180*np.pi)))*self.sigma_zz[:,i]
				self.sigma_xz[:,i] = -(ratio[2]+ratio[3])*self.sigma_zz[:,i]/2

			upindex = 0
			downindex = self.ny-1
			leftindex = self.segb[1]-1
			rightindex = self.sege[1]-1

			nL = rightindex - leftindex 
			nW = downindex - upindex

			if p.randomfield:
				W = (downindex - upindex )*p.dx[1]
				L = (rightindex - leftindex )*p.dx[0]
				srcpar = [W, L]
		
				G,spar=randomfieldspecdistr(srcpar,ratio[0],ratio[1],ratio[4],p.samp,'nod',1,[],25e3,90,'n','n')
				G = G-G.min()
				Fc = G/G.max()*(ratio[3]-ratio[2])+ratio[2]
		
				xc = np.arange(0,L+p.samp[1],p.samp[1])
				yc = np.arange(0,W+p.samp[0],p.samp[0])
		
				xf = np.arange(0,L+p.dx[0],p.dx[0])
				yf = np.arange(0,W+p.dx[0],p.dx[1])
		
				# print(xc.shape,yc.shape,Fc.shape)
		
				f = interpolate.interp2d(xc, yc, Fc, kind='linear')
				Ff = f(xf,yf)
				# print('\nR_min={0} Rmax={1}'.format(Ff.min(),Ff.max()))
				# make b as randomized
				# ftaper=taper2D([nW*p.dx[1],nL*p.dx[0]],p.taperlist,[p.dx[0],p.dx[1]])

				self.sigma_zz[upindex:downindex,leftindex:rightindex+1] = \
					2*self.sigma_yy[upindex:downindex,leftindex:rightindex+1]/(2-2*Ff[:-1,:]/(np.tan(2*psi/180*np.pi)))

				self.sigma_xx[upindex:downindex,leftindex:rightindex+1] = \
					(1-2*Ff[:-1,:] /(np.tan(2*psi/180*np.pi)))*self.sigma_zz[upindex:downindex,leftindex:rightindex+1]

				self.sigma_xz[upindex:downindex,leftindex:rightindex+1] = \
					-Ff[:-1,:]*self.sigma_zz[upindex:downindex,leftindex:rightindex+1]

		elif isinstance(ratio,float) and not isinstance(psi,float):
			for i in range(self.nx-1):
				self.sigma_zz[:,i] = 2*sigma_y/(2-2*ratio/(np.tan((psi[2]+psi[3])/180*np.pi)))
				self.sigma_xx[:,i] = (1-2*ratio/(np.tan((psi[2]+psi[3])/180*np.pi)))*self.sigma_zz[:,i]
				self.sigma_xz[:,i] = -ratio*self.sigma_zz[:,i]

			upindex = 0
			downindex = self.ny-1
			leftindex = self.segb[1]-1
			rightindex = self.sege[1]-1

			nL = rightindex - leftindex 
			nW = downindex - upindex

			if p.randomfield:
				W = (downindex - upindex )*p.dx[1]
				L = (rightindex - leftindex )*p.dx[0]
				srcpar = [W, L]
		
				G,spar=randomfieldspecdistr(srcpar,psi[0],psi[1],psi[4],p.samp,'nod',1,[],25e3,90,'n','n')
				G = G-G.min()
				Fc = G/G.max()*(psi[3]-psi[2])+psi[2]
		
				xc = np.arange(0,L+p.samp[1],p.samp[1])
				yc = np.arange(0,W+p.samp[0],p.samp[0])
		
				xf = np.arange(0,L+p.dx[0],p.dx[0])
				yf = np.arange(0,W+p.dx[0],p.dx[1])
		
				# print(xc.shape,yc.shape,Fc.shape)
		
				f = interpolate.interp2d(xc, yc, Fc, kind='linear')
				Ff = f(xf,yf)
				# print('\nR_min={0} Rmax={1}'.format(Ff.min(),Ff.max()))
				# make b as randomized
				# ftaper=taper2D([nW*p.dx[1],nL*p.dx[0]],p.taperlist,[p.dx[0],p.dx[1]])

				self.sigma_zz[upindex:downindex,leftindex:rightindex+1] = \
					2*self.sigma_yy[upindex:downindex,leftindex:rightindex+1]/(2-2*ratio/(np.tan(2*Ff[:-1,:]/180*np.pi)))

				self.sigma_xx[upindex:downindex,leftindex:rightindex+1] = \
					(1-2*ratio/(np.tan(2*Ff[:-1,:]/180*np.pi)))*self.sigma_zz[upindex:downindex,leftindex:rightindex+1]

				self.sigma_xz[upindex:downindex,leftindex:rightindex+1] = \
					-ratio*self.sigma_zz[upindex:downindex,leftindex:rightindex+1]
		else:
			print('Please do not simultaneously use psi and ratio as random fields')
			sys.exit()

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
		cnhat_x = n2c(nhat_x)
		cnhat_y = n2c(nhat_y)
		cnhat_z = n2c(nhat_z)

		# pre-traction
		t0_x = self.sigma_xx * cnhat_x
		t0_y = self.sigma_yy * cnhat_y + self.sigma_xz * cnhat_z
		t0_z = self.sigma_xz * cnhat_y + self.sigma_zz * cnhat_z

		# normal traction
		self.tn = t0_x * cnhat_x + t0_y * cnhat_y + t0_z * cnhat_z

		self.ts = np.sqrt((t0_x-self.tn * cnhat_x)**2 +\
			                               (t0_y-self.tn * cnhat_y)**2 +\
			                               (t0_z-self.tn * cnhat_z)**2)
		self.tn = -self.tn #convert to postive tn

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


	def rateandstate_seg(self, *args):
		try:
			b = args[1]
			v0= args[2]
			f0 = args[3]
			L = args[4]
			fw = args[5]
			a = args[0]
			vw = args[6]
		except:
			print('Input wrong in R&S friction')

		self.a   = np.zeros((self.ny,self.nx),dtype=idtype)
		self.b   = np.zeros((self.ny,self.nx),dtype=idtype)
		self.f0  = np.zeros((self.ny,self.nx),dtype=idtype)
		self.fw  = np.zeros((self.ny,self.nx),dtype=idtype)
		self.L   = np.zeros((self.ny,self.nx),dtype=idtype)
		self.v0  = np.zeros((self.ny,self.nx),dtype=idtype)
		self.vw  = np.zeros((self.ny,self.nx),dtype=idtype)

		upindex = 0
		downindex = self.ny

		# b
		if isinstance(b,float):
			self.b[:,:] = b
		
		# v0
		if isinstance(v0,float):
			self.v0[:,:] = v0
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>v0[idep+1][0]:
					idep+=1

				self.v0[j,:] = v0[idep][1]+(dep-v0[idep][0])*(v0[idep+1][1]-v0[idep][1])/ \
				                                                             (v0[idep+1][0]-v0[idep][0])
		
		# f0
		if isinstance(f0,float):
			self.f0[:,:] = f0
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>f0[idep+1][0]:
					idep+=1

				self.f0[j,:] = f0[idep][1]+(dep-f0[idep][0])*(f0[idep+1][1]-f0[idep][1])/ \
				                                                             (f0[idep+1][0]-f0[idep][0])

		# fw
		if isinstance(fw,float):
			self.fw[:,:] = fw
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>fw[idep+1][0]:
					idep+=1

				self.fw[j,:] = fw[idep][1]+(dep-fw[idep][0])*(fw[idep+1][1]-fw[idep][1])/ \
				                                                             (fw[idep+1][0]-fw[idep][0])

		# L
		if isinstance(L,float):
			self.L[:,:]  =  L
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>a[idep+1][0]:
					idep+=1

				self.L[j,:] = L[idep][1]+(dep-L[idep][0])*(L[idep+1][1]-L[idep][1])/ \
				                                                             (L[idep+1][0]-L[idep][0])

		# a
		if isinstance(a,float):
			self.a[:,:]  =  a
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>a[idep+1][0]:
					idep+=1

				self.a[j,:] = a[idep][1]+(dep-a[idep][0])*(a[idep+1][1]-a[idep][1])/ \
				                                                             (a[idep+1][0]-a[idep][0])


		# vw		                                                             	
		if isinstance(vw,float):
			self.vw[:,:]  =  vw
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>a[idep+1][0]:
					idep+=1

				self.vw[j,:] = vw[idep][1]+(dep-vw[idep][0])*(vw[idep+1][1]-vw[idep][1])/ \
					                                                                (vw[idep+1][0]-vw[idep][0])
		
		return




	def slipweakening_seg(self, *args):
		try:
			fco = args[0]
			mus = args[1]
			mud = args[2]
			dc = args[3]
		except:
			print('Input wrong in slip weakening friction')

		self.fco = np.zeros((self.ny,self.nx),dtype=idtype)
		self.mus = np.zeros((self.ny,self.nx),dtype=idtype)
		self.mud = np.zeros((self.ny,self.nx),dtype=idtype)
		self.dc  = np.zeros((self.ny,self.nx),dtype=idtype)

		upindex = 0
		downindex = self.ny

		# fc0
		if isinstance(fco,float):
			self.fco[:,:] = fco
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>fco[idep+1][0]:
					idep+=1

				self.fco[j,:] = fco[idep][1]+(dep-fco[idep][0])*(fco[idep+1][1]-fco[idep][1])/ \
				                                                             (fco[idep+1][0]-fco[idep][0])

		# mus
		if isinstance(mus,float):
			self.mus[:,:] = mus
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>mus[idep+1][0]:
					idep+=1

				self.mus[j,:] = mus[idep][1]+(dep-mus[idep][0])*(mus[idep+1][1]-mus[idep][1])/ \
				                                                             (mus[idep+1][0]-mus[idep][0])	
				                                                            	                                                          
		# mud
		if isinstance(mud,float):
			self.mud[:,:] = mud
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>mud[idep+1][0]:
					idep+=1

				self.mud[j,:] = mud[idep][1]+(dep-mud[idep][0])*(mud[idep+1][1]-mud[idep][1])/ \
				                                                             (mud[idep+1][0]-mud[idep][0])		                                                             

        # dc
		if isinstance(dc,float):
			self.dc[:,:] = dc
		else:
			idep=0
			for j in range(upindex,downindex):
				dep=self.ndepth[j]

				if dep>dc[idep+1][0]:
					idep+=1

				self.dc[j,:] = dc[idep][1]+(dep-dc[idep][0])*(dc[idep+1][1]-dc[idep][1])/ \
				                                                             (dc[idep+1][0]-dc[idep][0])
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
			closeness = args[4][0]
			phi = args[4][1]
			self.closeness=np.zeros((self.ny-1,self.nx-1),dtype=idtype)


		nseg = len(seglist)

		if method == 'HoekBrown':
			gh=gammaH(self.rho,self.dx[1])

		for ifault in range(nseg):
			leftindex=self.segb[seglist[ifault]-1]-1
			rightindex=self.sege[seglist[ifault]-1]-2
			upindex = 0
			downindex = self.ny-2

			
			idep1=0
			idep2=0
			for j in range(upindex,downindex+1):

				if method=='closeness':
					dep=self.ndepth[j]
					if dep>phi[idep1+1][0]:
						idep1+=1
					self.phi[j,:] = phi[idep1][1]+(dep-phi[idep1][0])*(phi[idep1+1][1]-phi[idep1][1])/(phi[idep1+1][0]-phi[idep1][0])
					


					dep=self.ndepth[j]
					if dep>closeness[idep2+1][0]:
						idep2+=1
					self.closeness[j,:] = closeness[idep2][1]+(dep-closeness[idep2][0])*(closeness[idep2+1][1]-closeness[idep2][1])/(closeness[idep2+1][0]-closeness[idep2][0])

				for i in range(leftindex,rightindex+1):

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
						# self.phi[j,i] = 0.75

						s11 = self.sigma_xx[j,i]
						s22 = self.sigma_yy[j,i]
						s33 = self.sigma_zz[j,i]
						s13 = self.sigma_xz[j,i]
						val=cohesionfromcloseness([s11,s22,s33,0,s13,0],self.phi[j,i],self.closeness[j,i])
						if val is False:
							print(i,'in',self.sigma_xx.shape[1])
							print(j,'in',self.sigma_xx.shape[0])
							print(s11,s22,s33,s13,-s13/s33)
							# sys.exit()
							self.plasticitypass=False
						else:
							self.pco[j,i]=val

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


		return

	def check_yieldsurface(self):
		self.plasticitypass=True
		print('checking yieldsurface: ',self.plasticitypass)
		for i in range(self.sigma_zz.shape[1]):
			for j in range(self.sigma_zz.shape[0]):
					# verify all 
					s11 = self.sigma_xx[j,i]
					s22 = self.sigma_yy[j,i]
					s33 = self.sigma_zz[j,i]
					s13 = self.sigma_xz[j,i]
					sbar,sy = plasticyieldsurface([s11,s22,s33,0,s13,0],self.phi[j,i],self.pco[j,i])
					# if i==leftindex:
						# print('Dep= {0:8.0f} m, closeness= {1:3.1f}% j={2}'.format(self.cdepth[j],sbar/sy*100,j))
					if sbar > sy:
						print('Warning: {0} should be < {1} at {2} m'.format(sbar,sy,self.cdepth[j]))
						self.plasticitypass=False
		print('checking yieldsurface: ',self.plasticitypass)
		return

if __name__ == '__main__':
	# multi_run()
	single_run()

