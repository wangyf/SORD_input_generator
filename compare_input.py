import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



idtype = np.dtype( 'f8' ).str #'<f8'


def readbin(file,shape,inputdtype=None):
	fd = open( file ,'rb')
	if inputdtype == None:
		inputdtype = idtype
	matrix = np.fromfile(fd, inputdtype).reshape(shape).astype(idtype)
# 	print('Read size check',file,matrix.shape)
	fd.close()
	return matrix


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

def compare_yieldsurface():
	ny = 351
	nx = 3438
	dx = 0.05
	stk=np.arange(nx)*dx
	dep=np.arange(ny)*dx+dx/2
	
	
	# iy = 100
	
	# foldername='verygood/'
	# fname=foldername+'out/tse.out'
	# slip1=readbin(fname,(ny,nx))[iy,:]
	
	# foldername='good/'
	# fname=foldername+'out/tse.out'
	# slip2=readbin(fname,(ny,nx))[iy,:]
	
	# foldername='average/'
	# fname=foldername+'out/tse.out'
	# slip3=readbin(fname,(ny,nx))[iy,:]
	
	# foldername='poor/'
	# fname=foldername+'out/tse.out'
	# slip4=readbin(fname,(ny,nx))[iy,:]
	
	
	# plt.plot(stk,slip1,label='verygood')
	# plt.plot(stk,slip2,label='good');
	# plt.plot(stk,slip3,label='average');
	# plt.plot(stk,slip4,label='poor');
	# plt.xlabel('Along strike (km)')
	# plt.gca().set_ylim([0,50e6])
	# # plt.ylabel('Fault displacement (m)')
	# plt.legend(loc='best',fontsize='x-small')
	# plt.show()
	
	foldername='averagebin/'
	fname=foldername+'sigma_xx.bin'
	s11=readbin(fname,(ny-1,nx-1))[:,0]
	fname=foldername+'sigma_yy.bin'
	s22=readbin(fname,(ny-1,nx-1))[:,0]
	fname=foldername+'sigma_zz.bin'
	s33=readbin(fname,(ny-1,nx-1))[:,0]
	fname=foldername+'sigma_xz.bin'
	s31=readbin(fname,(ny-1,nx-1))[:,0]
	
	
	foldername='averagebin/'
	fname=foldername+'pco.bin'
	f1=readbin(fname,(ny-1,nx-1))[:,0]
	fname=foldername+'phi.bin'
	g1=readbin(fname,(ny-1,nx-1))[:,0]
	
	# foldername='close0.5bin/'
	# fname=foldername+'pco.bin'
	# f2=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'phi.bin'
	# g2=readbin(fname,(ny-1,nx-1))[:,0]
	
	# foldername='close0.4bin/'
	# fname=foldername+'pco.bin'
	# f3=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'phi.bin'
	# g3=readbin(fname,(ny-1,nx-1))[:,0]
	
	# foldername='poorbin/'
	# fname=foldername+'pco.bin'
	# f4=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'phi.bin'
	# g4=readbin(fname,(ny-1,nx-1))[:,0]
	
	plt.subplot(131)
	plt.plot(f1,dep[0:-1],color='r',label='Cohension (Pa)');
	# plt.plot(f2,dep[0:-1],color='g',label='closeness05');
	# plt.plot(f3,dep[0:-1],color='b',label='closeness04');
	# plt.plot(f4,dep[0:-1],label='poor');
	plt.gca().set_xscale('log')
	plt.gca().invert_yaxis()
	plt.gca().set_xlim([1e5,1e9])
	plt.grid(True)
	plt.legend(loc='best',fontsize='small')
	plt.gca().set_ylabel('Depth (km)')
	
	plt.subplot(132)
	plt.plot(np.arctan(g1)*180/np.pi,dep[0:-1],color='r',label='Frictional angle $\circ$');
	# plt.plot(np.arctan(g2)*180/np.pi,dep[0:-1],color='g',label='closeness05');
	# plt.plot(np.arctan(g3)*180/np.pi,dep[0:-1],color='b',label='closeness04');
	# plt.plot(np.arctan(g4)*180/np.pi,dep[0:-1],label='poor');
	plt.grid(True)
	plt.gca().invert_yaxis()
	plt.legend(loc='best',fontsize='small')
	
	# yeild surface
	sy1 = np.empty(len(f1))
	sy2 = np.empty(len(f1))
	sy3 = np.empty(len(f1))
	# sy4 = np.empty(len(f1))
	
	sbar = np.empty(len(f1))
	
	for i in range(len(f1)):
		sbar[i], sy1[i]= plasticyieldsurface([s11[i],s22[i],s33[i],0,s31[i],0],g1[i],f1[i])
		# sbar[i], sy2[i]= plasticyieldsurface([s11[i],s22[i],s33[i],0,s31[i],0],g2[i],f2[i])
		# sbar[i], sy3[i]= plasticyieldsurface([s11[i],s22[i],s33[i],0,s31[i],0],g3[i],f3[i])
		# sbar[i], sy4[i]= plasticyieldsurface([s11[i],s22[i],s33[i],0,s31[i],0],g4[i],f4[i])
	
	plt.subplot(133)
	plt.plot(sy1,dep[0:-1],color='r',label='Yield stress (Pa)');
	# plt.plot(sy2,dep[0:-1],color='g',label='closeness05');
	# plt.plot(sy3,dep[0:-1],color='b',label='closeness04');
	# plt.plot(sy4,dep[0:-1],label='poor');
	plt.plot(sbar,dep[0:-1],color='black',linestyle='dashed',label='$\sqrt{J_2}$ (Pa)');
	# plt.plot(sbar*2,dep[0:-1],color='gray',linestyle='dashed',label='J2x2');
	plt.grid(True)
	plt.gca().invert_yaxis()
	plt.gca().set_xscale('log')
	plt.gca().set_xlim([1e5,1e9])
	plt.legend(loc='best',fontsize='small')
	
	plt.show()

def check_stress():
	print('Check stress\n')
	ny = 401
	nx = 1601
	dx = 0.05
	stk=np.arange(nx)*dx
	dep=np.arange(ny)*dx+dx/2

	# foldername='psi15/'
	# fname=foldername+'sigma_xx.bin'
	# s15xx=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_yy.bin'
	# s15yy=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_zz.bin'
	# s15zz=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_xz.bin'
	# s15xz=readbin(fname,(ny-1,nx-1))[:,0]

	# foldername='psi30/'
	# fname=foldername+'sigma_xx.bin'
	# s30xx=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_yy.bin'
	# s30yy=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_zz.bin'
	# s30zz=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_xz.bin'
	# s30xz=readbin(fname,(ny-1,nx-1))[:,0]

	# foldername='psi45/'
	# fname=foldername+'sigma_xx.bin'
	# s45xx=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_yy.bin'
	# s45yy=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_zz.bin'
	# s45zz=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_xz.bin'
	# s45xz=readbin(fname,(ny-1,nx-1))[:,0]

	# foldername='psi60/'
	# fname=foldername+'sigma_xx.bin'
	# s60xx=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_yy.bin'
	# s60yy=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_zz.bin'
	# s60zz=readbin(fname,(ny-1,nx-1))[:,0]
	# fname=foldername+'sigma_xz.bin'
	# s60xz=readbin(fname,(ny-1,nx-1))[:,0]


	# sxz = s45xz
	# szz = s45zz
	# sxx = s45xx

	# def s11(psi,dep):
	# 	psir = psi*np.pi/180.
	# 	sdif=sxz/np.sin(2*psir)
	# 	ssum=szz-sdif*np.cos(2*psir)
	# 	s11=ssum-sdif*np.cos(2*psir)

	# 	plt.plot(s11,dep[0:-1],label='$\Psi='+str(psi)+'\circ$')
	# 	return 

	# plt.plot(sxx,dep[0:-1],label='$\Psi=45\circ$')
	# s11(10,dep)
	# s11(20,dep)
	# s11(30,dep)
	# s11(40,dep)
	# s11(50,dep)
	# s11(60,dep)
	# s11(70,dep)
	# plt.gca().invert_yaxis()

	plt.subplot(2,2,1)
	plt.plot(-s15xx,dep[0:-1],label='$\Psi=15\circ$')
	plt.plot(-s30xx,dep[0:-1],label='$\Psi=30\circ$')
	plt.plot(-s45xx,dep[0:-1],label='$\Psi=45\circ$')
	plt.plot(-s60xx,dep[0:-1],label='$\Psi=60\circ$')
	plt.gca().invert_yaxis()
	# plt.gca().set_xscale('log')
	plt.legend(loc='best',fontsize='x-small')


	plt.subplot(2,2,2)
	plt.plot(-s15yy,dep[0:-1],label='$\Psi=15\circ$')
	plt.plot(-s30yy,dep[0:-1],label='$\Psi=30\circ$')
	plt.plot(-s45yy,dep[0:-1],label='$\Psi=45\circ$')
	plt.plot(-s60yy,dep[0:-1],label='$\Psi=60\circ$')
	plt.gca().invert_yaxis()
	plt.legend(loc='best',fontsize='x-small')

	plt.subplot(2,2,3)
	plt.plot(-s15zz,dep[0:-1],label='$\Psi=15\circ$')
	plt.plot(-s30zz,dep[0:-1],label='$\Psi=30\circ$')
	plt.plot(-s45zz,dep[0:-1],label='$\Psi=45\circ$')
	plt.plot(-s60zz,dep[0:-1],label='$\Psi=60\circ$')
	plt.gca().invert_yaxis()
	plt.legend(loc='best',fontsize='x-small')

	plt.subplot(2,2,4)
	plt.plot(s15xz,dep[0:-1],label='$\Psi=15\circ$')
	plt.plot(s30xz,dep[0:-1],label='$\Psi=30\circ$')
	plt.plot(s45xz,dep[0:-1],label='$\Psi=45\circ$')
	plt.plot(s60xz,dep[0:-1],label='$\Psi=60\circ$')
	plt.gca().invert_yaxis()
	plt.legend(loc='best',fontsize='x-small')

	# plt.legend(loc='best',fontsize='x-small')
	plt.show()


if __name__ == '__main__':
	# check_stress()
	compare_yieldsurface()
	