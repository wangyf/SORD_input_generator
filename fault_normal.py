import numpy as np
from numpy import zeros

# global variable
idtype = np.dtype( 'f8' ).str #'<f8'
#idtype = np.dtype( 'f' ).str #'<f4'

class Plot3d:  # pragma: no cover
	def __init__(self):
		self.x = {}
		self.y = {}
		self.z = {}
		self.block_shapes = {}

	def read_plot3d(self, p3d_name):  # pragma: no cover
		self.read_plot3d_ascii(p3d_name)

	def read_plot3d_ascii(self, p3d_name):  # pragma: no cover
		p3d_file = open(p3d_name, 'r')
		sline = p3d_file.readline().strip().split()
		assert len(sline) == 1, sline
		nblocks = int(sline[0])

		npoints = 0
		for i in range(nblocks):
			nx, ny, nz = p3d_file.readline().strip().split()
			nx = int(nx)
			ny = int(ny)
			nz = int(nz)
			self.block_shapes[i] = (nx, ny, nz)
			self.x[i] = zeros((nx * ny * nz), 'float32')
			self.y[i] = zeros((nx * ny * nz), 'float32')
			self.z[i] = zeros((nx * ny * nz), 'float32')
			npoints += nx * ny * nz

		nleft = npoints * 3
		iblock = 0
		nxyzi = 0
		ixyz = 0
		block = self.x[iblock]
		nxyz = len(block)
		nxyzi2 = None
		while nleft > 0:
			sline = p3d_file.readline().strip().split()
			floats = [float(s) for s in sline]
			nxyzi2 = nxyzi + len(floats)
			block[nxyzi : nxyzi2] = floats

			# print "sline = ", sline, nxyzi, nxyzi2, nxyz
			nxyzi = nxyzi2
			if nxyzi2 == nxyz:
				print("finished with block %i ixyz=%s" % (iblock, ixyz))
				#block = self.blocks[iblock]
				print("reshaping...", self.block_shapes[iblock][::-1])
				nleft -= nxyz
				blocki = block.reshape(self.block_shapes[iblock][::-1])
				if ixyz == 0:
					self.x[iblock] = blocki
				elif ixyz == 1:
					self.y[iblock] = blocki
				elif ixyz == 2:
					self.z[iblock] = blocki
				else:
					raise RuntimeError()

				# next block
				nxyzi = 0
				if ixyz == 0:
					block = self.y[iblock]
					nxyz = len(block)
					ixyz = 1
				elif ixyz == 1:
					block = self.z[iblock]
					nxyz = len(block)
					ixyz = 2
				elif ixyz == 2:
					iblock += 1
					if iblock == nblocks:
						break
					block = self.x[iblock]
					nxyz = len(block)
					ixyz = 0
				else:
					raise NotImplementedError()
				print("iblock=%s icoeff=%s nleft=%s" %(iblock, ixyz, nleft))

			elif nxyzi2 > nxyz:
				asdf2

		print("finished with all blocks")

def writebin(file,matrix,outdtype=idtype):
	fd = open( file ,'wb')
	# if dtype==None: dtype=np.dtype( 'f8' ).str #'<f4'
	print('Write size check',file,matrix.shape)
	matrix.astype(outdtype).tofile( fd )
	fd.close()
	return

def fault_normal_vector_custom1(x,y,z,i1,i2):
	"""
	Here we require fault is normal to z
	rank 0 is x
	rank 1 is y

	This methodology is surface intergral of jobobian matrix
	"""

	size = x.shape
	nhatvec = zeros(size+(3,))
	area = zeros(size)

	grid   = zeros([ix+2 for ix in size]+[3,])
	xghost = zeros([ix+2 for ix in size])
	yghost = zeros([ix+2 for ix in size])
	zghost = zeros([ix+2 for ix in size])

	xghost[1:-1,1:-1] = x
	yghost[1:-1,1:-1] = y
	zghost[1:-1,1:-1] = z
	#boundary condition
	xghost[0,:] = xghost[1,:]
	xghost[-1,:]= xghost[-2,:]
	xghost[:,0] = xghost[:,1]
	xghost[:,-1]= xghost[:,-2]

	yghost[0,:] = yghost[1,:]
	yghost[-1,:]= yghost[-2,:]
	yghost[:,0] = yghost[:,1]
	yghost[:,-1]= yghost[:,-2]

	zghost[0,:] = zghost[1,:]
	zghost[-1,:]= zghost[-2,:]
	zghost[:,0] = zghost[:,1]
	zghost[:,-1]= zghost[:,-2]

	grid[:,:,0] = xghost
	grid[:,:,1] = yghost
	grid[:,:,2] = zghost

	# A=[-1,0,1] #integral (from -1 to 1) over devivative of 1D 3rd order Lagrandre polynomial at 3 nodes
	# B=[1./3.,4./3.,1./3.] #integral (from -1,1) over 1D 3rd order Lagrandre polynomial at 3 nodes

	for a in range(3):
		a = a+1
		b = np.mod(a,3)+1
		c = np.mod(a+1,3)+1

		a = a-1
		b = b-1
		c = c-1

		for k in range(i1[1],i2[1]+1): #alo
			for j in range(i1[0],i2[0]+1): 
				nhatvec[j-1,k-1,a] =   gradient_xi(grid[j-1:j+2,k-1:k+2,b]) * \
									  gradient_eta(grid[j-1:j+2,k-1:k+2,c]) - \
									   gradient_xi(grid[j-1:j+2,k-1:k+2,c]) * \
									  gradient_eta(grid[j-1:j+2,k-1:k+2,b])

		#normalize normal vector
	for k in range(i1[1],i2[1]+1): #along z
		for j in range(i1[0],i2[0]+1): #along x
			area[j-1,k-1] = np.sqrt(nhatvec[j-1,k-1,0]**2 + \
				   		   			nhatvec[j-1,k-1,1]**2 + \
						   			nhatvec[j-1,k-1,2]**2)
			if area[j-1,k-1] < 1e-3:
				print(j,k,area)
			nhatvec[j-1,k-1,:] = nhatvec[j-1,k-1,:]/area[j-1,k-1]

	return nhatvec, area	

def gradient_xi(field):
	B=[1./3.,4./3.,1./3.]
	grad = -field[0,0]*B[0] - field[0,1]*B[1] - field[0,2]*B[2] +\
		    field[2,0]*B[0] + field[2,1]*B[1] + field[2,2]*B[2]
	grad = grad/4.

	return grad

def gradient_eta(field):
	B=[1./3.,4./3.,1./3.]
	grad = -field[0,0]*B[0] + field[0,2]*B[0] +\
		   -field[1,0]*B[1] + field[1,2]*B[1] +\
		   -field[2,0]*B[2] + field[2,2]*B[2]
	grad = grad/4.

	return grad

def fault_normal_vector(x,y,z,i1,i2):
	"""
	Here we require fault is normal to z
	rank 0 is x
	rank 1 is y
	"""
	size = x.shape
	nhatvec = zeros(size+(3,))
	area = zeros(size)

	grid   = zeros([ix+2 for ix in size]+[3,])
	xghost = zeros([ix+2 for ix in size])
	yghost = zeros([ix+2 for ix in size])
	zghost = zeros([ix+2 for ix in size])

	xghost[1:-1,1:-1] = x
	yghost[1:-1,1:-1] = y
	zghost[1:-1,1:-1] = z
	#boundary condition
	xghost[0,:] = xghost[1,:]
	xghost[-1,:]= xghost[-2,:]
	xghost[:,0] = xghost[:,1]
	xghost[:,-1]= xghost[:,-2]

	yghost[0,:] = yghost[1,:]
	yghost[-1,:]= yghost[-2,:]
	yghost[:,0] = yghost[:,1]
	yghost[:,-1]= yghost[:,-2]

	zghost[0,:] = zghost[1,:]
	zghost[-1,:]= zghost[-2,:]
	zghost[:,0] = zghost[:,1]
	zghost[:,-1]= zghost[:,-2]

	grid[:,:,0] = xghost
	grid[:,:,1] = yghost
	grid[:,:,2] = zghost

	h = 1./12.

	for a in range(3):
		a = a+1
		b = np.mod(a,3)+1
		c = np.mod(a+1,3)+1

		a = a-1
		b = b-1
		c = c-1
		# print(a,b,c)
		for k in range(i1[1],i2[1]+1): #along z
			for j in range(i1[0],i2[0]+1): #along x
				nhatvec[j-1,k-1,a] =  h*\
					  (grid[j+1,k,b] * (grid[j,k+1,c] + grid[j+1,k+1,c]\
					  				   -grid[j,k-1,c] - grid[j+1,k-1,c]) +\
					   grid[j-1,k,b] * (grid[j,k-1,c] + grid[j-1,k-1,c]\
									   -grid[j,k+1,c] - grid[j-1,k+1,c]) +\
					   grid[j,k+1,b] * (grid[j-1,k,c] + grid[j-1,k+1,c]\
					   				   -grid[j+1,k,c] - grid[j+1,k+1,c]) +\
					   grid[j,k-1,b] * (grid[j+1,k,c] + grid[j+1,k-1,c]\
					   				   -grid[j-1,k,c] - grid[j-1,k-1,c]) +\
					   grid[j+1,k+1,b] * (grid[j,k+1,c] - grid[j+1,k,c]) +\
					   grid[j-1,k-1,b] * (grid[j,k-1,c] - grid[j-1,k,c]) +\
					   grid[j-1,k+1,b] * (grid[j-1,k,c] - grid[j,k+1,c]) +\
					   grid[j+1,k-1,b] * (grid[j+1,k,c] - grid[j,k-1,c]) \
					   )
				# if nhatvec[j-1,k-1,a] == 0:
				# 	print(a,b,c, j, k)
				# 	print(grid[j-1:j+2,k-1,b])
				# 	print(grid[j-1:j+2,k,  b])
				# 	print(grid[j-1:j+2,k+1,b])
				# 	print(grid[j-1:j+2,k-1,c])
				# 	print(grid[j-1:j+2,k,  c])
				# 	print(grid[j-1:j+2,k+1,c])

	#normalize normal vector
	for k in range(i1[1],i2[1]+1): #along z
		for j in range(i1[0],i2[0]+1): #along x
			area[j-1,k-1] = np.sqrt(nhatvec[j-1,k-1,0]**2 + \
				   		   			nhatvec[j-1,k-1,1]**2 + \
						   			nhatvec[j-1,k-1,2]**2)
			if area[j-1,k-1] < 1e-3:
				print(j,k,area)
			nhatvec[j-1,k-1,:] = nhatvec[j-1,k-1,:]/area[j-1,k-1]

	return nhatvec, area

if __name__=='__main__':
	fin = 'rough_profile/'
	fault=Plot3d()
	fault.read_plot3d_ascii(fin+'faultnew.xyz')

	# print(fault.x[0].shape) #(1681, 351, 1)
	i1 = (1, 1) #not include ghost nodes
	i2 = fault.block_shapes[0][::-1][1:][::-1]

	import time
	tic = time.perf_counter()
	nhat,area=fault_normal_vector(fault.x[0][0,:,:].T,\
							 fault.y[0][0,:,:].T,\
							 fault.z[0][0,:,:].T,\
							 i1,i2)
	toc = time.perf_counter()
	print(f"1st run {toc - tic:0.4f} seconds")

	# tic = time.perf_counter()
	# nhat2,area2=fault_normal_vector_custom1(fault.x[0][0,:,:].T,\
	# 						 fault.y[0][0,:,:].T,\
	# 						 fault.z[0][0,:,:].T,\
	# 						 i1,i2)
	# toc = time.perf_counter()
	# print(f"2nd run {toc - tic:0.4f} seconds")
	writebin(fin+'rnhat_x',nhat[:,:,0].T)
	writebin(fin+'rnhat_y',nhat[:,:,1].T)
	writebin(fin+'rnhat_z',nhat[:,:,2].T)

	# import matplotlib
	# import matplotlib.pyplot as plt
	# import matplotlib.cm as cm
	# plt.subplot(411)
	# im = plt.imshow(nhat[:,:,0].T-nhat2[:,:,0].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(412)
	# im = plt.imshow(nhat[:,:,1].T-nhat2[:,:,1].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(413)
	# im = plt.imshow(nhat[:,:,2].T-nhat2[:,:,2].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(414)
	# im = plt.imshow(area.T-area2.T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q



	# plt.subplot(422)
	# im = plt.imshow(nhat2[:,:,0].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(424)
	# im = plt.imshow(nhat2[:,:,1].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(426)
	# im = plt.imshow(nhat2[:,:,2].T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q

	# plt.subplot(428)
	# im = plt.imshow(area2.T, interpolation=None, origin='upper',\
	# 				cmap=cm.jet)
	# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8) #Q



	# plt.show()





