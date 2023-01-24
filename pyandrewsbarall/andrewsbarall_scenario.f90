!     This code was used for the calculations reported in
 
! Code converted using TO_F90 by Alan Miller
! Date: 2021-10-02  Time: 16:01:55
 
!     "Specifying Initial Stress for Dynamic Heterogeneous
!     Earthquake Source Models"
!     by D. J. Andrews and Michael Barall.
!     This code is not subject to copyright.
!     The code is furnished only as an example for development.
!     The target rupture length is half the array length, lx/2
!     times the dimensional grid interval, dx.



subroutine initiate_scenario_nonpower2(lx,lz,asperity,hetero,tn_eff,dx,&
                cutratio,depthcut,lengthcut,nucl,iseed,&
          wout,shearstress,normalstress,statfric,dynafric,&
          istress,jstress)
IMPLICIT None
integer,intent(in) :: lx
integer,intent(in) :: lz
integer,intent(in) :: cutratio
integer,intent(in) :: iseed(4)
!     Make this flag false to select the high-stress model.
!     Make this flag true to select the low-stress asperity model.
logical,intent(in) :: asperity,hetero
real,intent(in) :: dx,nucl,tn_eff
real,intent(in) :: depthcut
real,intent(in) :: lengthcut
REAL,intent(out) :: wout(lx,lz),&
                    shearstress(lx,lz),&
                    normalstress(lx,lz),&
                    statfric(lx,lz),&
                    dynafric(lx,lz)
                    
integer,intent(out) :: istress, jstress
real,dimension(:,:),allocatable:: w
COMPLEX,dimension(:,:),allocatable :: transf
! REAL :: shearstress(lx,lz) 
! REAL :: stressdrop(lx,lz)

REAL :: stressvis(lx,lz),stressdrop(lx,lz)

REAL :: rho, rhowater, grav, fricdyn, friczero, fricstatmin
REAL :: fricstatfac, scale,lengthmid,ddepth

INTEGER :: nseed
PARAMETER ( nseed = 4 ) ! Number of random seeds
INTEGER :: kcut, npole
! LOGICAL :: hetero
REAL :: hurst

INTEGER :: i, j, ifric, jfric, i2
REAL :: depth, taudyn, filter, stressratio,&
                    filterx
REAL :: stresspeak, fricpeak

REAL :: fricasper, fraction
REAL :: chopout, fracout
real :: signorm(lz)

integer :: zendindex
integer :: nwin


!!!!!!!!!!!
nwin = int(nucl/dx)
rho=3000.               ! Density of rock, kg/m^3
rhowater=1000.          ! Density of water, kg/m^3
grav=9.8                ! Acceleration of gravity, m/s^2

! dx=60.                  ! Horizontal cell size, m
ddepth=dx               ! Vertical cell size, m

IF (asperity) THEN
  fricdyn=0.1           ! Dynamic friction coefficient
  fricasper=0.4         ! Shear/normal stress within asperities
  fraction=1./8.        ! Fraction of fault occupied by asperity
  fricstatmin=0.70      ! Min allowed static friction (f_sref)
  fricstatfac=1.01      ! Min yield/shear stress ratio (1+epsilon)
  scale=fricasper-fricdyn !Stress fluctuation scale factor (alpha)
  ! depthcut=14.0E3       ! Cutoff for depth conditioning, m
ELSE
  fricdyn=0.6           ! Dynamic friction coefficient
  friczero=0.65         ! Shear/normal stress for unit fluctuation
  fricstatmin=0.7      ! Min allowed static friction (f_sref)
  fricstatfac=1.01      ! Min yield/shear stress ratio (1+epsilon)
  scale=friczero-fricdyn ! Stress fluctuation scale factor (alpha)
  ! depthcut=23.5E3       ! Cutoff for depth conditioning, m
END IF

!     Establish program parameters
!     The seed for the random number generator is 4 integers long



! iseed(1)=irand()    ! Seed for random number generator
! iseed(2)=irand()    ! Seed for random number generator
! iseed(3)=irand()    ! Seed for random number generator
! iseed(4)=irand()    ! Seed for random number generator

! hetero=.true.    ! If false, calculate lowest mode only
hurst=0.0        ! Appropriate for self-similar stress
npole=4          ! Order of low-pass Butterworth filter

IF (asperity) THEN
  kcut=lx/4      ! Low-pass at factor of 2 below Nyquist
ELSE
  kcut=lx/cutratio/2     ! Low-pass at factor of 8 below Nyquist
END IF

!     Generate a self-affine function, lowest mode being one wavelength
!      in x-direction with unit amplitude
if (lx >= lz) then
  allocate(w(lx,lx))
  allocate(transf(lx,lx))
  CALL saffgauss2(w,transf,lx,lx,hetero,hurst,kcut,npole,iseed,nseed)
else
  allocate(w(lx,lz))
  allocate(transf(lx,lz))
  CALL saffgauss2(w,transf,lx,lz,hetero,hurst,kcut,npole,iseed,nseed)
end if


OPEN (UNIT=4,FILE='gausssurf_nonpower2.out',FORM='formatted',STATUS='replace')

WRITE (4,1002) lx, kcut, iseed(1), iseed(2), iseed(3), iseed(4), hurst

1002 FORMAT (' lx    =',i14/' kcut  =',i14/ ' iseed =',4I14/' hurst =',f14.3)
close(UNIT=4)

!     Generate static friction, initial shear stress, and stress drop
lengthmid = dx * (lx - 1) / 2

call normal_stress(depth,signorm,dx,lz)
if (tn_eff>0) then
  signorm = min(signorm, tn_eff)
end if

DO j=1,lz
  depth=(DBLE(j-0.5))*ddepth            ! depth at top of cell
  ! signorm=grav*(rho-rhowater)*depth   ! overburden pressure (constant gradient)
  
  taudyn=fricdyn*signorm(j)              ! dynamic shear stress
  filter= 1.0D0/(1.0D0+(depth/depthcut)**4) ! depth-conditioning
  DO i=1,lx
    
    if (abs(lengthmid-(i-0.5)*dx)<lengthcut) then
      filterx = 1
    else
      filterx = 1.0D0/(((abs(lengthmid-(i-0.5)*dx)/lengthcut)**2))
    endif

    wout(i,j) = w(i,j)
    stressratio=filter*(scale*w(i,j)+fricdyn)*filterx ! shear/normal ratio
    shearstress(i,j) = stressratio    
    IF (stressratio < 0.0D0) stressratio = 0.0D0
    statfric(i,j)=MAX(fricstatmin,fricstatfac*SNGL(stressratio))
    shearstress(i,j)=signorm(j)*stressratio
    stressdrop(i,j)=signorm(j)*stressratio-taudyn
    normalstress(i,j)=signorm(j)
    dynafric(i,j)=fricdyn
    stressvis(i,j)=(stressratio)
  END DO
END DO

!     Find locations of peak stress drop and peak static friction

stresspeak=0.
fricpeak=0.
istress=0
jstress=0
ifric=0
jfric=0

! nwin = 0
DO j=1+nwin,lz-nwin,nwin/10
  DO i=1+nwin,lx-nwin,nwin/10

    IF (sum(stressdrop(i-nwin:i+nwin,j-nwin:j+nwin)) > stresspeak) THEN
      stresspeak=sum(stressdrop(i-nwin:i+nwin,j-nwin:j+nwin))
      istress=i
      jstress=j
    END IF
    IF (statfric(i,j) > fricpeak) THEN
      fricpeak=statfric(i,j)
      ifric=i
      jfric=j
    END IF
  END DO
END DO

istress = istress - 1 !convert horizontal index range for python
jstress = jstress - 1 !convert vertical index range for python
end subroutine


!*************************************************
 subroutine velstru(depth,rou0)
 !*************************************************
 implicit none

 real,intent(in) :: depth ! in km, positive below surface
 real,intent(out) :: rou0 ! SI unit
 real :: vp0,vs0

 real :: zandrews

 zandrews=depth+0.0073215 ! in km
 if(zandrews<0.03) then
  vs0=2.206*zandrews**0.272
 elseif(zandrews<0.19) then
  vs0=3.542*zandrews**0.407
 elseif(zandrews<4.) then
  vs0=2.505*zandrews**0.199
 elseif(zandrews<8.) then
  vs0=2.927*zandrews**0.086
 else
  vs0=2.927*8.**0.086
 end if
 vp0=max(1.4+1.14*vs0,1.68*vs0)
 rou0=2.4405+0.10271*vs0

 rou0=rou0*1000. ! convert g/cm^3 to kg/m^3
 vp0=vp0*1000. ! convert km/s to m/s
 vs0=vs0*1000.
 end subroutine



SUBROUTINE normal_stress(dep,sigma,dz,nz)
implicit none
real,intent(in) :: dep,dz
integer,intent(in) :: nz
real,intent(out) :: sigma(nz)

integer :: i
real :: rho,rho2

call velstru(dz/1e3/2,rho)
sigma(1) = 9.81*dz*(rho-1000)/2
do i = 2, nz
  call velstru((i-0.5)*dz,rho2)
  sigma(i) = sigma(i-1) + 9.81*dz*((rho+rho2)/2-1000)
  rho = rho2 
end do

END SUBROUTINE


!     saffgauss - Generate a self-affine random function
!     Input   lx      dimension of square array, power of 2
!             hurst   Hurst exponent
!             kcut    cutoff for low-pass filter, integer, Nyquist = lx/2
!             npole   order of Butterworth low-pass filter
!             iseed   seeds for random number generator, positive
!             nseed   number of seeds for random number generator
!     Scratch transf  working storage for Fourier transform
!     Output  w       self-affine function of two variables

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SUBROUTINE saffgauss2 (w, transf, lx, lz, hetero, hurst, kcut, npole, iseed, nseed)


use, intrinsic :: iso_c_binding
include 'fftw3.f03'


REAL, INTENT(OUT)                        :: w(lx,lz)
COMPLEX, INTENT(OUT)                     :: transf(lx,lz)
INTEGER, INTENT(IN)                      :: lx
INTEGER, INTENT(IN)                      :: lz
LOGICAL, INTENT(IN)                  :: hetero
REAL, INTENT(IN)                         :: hurst
INTEGER, INTENT(IN)                      :: kcut
INTEGER, INTENT(IN)                  :: npole
INTEGER, INTENT(IN)                  :: iseed(nseed)
INTEGER, INTENT(IN)                  :: nseed



COMPLEX :: xran, yran
INTEGER :: i, j, nn(2)
REAL :: hrttwo, v1, v2, coef
DOUBLE PRECISION :: fcsq,fsq, pspec, filter, pony

integer :: ik,jk,lhx, lhz, ksq
real :: scale
type(C_PTR) :: plan
complex(C_DOUBLE_COMPLEX), dimension(lx,lz) :: in, out
!     Initialize random number generator


CALL init_by_array (iseed, nseed)

!     Initialize some parameters

scale = real(lx)/real(lz)

nn(1)=lx              ! array size for FFT
nn(2)=lz              ! array size for FFT
lhx=lx/2               ! wavenumber for Nyquist frequency
lhz=lz/2               ! wavenumber for Nyquist frequency
fcsq=kcut*kcut        ! square of filter cutoff wavenumber
hrttwo=0.5*SQRT(2.)   ! factor to obtain variance 0.5
pony=hurst+1.         ! exponent of power-law spectral transform

!     Real and imaginary parts of transf are each random with
!      Gaussian distribution with variance of 1./2.




DO j=1,lz
  DO i=1,lx
    IF (hetero) THEN
      CALL gaussrand(v1,v2)
      xran=(1.,0.)*hrttwo*v1
      yran=(0.,1.)*hrttwo*v2
      transf(i,j)=xran+yran
    ELSE
      transf(i,j)=(0.,0.)
    END IF
    
  END DO
END DO




DO j=1,lz
  DO i=1,lx
!       Find square of wavenumber
    ik=i-1
    jk=j-1
    IF (ik > lhx) ik=ik-lx
    IF (jk > lhz) jk=jk-lz
    ! ksq=ik*ik+jk*jk
    fsq=ik*ik+jk*jk*scale*scale

    IF (ik==0 .and.jk==0) THEN
!         Mode (0,0) has zero amplitude
      transf(i,j)=(0.,0.)
      pspec=0.0D0
    else if((ik==1 .or. ik==-1) .and.jk==0) then
        transf(i,j)=(-2.,0.)
        pspec=1.0D0/fsq**pony
    else if(ik==0 .and. (jk==-1 .or. jk==1)) then
        transf(i,j)=(0.,0.)
        pspec=1.0D0/fsq**pony
    ELSE IF ((ik==-1 .or. ik==1) .and. &
             (jk==-1 .or. jk==1)) THEN
      transf(i,j)=(0.,0.)
      pspec=1.0D0/fsq**pony
    ELSE
!         Higher modes have random phase
      pspec=1.0D0/fsq**pony
    END IF




!       Multiply transform by desired power spectrum (pspec) and filter
    filter=1.0D0/(1.0D0+(fsq/fcsq)**npole)  ! low-pass filter
    transf(i,j)=SQRT(pspec*filter)*transf(i,j)

  END DO
END DO


!     Inverse Fourier transform

! CALL fft_ndim(transf,nn,2,-1)

in = transf
plan = fftw_plan_dft_2d(lz,lx, in,out, FFTW_FORWARD,FFTW_ESTIMATE)
call fftw_execute_dft(plan, in, out)
call fftw_destroy_plan(plan)
transf = out


!     Form self-affine function
!     normalized by sum of mode 1 Fourier coefficients

coef=1./4.
DO j=1,lz
  DO i=1,lx
    w(i,j)=coef*transf(i,j)

  END DO
END DO

RETURN
END SUBROUTINE saffgauss2



!     gaussrand - Return a pair of Gaussian random values,
!                 with zero mean and unit variance.
!     Note:  You must initialize the random number generator first.
!     Note:  Returned values cannot exceed 9.20 in magnitude.
!     Implementation note:  This routine uses the polar method,
!      which produces Gaussian random values as output given
!      uniform random values as input.  The polar method requires
!      discarding some of the uniform random values.  The discard
!      logic is implemented entirely with integer arithmetic, so
!      this generator is guaranteed to produce the same sequence of
!      values on any computer, assuming it receives the same input.

SUBROUTINE gaussrand (v1, v2)

IMPLICIT NONE
REAL, INTENT(OUT)                        :: v1
REAL, INTENT(OUT)                        :: v2





INTEGER, PARAMETER :: msign = 1073741824
INTEGER, PARAMETER :: mlow14 = 16383
INTEGER, PARAMETER :: mlow15 = 32767
INTEGER, PARAMETER :: mlow16 = 65535
INTEGER, PARAMETER :: mlow30 = 1073741823
DOUBLE PRECISION, PARAMETER :: amsign = 1073741824.0D0

INTEGER :: genrand_int31
INTEGER :: iv1, iv2, iv1_lo, iv2_lo, iv1_hi, iv2_hi, ir_lo, ir_hi
INTEGER :: iv1_mag, iv2_mag
DOUBLE PRECISION :: rsq, fac

! Get two uniform random values in range 0 to 2**31 - 1
! We will treat them as a sign bit, plus a magnitude ranging
! from 0 to 2**30 - 1.  We impose an offset of 0.5, so that
! the value is considered to be ((-1)**sign)*(mag + 0.5).

10   iv1 = genrand_int31()
iv2 = genrand_int31()

! Get the magnitudes, and split into lower and upper 15 bits

iv1_mag = IAND(iv1, mlow30)
iv1_lo = IAND(iv1_mag, mlow15)
iv1_hi = ishft(iv1_mag, -15)

iv2_mag = IAND(iv2, mlow30)
iv2_lo = IAND(iv2_mag, mlow15)
iv2_hi = ishft(iv2_mag, -15)

! Compute the polar radius squared, which is
! (mag1 + 0.5)**2 + (mag2 + 0.5)**2
!  = mag1**2 + mag2**2 + mag1 + mag2 + 0.5
! It is stored 30 bits in lower word and 31 bits in upper word,
! with the final +0.5 implied.
! First compute mag1**2 + mag2**2.

ir_lo = iv1_lo*iv1_lo + iv2_lo*iv2_lo
! <= 2**31 - 2**17 + 2**1
ir_hi = iv1_lo*iv1_hi + iv2_lo*iv2_hi + ishft(ir_lo, -16)
! <= 2**31 - 2**17 + 2**15
ir_lo = ior(IAND(ir_lo, mlow16), ishft(IAND(ir_hi, mlow14), 16))
! <= 2**30 - 1
ir_hi = iv1_hi*iv1_hi + iv2_hi*iv2_hi + ishft(ir_hi, -14)
! <= 2**31 - 2**3 + 2**2

! Now add mag1 + mag2

ir_lo = ir_lo + iv1_mag
ir_hi = ir_hi + ishft(ir_lo, -30)
ir_lo = IAND(ir_lo, mlow30)

ir_lo = ir_lo + iv2_mag
ir_hi = ir_hi + ishft(ir_lo, -30)
ir_lo = IAND(ir_lo, mlow30)

! Discard if polar radius squared >= 2**60

IF (ir_hi >= msign) THEN
  GO TO 10
END IF

! Square of polar radius, normalized, in floating-point

rsq = (((0.5D0 + DBLE(ir_lo))/amsign) + DBLE(ir_hi))/amsign

! Gaussian factor

fac = SQRT(MAX(0.0D0, -2.0D0*LOG(rsq)/rsq))

! Return values, with proper sign

IF (IAND(iv1, msign) == 0) THEN
  v1 = ((0.5D0 + DBLE(iv1_mag))/amsign)*fac
ELSE
  v1 = -((0.5D0 + DBLE(iv1_mag))/amsign)*fac
END IF

IF (IAND(iv2, msign) == 0) THEN
  v2 = ((0.5D0 + DBLE(iv2_mag))/amsign)*fac
ELSE
  v2 = -((0.5D0 + DBLE(iv2_mag))/amsign)*fac
END IF

RETURN
END SUBROUTINE gaussrand



!     fft_ndim - Fast Fourier transform in n dimensions.
!     Input parameters:
!      z = Data array.
!      nn = Array giving extent of the data array in each dimension.
!      ndim = Number of dimensions = Rank of data array.
!      isign = +1 for forward fft, -1 for reverse fft.
!     Each extent must be a power of 2.
!     In the data array, each adjacent pair of array elements holds
!      real and imaginary parts of a complex value (real part first).
!      The complex values are stored in column-major (Fortran) order.
!      (To use data stored in row-major (C/C++) order, reverse nn.)
!     The transformed data is stored in z, replacing the original.
!     The number N of complex values is the product of the elements
!      of nn.  The number of (real) elements of z is 2*N.
!     No scaling is applied on either forward or reverse fft.  So,
!      if you apply forward and then reverse fft (or vice versa) you
!      need to divide by N to recover the original values.
!     The sign convention is that forward fft is:
!      Z(k) = SUM(n)(z(n) * exp(2*pi*i*k.n/N))
!     and reverse fft is:
!      z(n) = SUM(k)(Z(k) * exp(-2*pi*i*k.n/N))
!     where k.n is the dot product of the index vectors k and n.

SUBROUTINE fft_ndim (z, nn, ndim, ISIGN)

IMPLICIT NONE
COMPLEX, INTENT(INOUT)                     :: z(*)
INTEGER, INTENT(IN)                      :: nn(ndim)
INTEGER, INTENT(IN)                      :: ndim
INTEGER, INTENT(IN)                  :: ISIGN




INTEGER :: nsize, idim, i_stride, next_stride, i_extent
INTEGER :: i, i_rev, i_bit, m1, m2, k1, k2, j1, j2, i1, i2
REAL :: tr, ti
DOUBLE PRECISION :: ar, ai, br, bi, wr, wi, theta

! calculate total size of array, in elements

nsize = 2    ! 2 elements for real and imaginary parts
DO idim = 1, ndim
  nsize = nsize*nn(idim)
END DO

! loop over dimensions

next_stride = 2

DO idim = 1, ndim
  i_stride = next_stride    ! element stride in current dim
  i_extent = nn(idim)    ! element extent in current dim
  next_stride = i_stride*i_extent   ! element stride in next dim
  
! bit reversal loop
  
  i_rev = i_extent/2    ! value of i, bit-reversed
  DO i = 1, i_extent-2
    
    IF (i < i_rev) THEN
      DO m1 = 0, nsize - next_stride, next_stride
        DO m2 = 0, i_stride - 2, 2
          k1 = i*i_stride + m1 + m2 + 1
          k2 = i_rev*i_stride + m1 + m2 + 1
          
! swap elements at bit-reversed indexes
          
          tr = z(k1)
          ti = z(k1 + 1)
          z(k1) = z(k2)
          z(k1 + 1) = z(k2 + 1)
          z(k2) = tr
          z(k2 + 1) = ti
          
        END DO
      END DO
    END IF
    
! increment i_rev, with bits reversed
    
    i_bit = i_extent/2
    10       IF (i_bit <= i_rev) THEN
      i_rev = i_rev - i_bit
      i_bit = i_bit/2
      GO TO 10
    END IF
    i_rev = i_rev + i_bit
    
  END DO    ! end bit reversal loop
  
! fft loop
  
  theta = 6.283185307179586476925286766559D0    ! 2*pi
  IF (ISIGN < 0) THEN
    theta = -theta
  END IF
  
  j2 = 1
  20     IF (j2 < i_extent) THEN
    j1 = j2
    j2 = j2*2
    
    theta = theta * 0.5D0
    
    ar = SIN(theta * 0.5D0)
    ar = -2.0D0 * ar * ar    ! ar = cos(theta) - 1
    ai = SIN(theta)
    
    br = 1.0D0
    bi = 0.0D0
    
    DO i1 = 0, j1 - 1
      DO i2 = 0, i_extent - j2, j2
        DO m1 = 0, nsize - next_stride, next_stride
          DO m2 = 0, i_stride - 2, 2
            k1 = (i1 + i2)*i_stride + m1 + m2 + 1
            k2 = (i1 + i2 + j1)*i_stride + m1 + m2 + 1
            
! butterfly transform
            
            tr = br*z(k2) - bi*z(k2 + 1)
            ti = br*z(k2 + 1) + bi*z(k2)
            z(k2) = z(k1) - tr
            z(k2 + 1) = z(k1 + 1) - ti
            z(k1) = z(k1) + tr
            z(k1 + 1) = z(k1 + 1) + ti
            
          END DO
        END DO
      END DO
      
      wr = br*ar - bi*ai
      wi = br*ai + bi*ar
      br = br + wr
      bi = bi + wi
      
    END DO
    
    GO TO 20
  END IF    ! end fft loop
  
END DO    ! end loop over dimensions

RETURN
END SUBROUTINE fft_ndim








!
 
! Code converted using TO_F90 by Alan Miller
! Date: 2021-10-02  Time: 16:00:27
 
!  A C-program for MT19937, with initialization improved 2002/1/26.
!  Coded by Takuji Nishimura and Makoto Matsumoto.

!  Before using, initialize the state by using init_genrand(seed)
!  or init_by_array(init_key, key_length).

!  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
!  All rights reserved.
!  Copyright (C) 2005, Mutsuo Saito,
!  All rights reserved.

!  Redistribution and use in source and binary forms, with or without
!  modification, are permitted provided that the following conditions
!  are met:

!    1. Redistributions of source code must retain the above copyright
!       notice, this list of conditions and the following disclaimer.

!    2. Redistributions in binary form must reproduce the above copyright
!       notice, this list of conditions and the following disclaimer in the
!       documentation and/or other materials provided with the distribution.

!    3. The names of its contributors may not be used to endorse or promote
!       products derived from this software without specific prior written
!       permission.

!  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
!  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
!  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
!  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
!  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
!  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
!  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
!  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
!  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
!  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


!  Any feedback is very welcome.
!  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
!  email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)

!-----------------------------------------------------------------------
!  FORTRAN77 translation by Tsuyoshi TADA. (2005/12/19)

!     ---------- initialize routines ----------
!  subroutine init_genrand(seed): initialize with a seed
!  subroutine init_by_array(init_key,key_length): initialize by an array

!     ---------- generate functions ----------
!  integer function genrand_int32(): signed 32-bit integer
!  integer function genrand_int31(): unsigned 31-bit integer
!  double precision function genrand_real1(): [0,1] with 32-bit resolution
!  double precision function genrand_real2(): [0,1) with 32-bit resolution
!  double precision function genrand_real3(): (0,1) with 32-bit resolution
!  double precision function genrand_res53(): (0,1) with 53-bit resolution

!  This program uses the following non-standard intrinsics.
!    ishft(i,n): If n>0, shifts bits in i by n positions to left.
!                If n<0, shifts bits in i by n positions to right.
!    iand (i,j): Performs logical AND on corresponding bits of i and j.
!    ior  (i,j): Performs inclusive OR on corresponding bits of i and j.
!    ieor (i,j): Performs exclusive OR on corresponding bits of i and j.

!  Modified by Michael Barall (2010/03/17)
!  - In the real functions, removed assumption that integers are
!     represented in two's complement form.
!  - Added "implicit none" to all routines.
!  - Changed init_key so its lower index is 1.

!-----------------------------------------------------------------------
!     initialize mt(0:N-1) with a seed
!-----------------------------------------------------------------------

SUBROUTINE init_genrand(s)

IMPLICIT NONE
INTEGER, INTENT(IN)                  :: s


INTEGER :: allbit_mask
INTEGER, PARAMETER :: n=624
INTEGER, PARAMETER :: done=123456789
INTEGER :: mti,initialized
INTEGER :: mt(0:n-1)
COMMON /mt_state1/ mti,initialized
COMMON /mt_state2/ mt
COMMON /mt_mask1/ allbit_mask

CALL mt_initln
mt(0)=IAND(s,allbit_mask)
DO  mti=1,n-1
  mt(mti)=1812433253* IEOR(mt(mti-1),ishft(mt(mti-1),-30))+mti
  mt(mti)=IAND(mt(mti),allbit_mask)
END DO
initialized=done

RETURN
END SUBROUTINE init_genrand
!-----------------------------------------------------------------------
!     initialize by an array with array-length
!     init_key is the array for initializing keys
!     key_length is its length
!-----------------------------------------------------------------------

SUBROUTINE init_by_array(init_key,key_length)
IMPLICIT NONE
INTEGER, INTENT(IN)                  :: init_key(key_length)
INTEGER, INTENT(IN)                  :: key_length




INTEGER :: allbit_mask
INTEGER :: topbit_mask
INTEGER, PARAMETER :: n=624
INTEGER :: i,j,k
INTEGER :: mt(0:n-1)
COMMON /mt_state2/ mt
COMMON /mt_mask1/ allbit_mask
COMMON /mt_mask2/ topbit_mask

CALL init_genrand(19650218)
i=1
j=0
DO  k=MAX(n,key_length),1,-1
  mt(i)=IEOR(mt(i),IEOR(mt(i-1),ishft(mt(i-1),-30))*1664525) +init_key(j+1)+j
  mt(i)=IAND(mt(i),allbit_mask)
  i=i+1
  j=j+1
  IF(i >= n)THEN
    mt(0)=mt(n-1)
    i=1
  END IF
  IF(j >= key_length)THEN
    j=0
  END IF
END DO
DO  k=n-1,1,-1
  mt(i)=IEOR(mt(i),IEOR(mt(i-1),ishft(mt(i-1),-30))*1566083941)-i
  mt(i)=IAND(mt(i),allbit_mask)
  i=i+1
  IF(i >= n)THEN
    mt(0)=mt(n-1)
    i=1
  END IF
END DO
mt(0)=topbit_mask

RETURN
END SUBROUTINE init_by_array
!-----------------------------------------------------------------------
!     generates a random number on [0,0xffffffff]-interval
!-----------------------------------------------------------------------

FUNCTION genrand_int32()
IMPLICIT NONE
INTEGER :: genrand_int32
INTEGER :: n,m
INTEGER :: done
INTEGER :: upper_mask,lower_mask,matrix_a
INTEGER :: t1_mask,t2_mask
PARAMETER (n=624)
PARAMETER (m=397)
PARAMETER (done=123456789)
INTEGER :: mti,initialized
INTEGER :: mt(0:n-1)
INTEGER :: y,kk
INTEGER :: mag01(0:1)
COMMON /mt_state1/ mti,initialized
COMMON /mt_state2/ mt
COMMON /mt_mask3/ upper_mask,lower_mask,matrix_a,t1_mask,t2_mask
COMMON /mt_mag01/ mag01

IF(initialized /= done)THEN
  CALL init_genrand(21641)
END IF

IF(mti >= n)THEN
  DO  kk=0,n-m-1
    y=ior(IAND(mt(kk),upper_mask),IAND(mt(kk+1),lower_mask))
    mt(kk)=IEOR(IEOR(mt(kk+m),ishft(y,-1)),mag01(IAND(y,1)))
  END DO
  DO  kk=n-m,n-1-1
    y=ior(IAND(mt(kk),upper_mask),IAND(mt(kk+1),lower_mask))
    mt(kk)=IEOR(IEOR(mt(kk+(m-n)),ishft(y,-1)),mag01(IAND(y,1)))
  END DO
  y=ior(IAND(mt(n-1),upper_mask),IAND(mt(0),lower_mask))
  mt(kk)=IEOR(IEOR(mt(m-1),ishft(y,-1)),mag01(IAND(y,1)))
  mti=0
END IF

y=mt(mti)
mti=mti+1

y=IEOR(y,ishft(y,-11))
y=IEOR(y,IAND(ishft(y,7),t1_mask))
y=IEOR(y,IAND(ishft(y,15),t2_mask))
y=IEOR(y,ishft(y,-18))

genrand_int32=y
RETURN
END FUNCTION genrand_int32
!-----------------------------------------------------------------------
!     generates a random number on [0,0x7fffffff]-interval
!-----------------------------------------------------------------------

FUNCTION genrand_int31()
IMPLICIT NONE
INTEGER :: genrand_int31
INTEGER :: genrand_int32
genrand_int31=INT(ishft(genrand_int32(),-1))
RETURN
END FUNCTION genrand_int31
!-----------------------------------------------------------------------
!     generates a random number on [0,1]-real-interval
!-----------------------------------------------------------------------

FUNCTION genrand_real1()
IMPLICIT NONE
DOUBLE PRECISION :: genrand_real1,r
INTEGER :: genrand_int32,ir
!C      r=dble(genrand_int32())
!C      if(r.lt.0.d0)r=r+2.d0**32
ir=genrand_int32()
r=DBLE(ishft(ir,-1))*2.d0+DBLE(IAND(ir,1))
genrand_real1=r/4294967295.d0
RETURN
END FUNCTION genrand_real1
!-----------------------------------------------------------------------
!     generates a random number on [0,1)-real-interval
!-----------------------------------------------------------------------

FUNCTION genrand_real2()
IMPLICIT NONE
DOUBLE PRECISION :: genrand_real2,r
INTEGER :: genrand_int32,ir
!C      r=dble(genrand_int32())
!C      if(r.lt.0.d0)r=r+2.d0**32
ir=genrand_int32()
r=DBLE(ishft(ir,-1))*2.d0+DBLE(IAND(ir,1))
genrand_real2=r/4294967296.d0
RETURN
END FUNCTION genrand_real2
!-----------------------------------------------------------------------
!     generates a random number on (0,1)-real-interval
!-----------------------------------------------------------------------

FUNCTION genrand_real3()
IMPLICIT NONE
DOUBLE PRECISION :: genrand_real3,r
INTEGER :: genrand_int32,ir
!C      r=dble(genrand_int32())
!C      if(r.lt.0.d0)r=r+2.d0**32
ir=genrand_int32()
r=DBLE(ishft(ir,-1))*2.d0+DBLE(IAND(ir,1))
genrand_real3=(r+0.5D0)/4294967296.d0
RETURN
END FUNCTION genrand_real3
!-----------------------------------------------------------------------
!     generates a random number on [0,1) with 53-bit resolution
!-----------------------------------------------------------------------

FUNCTION genrand_res53()
IMPLICIT NONE
DOUBLE PRECISION :: genrand_res53
INTEGER :: genrand_int32
DOUBLE PRECISION :: a,b
a=DBLE(ishft(genrand_int32(),-5))
b=DBLE(ishft(genrand_int32(),-6))
!C      if(a.lt.0.d0)a=a+2.d0**32
!C      if(b.lt.0.d0)b=b+2.d0**32
genrand_res53=(a*67108864.d0+b)/9007199254740992.d0
RETURN
END FUNCTION genrand_res53
!-----------------------------------------------------------------------
!     initialize large number (over 32-bit constant number)
!-----------------------------------------------------------------------

SUBROUTINE mt_initln
IMPLICIT NONE
INTEGER :: allbit_mask
INTEGER :: topbit_mask
INTEGER :: upper_mask,lower_mask,matrix_a,t1_mask,t2_mask
INTEGER :: mag01(0:1)
COMMON /mt_mask1/ allbit_mask
COMMON /mt_mask2/ topbit_mask
COMMON /mt_mask3/ upper_mask,lower_mask,matrix_a,t1_mask,t2_mask
COMMON /mt_mag01/ mag01
!C    TOPBIT_MASK = Z'80000000'
!C    ALLBIT_MASK = Z'ffffffff'
!C    UPPER_MASK  = Z'80000000'
!C    LOWER_MASK  = Z'7fffffff'
!C    MATRIX_A    = Z'9908b0df'
!C    T1_MASK     = Z'9d2c5680'
!C    T2_MASK     = Z'efc60000'
topbit_mask=1073741824
topbit_mask=ishft(topbit_mask,1)
allbit_mask=2147483647
allbit_mask=ior(allbit_mask,topbit_mask)
upper_mask=topbit_mask
lower_mask=2147483647
matrix_a=419999967
matrix_a=ior(matrix_a,topbit_mask)
t1_mask=489444992
t1_mask=ior(t1_mask,topbit_mask)
t2_mask=1875247104
t2_mask=ior(t2_mask,topbit_mask)
mag01(0)=0
mag01(1)=matrix_a
RETURN
END SUBROUTINE mt_initln

