rm *.so *.out 

# f2py3 --f90flags='-fcheck=all -fbacktrace -Wall' --verbose --opt='-O3' \
# -I'/opt/fftw3/include' -L'/opt/fftw3/lib' -l'fftw3' --fcompiler='gfortran' \
# -c andrewsbarall_scenario.f90 -m andrewsbarall_scenario

# Frontera
module load fftw3

 f2py3 --f90flags='-fcheck=all -fbacktrace -Wall' --verbose --opt='-O3' \
 -L$TACC_FFTW3_LIB -lfftw3f -I$TACC_FFTW3_INC --fcompiler='gfortran' \
 -c andrewsbarall_scenario.f90 -m andrewsbarall_scenario

# python run_andrewsbarall.py


# gfortran -g -fcheck=all -fbacktrace -Wall -I/opt/fftw3/include -L/opt/fftw3/lib -lfftw3 \
# andrewsbarall_scenario.f90  -o run
# cat gausssurf.out 
