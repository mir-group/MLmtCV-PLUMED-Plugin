PLUMED_SRC=$HOME/software/PLUMED/plumed-2.5.1
DEST=$HOME/software/PLUMED/2.5.1

# set up environment variables and the location of VMD and TENSORFLOW
export PLUMED_USE_LEPTON=yes
export TENSORFLOW=$HOME/software/tensorflow
export VMD=$HOME/software/vmd/1.9.4/plugins
export LD_LIBRARY_PATH=$TENSORFLOW/lib:$LD_LIBRARY_PATH


# patch the code to plumed
for file in tf_utils.cpp  tf_utils.hpp ANN.cpp
do
    cp src/$file $PLUMED_SRC/src/function/
done

# configs and compile
# # gcc
# import gcc and open mpi module
# module load gcc/7.1.0-fasrc01 openmpi/2.1.0-fasrc02
# # configs and compile
# cd $PLUMED_SRC
# ./configure \
#     CXX=mpic++ MPIF90=mpif90 F70=mpif90 \
#     LDFLAGS="-L$VMD/LINUXAMD64/molfile -lm -ldl -L$TENSORFLOW/lib -ltensorflow -ltensorflow_framework" \
#     CPPFLAGS="-I$VMD/include -I$VMD/ARCH/molfile -I$TENSORFLOW/include" \
#     --prefix=$HOME/software/PLUMED/2.4.3-gcc-new \
#     --disable-openmp PLUMED_USE_LEPTON=yes
#
# # start compilation
# filename=make.log$(date +%Y-%m-%d-%H-%M-%S)
# make -j 16 2>&1 | tee $filename
# make install| tee -a $filename
# echo $PLUMED_SRC/$filename

# intel

module load intel/17.0.4-fasrc01 intel-mkl/2017.2.174-fasrc01   impi/2017.2.174-fasrc01 fftw/3.3.7-fasrc02
MKL_LIB=$MKLROOT/lib/intel64
LDFLAGS="-L$HOME/software/vmd/1.9.4/plugins/LINUXAMD64/molfile -L$MKL_LIB -lmkl_intel_lp64  -lmkl_sequential -lmkl_core -L$MKL_LIB -I$MKLROOT/include -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -L$FFTW_LIB -I$FFTW_INCLUDE -lfftw3 -L$TENSORFLOW/lib -ltensorflow -ltensorflow_framework"
CPPFLAGS="-I$HOME/software/vmd/1.9.4/plugins/include -I$HOME/software/vmd/1.9.4/plugins/ARCH/molfile -I$TENSORFLOW/include"
CC="mpiicc $LDFLAGS $CPPFLAGS"
CXX="mpiicpc $LDFLAGS $CPPFLAGS"

cd $PLUMED_SRC
./configure CC="$CC" CXX="$CXX" MPIF90=mpiifort F70=mpiifort --enable-modules=all \
            LDFLAGS="$LDFLAGS" CPPFLAGS="$CPPFLAGS" \
            --prefix=$DEST # --enable-static-patch

# start compilation
filename=make.log$(date +%Y-%m-%d-%H-%M-%S)
make -j 4 2>&1 | tee $filename
make install| tee -a $filename
echo $PLUMED_SRC/$filename
