module load gcc/7.1.0-fasrc01 openmpi/2.1.0-fasrc02
export PATH=$HOME/software/PLUMED/2.4.3-gcc-new/bin:$PATH
export LD_LIBRARY_PATH=$HOME/software/tensorflow/lib:$LD_LIBRARY_PATH

if [ -f colv ]; then
    rm colv
fi

cat >plumed.dat <<EOF
t1: TORSION ATOMS=5,7,9,15 NOPBC
t2: TORSION ATOMS=7,9,15,17 NOPBC
a: ANN ARG=t1,t2 MODELPATH=linear_simple INPUT=x OUTPUT=nn_return GRAD=grad_return
PRINT ARG=* STRIDE=1 FILE=colv
FLUSH STRIDE=1
EOF
echo $i
plumed driver --plumed plumed.dat --ixyz data.xyz --length-units A 2>&1 | tee log
