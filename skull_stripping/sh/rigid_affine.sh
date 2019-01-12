#!/bin/sh

# ANTS Registration
#-----------------------
f_in="$1"
m_in="$2"

imgs="$1, $2"
dim=3
out_this="$3"

echo "Calculating rigid registeration between:"
echo ${f_in}
echo ${m_in}

antsRegistration -d $dim -r [ $imgs ,1] \
                         -m mattes[  $imgs , 1 , 32, regular, 0.1 ] \
                         -t rigid[ 0.1 ] \
                         -c [1000x500,1.e-6,10]  \
                         -s 4x2vox  \
                         -f 4x2 -l 1 \
                         -o ${out_this}

antsApplyTransforms -d $dim -i $m_in -r $f_in -t ${out_this}0GenericAffine.mat -o ${out_this} --float 1 
echo "Rigid transformation computed and results is saved to:"
echo ${out_this}

rm ${out_this}0GenericAffine.mat