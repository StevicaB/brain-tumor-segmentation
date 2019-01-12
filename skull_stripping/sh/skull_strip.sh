#!/bin/bash

# ANTS Registration
#-----------------------
m="$2"
out="$3"
f_in="$1"
m_in=${m}/atlas_pd.nii
name="$4"
out_this=${out}/${name}_atlas_reg.nii 
imgs=" $f_in, $m_in"
its=10000x1111x5
dim=3

echo "Skull stripping using:"
echo ${f_in}
echo ${m_in}
antsRegistration -d $dim -r [ $imgs ,1] \
                         -m mattes[  $imgs , 1 , 32, regular, 0.05 ] \
                         -t translation[ 0.1 ] \
                         -c [1000,1.e-8,20]  \
                         -s 4vox  \
                         -f 6 -l 1 \
                         -m mattes[  $imgs , 1 , 32, regular, 0.1 ] \
                         -t rigid[ 0.1 ] \
                         -c [1000x1000,1.e-8,20]  \
                         -s 4x2vox  \
                         -f 4x2 -l 1 \
                         -m mattes[  $imgs , 1 , 32, regular, 0.1 ] \
                         -t affine[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                         -s 4x2x1vox  \
                         -f 3x2x1 -l 1 \
                         -o ${out_this}


antsApplyTransforms -d $dim -i $m_in -r $f_in -t ${out_this}0GenericAffine.mat -o ${out_this}.nii.gz --float 1 
echo "tranformation computed"
echo ${d}


# Apply transformation to atlas: mask, wm, gm, csf
echo "Apply tranformation to T1 tissues"
labels=(mask wm gm csf) 
for label in ${labels[*]}
     do
     m_tmp=${m}/atlas_${label}.nii
     out_tmp=${out}/${name}_${label}.nii
     antsApplyTransforms -d $dim -i $m_tmp -r $f_in -t ${out_this}0GenericAffine.mat -o ${out_tmp}.nii.gz --float 1
     echo "${label}"
done

rm ${out_this}0GenericAffine.mat