#!/bin/bash

# ANTS Registration
#-----------------------
f_in="$1"
m_in="$2"

imgs="$1, $2"
dim=3
transform_path="$3"
out_this="$4"

echo "Calculating rigid registeration between:"
echo ${f_in}
echo ${m_in}


antsApplyTransforms -d $dim -i $m_in -r $f_in -t ${transform_path} -o ${out_this} --float 1 
echo "Rigid transformation computed and results is saved to:"
echo ${out_this}

