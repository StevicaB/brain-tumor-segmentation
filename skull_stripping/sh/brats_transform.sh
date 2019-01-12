#!/bin/bash

# ANTS Registration
#-----------------------
f_in="$1"
m_in="$2"

imgs="$1, $2"
dim=3
out_this="$3"

echo "Calculating brats transform between:"
echo ${f_in}
echo ${m_in}

antsRegistration -d $dim -r [ $imgs ,1] \
                         -m mattes[  $imgs , 1 , 32, regular, 0.1 ] \
                         -t rigid[ 0.1 ] \
                         -c [1000x500,1.e-6,10]  \
                         -s 4x2vox  \
                         -f 4x2 -l 1 \
                         -o ${out_this}


echo "Transform path"
echo ${out_this}