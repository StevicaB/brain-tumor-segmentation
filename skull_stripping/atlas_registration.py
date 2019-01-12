'''
Created on May 23, 2017

@author: Esther Alberts
'''
import os
import numpy as np
import time

import registration as reg
import paths

def niftireg_nonrigid_atlas_registration(t1_patient_path,
                                         save_dir, 
                                         masked=True):
    """ Register gm, wm, csf and atlas mask to the given t1 image using
    a t1 atlas image.

    First an atlas t1 image is affinely registered to the patient t1 image.
    Then, the atlas t1 image is non-rigidly registered to the patient t1
    image using the former affine transformation as initialisation.
    Lastly, atlas tissues gm, wm, csf and the atlas mask are non-rigidly
    registered to the patient t1 image using the former non-rigid atlas
    t1 image transformation.
    
    Parameters
    ----------
    t1_patient_path : str
        path to the patient t1 image
    save_dir : str
        patient directory 
        (intermediate directories for registration results are created)
    masked : bool
        whether the supplied image is masked. If so, a masked version of 
        the t1 atlas is used for registration (recommended).

    """    

    reg_tissues = paths.get_reg_tissue_paths(save_dir, exist=False)

    if np.all([os.path.exists(path) for path in reg_tissues.values()]):
        print 'Have already been registered!'
        return reg_tissues

    # define output paths for intermediate results
    reg_dir = paths.get_reg_dir(save_dir)
    reg_dir = os.path.join(reg_dir, 'registration_intermediate')
    paths.make_dir(reg_dir)

    aff_reg = os.path.join(reg_dir, 't1_atlas_aff_reg.nii.gz')
    aff_trans = os.path.join(reg_dir, 't1_atlas_aff_transformation.txt')
    f3d_reg = os.path.join(reg_dir, 't1_atlas_f3d_reg.nii.gz')
    f3d_cpp = os.path.join(reg_dir, 't1_atlas_f3d_cpp.nii.gz')
    
    # deffine atlas paths
    mod_str = 't1'
    if masked:
        print 'I am using the masked atlas!'
        mod_str += '_masked'
    t1_atlas_path = paths.ATLAS_SPM[mod_str]
    atlas_tissues = {tissue: paths.ATLAS_SPM[tissue]['high_uint8']
                     for tissue in paths.TISSUES}

    # affine registration of t1 atlas to this image (1-2 min)
    print 'Affine registration from t1 atlas to t1 patient image ...'
    start = time.time()
    reg.niftireg_affine_registration(t1_atlas_path,
                                     t1_patient_path,
                                     transform_path=aff_trans,
                                     result_path=aff_reg)
    print 'Done (' + str((time.time() - start) / 60.) + ' min)'

    # non-rigid registration of t1 atlas to this image using the former
    # affine transformation (13-15 min)
    print 'Non-rigid registration from t1 atlas to t1 patient image ...'
    start = time.time()
    reg.niftireg_nonrigid_registration(t1_atlas_path,
                                       t1_patient_path,
                                       transform_path=aff_trans,
                                       cpp_path=f3d_cpp,
                                       result_path=f3d_reg)
    print 'Done (' + str((time.time() - start) / 60.) + ' min)'

    # resampling tissue atlases to this image using the former cpp grid
    print 'Resampling tissue atlas using non-rigid ccp to t1 patient image ...'
    start = time.time()
    for tissue in paths.TISSUES:
        reg.niftireg_transform(atlas_tissues[tissue],
                               t1_patient_path,
                               f3d_cpp,
                               result_path=reg_tissues[tissue],
                               cpp=True)
    print 'Done (' + str((time.time() - start)) + ' s)'
    
    return reg_tissues
