'''
Created on Oct 25, 2017

@author: alberts
'''
import os
import numpy as np

from em import ems_initializer as ei
from em import ems_read_input as er
from em import ems_write_output as ew
from em import ems_tumor_original as eto

from utilities import own_itk as oitk
 
modality_order = ['t1', 't2', 't1c', 'flair']
tumors = ['enhanced', 'nonactive', 'necrotic', 'edema']
tissues = ['gm','wm','csf']
em_labels = tissues + tumors

def get_segmentation(mod_maps, tissue_maps, init_maps, mask, save_dir=None):
    """ Get an EM estimation of the tumor segmentation using some kind of
    initialized tumor maps. 
    
    Parameters
    ----------
    mod_maps : dict (str : imagelike)
        dictionary with keys  't1', 't2', 't1c', 'flair' mapping to the 
        corresponding image paths (or np.ndarrays)
    tissue_maps : dict (str : imagelike)
        dictionary with keys  'wm', 'gm' AND 'csf' mapping to the corresponding 
        registered tissue paths (or np.ndarrays)
    init_maps : dict (str : imagelike)
        dictionary with at least keys  'enhanced', 'nonactive', 'necrotic', 'edema' 
        mapping to the corresponding tumor segmentation paths (or np.ndarrays)        
    mask : imagelike
        the brain mask of this case
    save_dir : str
        path to the directory where to store results
        
    Returns
    -------
    soft_maps : dict (str : imagelike)
        dictionary with keys 'gm', 'wm', 'csf''enhanced', 'nonactive', 'necrotic'
        AND 'edema',  mapping to the corresponding probabilistic segmentations
        
    """

    if save_dir is not None and ew.all_result_paths_exist(save_dir, em_labels):
        print 'Result paths already exist in %s' % save_dir
        print 'Returning previous calculation...'
        print ''
        return

    # join tissue and tumor maps
    _check_input(mod_maps, tissue_maps, init_maps)
    tissue_maps = join_tumor_and_tissues(tissue_maps, init_maps)
    if set(tissue_maps.keys()) != set(em_labels):
        err = 'I needed %s keys, but I got %s' % \
                    (str(em_labels),str(tissue_maps.keys()))
        raise RuntimeError(err)
    
    # set parameters and paths to images for this run
    label_functions = {key : 'multivariate' for key in em_labels}
    param = ei.Param(max_iteration=15,
                     sub_sample=1)
    reader = er.ReaderEM(mod_maps,
                         tissue_maps,
                         mask_input=mask,
                         param_instance=param,
                         distr_for_label=label_functions)
    writer = ew.WriterEM(save_dir=save_dir)

    # read in the images and set paths to store results
    init_em = ei.Initializer(reader, writer)

    # initialize all variables
    em = eto.IteraterEM(init_em)

    # avoid divergence by enforcing to stay close to the initial maps
    em.set_stable_prior()

    # run the algorithm
    soft_maps = em.run()
        
    return soft_maps

def _check_input(mod_maps, tissue_maps, init_maps):
    
    # check input
    err = ''
    if not isinstance(mod_maps, dict):
        err = 'Please provide a dictionary with the modality identifiers as keys,'
        err += 'such that I know which path belongs to which modality!' 
    elif set(mod_maps.keys()) != set(modality_order):
        err = "Please provide all %s paths in mod_maps!" % str(modality_order)
    elif not isinstance(tissue_maps, dict):
        err = 'Please provide a dictionary with the tissue identifiers as keys,'
        err += 'such that I know which path belongs to which tissue map!' 
    elif set(tissue_maps.keys()) != set(tissues):
        err = "Please provide all %s paths in tissue_paths!" % str(tissues)
    elif not isinstance(init_maps, dict):
        err = 'Please provide a dictionary with the modality identifiers as keys,'
        err += 'such that I know which path belongs to which modality!' 
    elif not set(init_maps.keys()) > set(tumors):
        err = "Please provide all %s paths in init_maps!" % str(tumors)
    if len(err) > 0:
        raise ValueError(err)  

def join_tumor_and_tissues(tissue_maps, init_maps):
    """ Add the tumor estimations in init_maps (loose estimations of the tissues 
    and tumors) to tissue_maps (= registered atlas) and rescale to sum up to 1. 
    
    Tissue maps in the registered atlas are usually nicer than the tissue maps as 
    predicted in init_maps.
    
    """
 
    # Sum tumor maps in init_maps and set as foreground
    fg_maps = [oitk.get_itk_array(init_maps[key]) for key in tumors]
    foreground = np.sum(np.asarray(fg_maps), axis=0)
    
    # Get background
    foreground = np.fmax(foreground, 0)
    foreground = np.fmin(foreground, 1)
    background = (1. - foreground)
    
    # Rescale healthy tissues with background coefficients
    for key in tissues:
        tissue_maps[key] = oitk.get_itk_array(tissue_maps[key])*background
 
    # Add tumors to tissue_maps
    for key in tumors:
        tissue_maps[key] = oitk.get_itk_array(init_maps[key])
 
    return tissue_maps 

if __name__ == '__main__':
    
    # assemble the paths for your patient
    this_dir = os.path.dirname(os.path.realpath(__file__))
    mod_maps = {'t1':os.path.join(this_dir, 'example', 't1.nii.gz'),
                't1c':os.path.join(this_dir, 'example', 't1c.nii.gz'),
                't2':os.path.join(this_dir, 'example', 't2.nii.gz'),
                'flair':os.path.join(this_dir, 'example', 'flair.nii.gz')}
    tissue_maps = {'gm':os.path.join(this_dir, 'example', 'gm_atlas_reg.nii.gz'),
                'wm':os.path.join(this_dir, 'example', 'wm_atlas_reg.nii.gz'),
                'csf':os.path.join(this_dir, 'example', 'csf_atlas_reg.nii.gz')}
    init_maps = {}
    for label in em_labels:
        init_maps[label] = os.path.join(this_dir, 'example', 'init_results', label+'.nii.gz')
    mask = os.path.join(this_dir, 'example', 'brain_mask.nii.gz')
    save_dir = os.path.join(this_dir, 'example', 'em_results')
    
    # segment and write results
    get_segmentation(mod_maps, tissue_maps, init_maps, mask, save_dir=None)
    
