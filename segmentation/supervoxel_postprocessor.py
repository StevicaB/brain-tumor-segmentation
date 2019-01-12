'''
Created on Aug 3, 2017

@author: alberts
'''

import time
import os
import numpy as np

from scipy.stats import norm as gnorm

from supervoxels import supervoxel_operations as supop
import supervoxel_initializer as slic_init
from utilities import own_itk as oitk

FINAL_LABELS = ['enhanced', 'tumor_core', 'whole_tumor']

########################################################################

class SlicNeighbourLabelClassifier(slic_init.SlicMeansClassifier):
    """ Class that allows to classify supervoxels based on a training set
    of modality images and labelled images.
    
    In the example, supervoxels are classified in healthy, enhanced,
    tumor_core and whole_tumor based on T1, T1c, T2 and FLAIR images
    and segmentations.
    """
    
    def __init__(self, save_path, clf=None):
        """ Constructor.
            
        """        

        slic_init.SlicMeansClassifier.__init__(self, save_path, clf)
        
    ### Getting features and labels for all supervoxels in an image        
        
    def _get_X_y(self, 
                 soft_maps, 
                 mod_maps, 
                 supervoxels, 
                 mask, 
                 gt,
                 return_supervoxel_inst=False):
        """ Get relevant data to substract features. 
        
        Parameters
        ----------
        soft_maps : list of image like
            list of probabilistic input segmentations for each label
            (1st image for label 0, 2nd image for label 1, ...)
        mod_maps : list of image like
            modality images, can be supplied as paths, itk images
            or np.ndarrays
        supervoxels : list of image like
            supervoxel imag, in which each integer indicates a supervoxel
        mask : list of image like
            binary brain mask
        gt : image like or None
            ground truth segmentation with all labels (as a hard integer mask)
        return_supervoxel_inst : bool
            return the supervoxel instance 
            (i.e. to regerenate an image with predicted labels)
            
        """        
        
        # Create supervoxel instance 
        # make sure the supervoxels are separated under the current segmentation
        inst = supop.Supervoxels(supervoxels, mask)
            
        # Get features
        features_X = self._get_X(inst,
                                 soft_maps,
                                 mod_maps)
        
        # get labels
        labels_y = None
        if gt is not None:
            labels_y = inst.get_slic_most_common(gt)
            
        if return_supervoxel_inst:
            return features_X, labels_y, inst            
        return features_X, labels_y
        
    def _get_X(self,
               inst, 
               soft_maps,
               mod_maps):
        """ Get features using the supervoxel instance. 
        
        Called by _get_X_y()."""
        
        def quick_subfeature_check(sub_features):
            assert sub_features.shape[0] == nb_supervoxels
            assert np.isfinite(sub_features).all()
        
        soft_maps = list(soft_maps)
        soft_maps = slic_init.oitk.load_arr_from_paths(soft_maps)
        
        # get label of the prior segmentations for this supervoxel
        slic_labels = inst.get_slic_means(soft_maps)
        nb_supervoxels = slic_labels.shape[0]
        nb_labels = len(self.labels)
        assert slic_labels.shape[1] == nb_labels
        quick_subfeature_check(slic_labels)
        
        # add intensity difference to mean label intensity in each modality for this supervoxel
        slic_i_diff, slic_i_means = \
            self.get_slic_intensity_differences(inst,
                                                soft_maps,
                                                mod_maps,
                                                return_slic_means=True)
        assert slic_i_diff.shape[1] == nb_labels*self.nb_modalities
        assert slic_i_means.shape[1] == self.nb_modalities
        quick_subfeature_check(slic_i_diff)
        quick_subfeature_check(slic_i_means)
        
        # add percentage of neighbours for each label for this supervoxel
        slic_neighbours = inst.get_slic_neighbours(soft_maps)
        assert slic_neighbours.shape[1] == nb_labels 
        quick_subfeature_check(slic_neighbours)     
        
        # add distances of supervoxel to label com of this scan
        slic_distances = inst.get_slic_label_distance(soft_maps[1:],
                                                      cumulative=True)
        slic_distances = fill_nan_columns_with_median(slic_distances, 
                                                      soft_maps[0].shape)
        assert slic_distances.shape[1] == nb_labels-1
        quick_subfeature_check(slic_distances)
        
        # add distances of supervoxel to label com of this scan
        slic_distances_sep = inst.get_slic_label_distance(soft_maps[1:],
                                                          cumulative=False)
        slic_distances_sep = fill_nan_columns_with_median(slic_distances_sep, 
                                                          soft_maps[0].shape)
        assert slic_distances_sep.shape[1] == nb_labels-1
        quick_subfeature_check(slic_distances_sep)
            
        X = np.c_[slic_labels, 
                  slic_i_diff, 
                  slic_i_means,
                  slic_neighbours, 
                  slic_distances,
                  slic_distances_sep]
        
        return X
    
    def get_slic_intensity_differences(self,
                                       supervoxel_inst,
                                       soft_maps,#hard_map,
                                       mod_maps,
                                       return_slic_means=False):
        """ For each soft_map in soft_maps, get the mean intenisty in the mod_maps.
        Then calculate the mean intensity in each of the mod_maps for each of the
        supervoxel and calculate the difference in supervoxel intensities and label
        intensities. """
        
        # normalize images and get intensity means in every supervoxel
        slic_means, norm_mod_maps = \
                slic_init.SlicMeansClassifier._get_X(self, 
                                                     supervoxel_inst, 
                                                     mod_maps,
                                                     return_norm_mod_maps=True)
        
        # for each label get the mean in the modalities and calculate the difference
        # difference towards the supervoxel intensity means
        slic_i_diff = []
        for label in self.labels:
            weights = soft_maps[int(label)]
            if np.all(weights<=np.finfo(np.float).eps):
                print '%d th label not present' % label
            for i in range(len(norm_mod_maps)):
                try:
                    values = norm_mod_maps[i]
                    mean = np.average(values, weights=weights)
                    variance = np.average((values-mean)**2, weights=weights)
                except ZeroDivisionError:
                    if i==0:
                        print '%d th label not present' % label
                    gauss_eval = np.zeros_like(slic_means[:,i], dtype=np.float)
                    gauss_eval[:] = np.nan
                    assert np.all(np.isnan(gauss_eval)) # just a little numpy check
                else:
                    gauss_eval = gnorm.pdf(slic_means[:,i], 
                                           loc=mean, 
                                           scale=np.sqrt(variance))
                slic_i_diff.append(gauss_eval)
        slic_i_diff = np.asarray(slic_i_diff).T
        
        # fill in intenisty difference towards missing labels (median values)
        median_difference = np.median(slic_i_diff[~np.isnan(slic_i_diff)])
        slic_i_diff[np.isnan(slic_i_diff)] = median_difference

        if return_slic_means:
            return slic_i_diff, slic_means
        return slic_i_diff
    
    ### Save and load the classifier and its training data
    
    def load(self):
        """ Load the classifier written at self.path. 
        
        Throws an error if self.path doesnt exist or if the
        classifier at this path isnt valid for this instance. """
               

        slic_init.SlicMeansClassifier.load(self)
        
        # adapt self.nb_modalities (isnt equal to self.clf.n_features)
        nr_labels = len(self.labels)
        enumerater = (self.clf.n_features_ - (4*nr_labels) + 2)
        self.nb_modalities = enumerater / (1 + nr_labels)
        if enumerater % nr_labels != 0:
            err = 'Hm, this should be a clean division: %d/%d' % (\
                                enumerater, nr_labels)
            raise AttributeError(err)
            
def fill_nan_columns_with_median(arr, original_shape):
    """ Within this array, fill nan columns with the median value of their
    row. In this array, nans should only occur in full columns. """
    
    # if the whole array is NaN, set an average distance
    if np.all(np.isnan(arr)):
        print 'No non-zero labels present!'
        
        original_shape = [dim/2. for dim in original_shape]
        x_dist, y_dist, z_dist = original_shape
        far_distance = np.sqrt((x_dist**2) + (y_dist**2) + (z_dist**2))
        
        arr[:] = far_distance
        return arr
    
    # remove NaN columns present due to missing labels
    if np.isnan(arr).any():
        print 'Again, there is a label not present'
        # check nans are only occurring in entire nan columns
        assert False not in [np.all(np.isnan(sub_arr)) or np.all(~np.isnan(sub_arr)) \
                             for sub_arr in arr.T]
        
        # get median label distance for each supervoxel
        median_distances = np.nanmedian(arr, axis=1)
        print median_distances.shape
        print 'Do medians contain nans? %s' % np.any(np.isnan(median_distances))
        # fill in nan columns
        for sub_arr in arr.T:
            print sub_arr.shape
            if np.all(np.isnan(sub_arr)):
                sub_arr[:] = median_distances
            elif np.isnan(sub_arr).any():
                print 'BAM BAM BAM something is wrong'
            print 'Does subarr contain nans: %s' % np.any(np.isnan(sub_arr))
            print 'Does arr contain nans? %s' % np.any(np.isnan(arr))
                                
        print 'Does arr contain nans? %s' % np.any(np.isnan(arr))
        print 'Does arr contain nans? %s' % (np.isnan(arr)).any()
        assert not np.any(np.isnan(arr))

    return arr

def get_segmentation(mod_maps, init_maps, mask, supervoxels=None):
    """ Get a rough estimation of probabilistic tumor segmentation maps using 
    a supervoxel classifier. 
    
    Parameters
    ----------
    mod_maps : dict (str : imagelike)
        dictionary with keys  't1', 't2', 't1c', 'flair' mapping to the 
        corresponding image paths (or np.ndarrays)
    init_maps : dict (str : imagelike)
        dictionary with keys slic_init.SLIC_LABELS mapping to the 
        corresponding image paths (or np.ndarrays)
    mask : imagelike
        the brain mask of this case
    supervoxels : imagelike, optional
        the supervoxels calculated on the modality images, defaults to
        None, in which case they will be calculated
        
    Returns
    -------
    soft_maps : dict (str : imagelike)
        dictionary with keys 'enhanced', 'tumor_core' and 'whole_tumor',  
        mapping to the corresponding probabilistic segmentations
    hard_map : np.ndarray
        the integer mask of the predicted segmentation
    supervoxels : imagelike
        the supervoxels used for supervoxel classification
        
    """
    
    modality_order = ['t1', 't2', 't1c', 'flair']
    
    # check input
    err = ''
    if not isinstance(mod_maps, dict):
        err = 'Please provide a dictionary with the modality identifiers as keys,'
        err += 'such that I know which path belongs to which modality!' 
    elif set(mod_maps.keys()) != set(modality_order):
        err = "Please provide 't1', 't2', 't1c' AND 'flair' paths in mod_maps!"
    if len(err) > 0:
        raise ValueError(err)
    
    # load instance to classify (which has an intern RF classifier)
    this_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(this_dir, 'pickles', 'supervoxel_classifier_post.pickle')
    clf = SlicNeighbourLabelClassifier(save_path)
    
    # create input data
    mod_list = [mod_maps[mod] for mod in modality_order]
    if supervoxels is None:
        supervoxels = SlicNeighbourLabelClassifier.get_supervoxels(mod_list, mask)
    
    soft_segmentations, hard_map = clf.predict_proba(mod_list,
                                                     supervoxels,
                                                     mask,
                                                     None, 
                                                     init_maps)
    
    soft_maps = {} 
    for i, label in enumerate(FINAL_LABELS):
        soft_maps[label] = soft_segmentations[i+1] #+1 because bg at 0
    
    return soft_maps, hard_map, supervoxels
    
    
if __name__ == '__main__':
    
    # assemble the paths for your patient
    this_dir = os.path.dirname(os.path.realpath(__file__))
    mod_maps = {'t1':os.path.join(this_dir, 'example', 't1.nii.gz'),
                't1c':os.path.join(this_dir, 'example', 't1c.nii.gz'),
                't2':os.path.join(this_dir, 'example', 't2.nii.gz'),
                'flair':os.path.join(this_dir, 'example', 'flair.nii.gz')}
    mask = os.path.join(this_dir, 'example', 'brain_mask.nii.gz')
    
    # load supervoxels if they exist
    supervoxel_path = os.path.join(this_dir, 'example', 'init_supervoxels.nii.gz')
    loaded_supervoxels = None
    if os.path.exists(supervoxel_path):
        loaded_supervoxels = oitk.get_itk_array(supervoxel_path)
        
    # load em segmentation maps
    em_maps = {}
    em_path = os.path.join(this_dir, 'example', 'ems_results', 'to_replace.nii.gz')
    for label in slic_init.SLIC_LABELS:
        em_maps[label] = oitk.get_itk_array(em_path.replace('to_replace',label))
    em_maps_list = []
    em_maps_list.append(em_maps['gm']+em_maps['wm']+em_maps['csf'])
    em_maps_list.append(em_maps['enhanced'])
    em_maps_list.append(em_maps['nonactive'] + em_maps['necrotic'])
    em_maps_list.append(em_maps['edema'])
        
    soft_maps, hard_map, supervoxels = get_segmentation(mod_maps, 
                                              em_maps_list, 
                                              mask, 
                                              loaded_supervoxels)
    
    # write results
    proto = oitk.get_itk_image(mod_maps['t1'])
    if loaded_supervoxels is None:
        im = oitk.make_itk_image(supervoxels, proto, verbose=False)
        oitk.write_itk_image(im, supervoxel_path)
    dirname = os.path.join(this_dir, 'example', 'post_results')
    if not os.path.exists(dirname):
        os.mkdir(dirname) 
    for label, prob_map in soft_maps.iteritems():
        path = os.path.join(dirname, label+'.nii.gz')
        im = oitk.make_itk_image(soft_maps[label], proto, verbose=False)
        oitk.write_itk_image(im, path)
    path = os.path.join(dirname, 'hard_map.nii.gz')
    im = oitk.make_itk_image(hard_map, proto, verbose=False)
    oitk.write_itk_image(im, path)
    
