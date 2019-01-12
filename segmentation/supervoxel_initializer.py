'''
Created on Oct 24, 2017

@author: alberts
'''
import os
import numpy as np
import cPickle
import time

from .supervoxels import supervoxel_operations as supop
from .supervoxels import supervoxels as sup

from .utilities import intensity_normalisation as inorm
from .utilities import own_itk as oitk

SLIC_LABELS = ['gm', 'wm', 'csf'] + ['necrotic','edema','nonactive','enhanced']

class SlicMeansClassifier():
    """ Class that allows to classify supervoxels based on a training set
    of modality images and labelled images.
    
    In the example, supervoxels are classified in WM, GM, CSF, enhanced,
    nonactive, necrotic and edema based on T1, T1c, T2 and FLAIR images.
    """

    def __init__(self, save_path, clf=None):
        """ Constructor.
        
        Throws an error if clf is None and there is no classifier present at 
        save_path.
        
        Parameters
        ----------
        clf : sklearn classifier with attributes: classes_ and n_features_ and 
                functions: fit(), predict(), predict_proba() and score()
            the classifier trained to predict the labels of the supervoxels
            if set to None, one will be loaded at save_path
        save_path : str
            the path where self.clf is stored are is to be saved.
            
        """

        if clf is None and not os.path.exists(save_path):
            err = 'Please supply a classifier, %s doesnt exist' % save_path
            raise ValueError(err)
        if save_path[-7:] != '.pickle':
            err = '%s not a pickle path!' % save_path
            err += 'Please supply a pickle path to save the classifier'
            raise ValueError(err)
        if not os.path.exists(os.path.dirname(save_path)):
            err = 'Directory doesnt exist: %s' % save_path
            raise ValueError(err)   
        self.path = save_path
        if clf is None:
            self.load()
        else:  
            self._check_valid_clf(clf)
            self.clf = clf
            
    def _check_valid_clf(self, clf):
        """ Raise an error if this isnt a valid classifier. """
        
        if (not hasattr(clf, 'fit')) or \
             (not hasattr(clf, 'predict')) or \
              (not hasattr(clf, 'predict_proba')) or \
               (not hasattr(clf, 'score')):
            err = 'I somehow dont understand this as a classifier: %s' % clf
            raise ValueError(err)
        
    def _is_trained(self):
        """ Return if the classifier has been trained or not. """
        
        if hasattr(self.clf, 'classes_') and hasattr(self.clf, 'n_features_'):
            return True
        
        return False
        
    ### Training the classifier
        
    def fit(self,
            all_modalities,
            all_supervoxels,
            all_masks,
            all_truths,
            *args):
        """ Fit the classifier. Arguments are forwarded to self._get_X_y.
        
        Parameters
        ----------
        all_modalities : list of lists of image like
            modality images for all patients, can be supplied as paths, 
            itk images or np.ndarrays. The order of the modality images
            should be consistent across the patients.
        all_supervoxels : list of image like
            supervoxel image for all patients, in which each integer indicates
            a supervoxel
        all_masks : list of image like
            binary brain masks for all patients
        all_truths : list of image like
            ground truth segmentations of all patients, as hard integer mask
        
        """
        
        # Set modalities and labels
        self._set_variables(all_modalities, all_masks, all_truths)
        
        # Get features and labels
        train_X = None
        print 'Building feature matrix',
        args = list(args)
        args.extend([all_modalities, 
                     all_supervoxels,
                     all_masks, 
                     all_truths])
        all_data = map(list, zip(*args))
        for _ in range(len(all_data)):

            slic_features, true_slic_labels = self._get_X_y(*all_data.pop(0))
            
            sample_nb = len(true_slic_labels)
            print '+ %d' % sample_nb,                        
            if len(slic_features) != sample_nb:
                print '... should be equal to %d' % len(sample_nb)    
                err = 'Feature matrix and labels dont have'+\
                            ' the same number of samples!'
                raise RuntimeError(err)

            if train_X is None:
                train_X = slic_features
                train_y = true_slic_labels
            else:
                if slic_features.shape[1] != train_X.shape[1]:
                    err = 'Expected %d features, got %d' % (
                            train_X.shape[1],
                            slic_features.shape[1])
                    raise RuntimeError(err)
                train_X = np.concatenate((train_X, slic_features), axis=0)
                train_y = np.concatenate((train_y, true_slic_labels), axis=0)
                

        # Train the classifier
        print 'Training shape: '+str(train_X.shape)
        print 'Testing shape: '+str(train_y.shape)
        print 'Saving training features and labels ...'
        self.save_training(train_X,train_y)
        
#         sample_weight = np.ones_like(train_y)
#         sample_weight[train_y>0] = 2
#         sample_weight[train_y==np.max(train_y)] = 3
        self.clf.fit(train_X, train_y)
        assert set(self.clf.classes_) == set(self.labels)

        return train_X, train_y
    
    def _set_variables(self, all_modalities, all_masks, all_truths):
        """ Set expected number of modalities and labels for this classifier 
        instance. """
        
        # Checking and setting nb of modalities
        self.nb_modalities = len(all_modalities[0])
        len_images = [len(mod_maps) for mod_maps in all_modalities]
        if True in [l != self.nb_modalities for l in len_images]:
            err = 'Expected %d images, got %s' % (self.nb_modalities, 
                                                  [l for l in len_images if l!=self.nb_modalities])
            raise RuntimeError(err)  
        
        # Setting labels
        labels = [label for i in range(len(all_truths))
                  for label in np.unique(all_truths[i][all_masks[i]>0])]
        self.labels = np.unique(np.asarray(labels))
        
    def _check_modalities(self, modalities):
        
        # Checking modalities
        if len(modalities) != self.nb_modalities:
            err = 'Expected %d images, got %s' % (self.nb_modalities, 
                                                  len(modalities))
            raise RuntimeError(err)     

    ### Predicting with the classifier

    def predict_proba(self,
                      mod_maps,
                      supervoxels,
                      mask,
                      gt,
                      *args):
        """ Predict slic segemtnation for this image. Set gt to None if
        gt is not available. 
        
        Parameters
        ----------
        mod_maps : list of image like
            modality images, can be supplied as paths, itk images or 
            np.ndarrays
            Carefull, the order should be the same as how they were given
            to the fit() method!!!
        supervoxels : list of image like
            supervoxel imag, in which each integer indicates a supervoxel
        mask : list of image like
            binary brain mask
        gt : image like or None
            ground truth segmentation as a hard integer mask
            
        Returns
        -------
        pred_im_proba : list of np.ndarrays
            probabilistic segmentations (in the original image dimensions)
            for each label.
        pred_im : np.ndarray
            hard integer mask (in the original image dimensions) of the predicted 
            segmentation, where each integer indicates a label.
        
        """
        
        if not self._is_trained():
            err = 'Classifier hasnt been trained yet! Call fit first...'
            raise RuntimeError(err)

        print '\t Classification of slic regions: ',
        start = time.time()

        args = list(args)
        args.extend([mod_maps, supervoxels, mask, gt])
        test_X, test_y, inst = self._get_X_y(*args, 
                                            return_supervoxel_inst=True)

        # Get predictions
        pred_y_proba = self.clf.predict_proba(test_X)
        print '\t ' + str(time.time() - start) + 's'
            
        # Get accuracy score
        if test_y is not None:
            score = self.clf.score(test_X, test_y)
            print '\t Accuracy: ' + str(score)

        # Get image segmentations
        print '\t Creating label images: ',
        start = time.time()
        
        # Get the soft full label maps (fill in predictions on the slic regions)
        dim = supervoxels.shape
        nr_labels = pred_y_proba.shape[1]
        pred_im_proba = np.zeros(shape=(nr_labels, dim[0], dim[1], dim[2]))
        for label_ind in range(nr_labels):
            discrete_probs = np.around(pred_y_proba[:, label_ind], 1)
            this_proba_im = inst.map_supervoxel_integers(discrete_probs)
            pred_im_proba[label_ind] = this_proba_im
                        
        # Get the full hard label map
        pred_y = self.clf.predict(test_X)
        pred_im = inst.map_supervoxel_integers(pred_y)
        print '\t ' + str(time.time() - start) + 's'

        return pred_im_proba, pred_im
    
    ### Getting features and labels for all supervoxels in an image
        
    def _get_X_y(self, 
                mod_maps, 
                supervoxels, 
                mask, 
                gt,
                return_supervoxel_inst=False):
        """ Get relevant data to substract features. 
        
        Parameters
        ----------
        mod_maps : list of image like
            modality images, can be supplied as paths, itk images
            or np.ndarrays
        supervoxels : list of image like
            supervoxel imag, in which each integer indicates a supervoxel
        mask : list of image like
            binary brain mask
        gt : image like or None
            ground truth segmentation as a hard integer mask
        return_supervoxel_inst : bool
            return the supervoxel instance 
            (i.e. to regerenate an image with predicted labels)
            
        """        
        
        # Create supervoxel instance 
        inst = supop.Supervoxels(supervoxels, mask)
        
        # Get features
        features_X = self._get_X(inst, mod_maps)
       
        # get labels
        labels_y = None
        if gt is not None:
            labels_y = inst.get_slic_most_common(gt)
            
        if return_supervoxel_inst:
            return features_X, labels_y, inst
        return features_X, labels_y

    def _get_X(self, inst, mod_maps, return_norm_mod_maps=False):
        """ Get features for each supervoxel, based on the modality images and the
        supervoxel instance. 
        
        Called by _get_X_y()."""
        
        # Read modality maps
        mod_maps = list(mod_maps)
        mod_maps = oitk.load_arr_from_paths(mod_maps)

        # Get mean of each supervoxel in the normalized mod_maps
        norm_mod_maps = [inorm.whiten(mod_map, inst.mask) \
                         for mod_map in mod_maps] 
        features = inst.get_slic_means(norm_mod_maps)
        
        if return_norm_mod_maps:
            return features, norm_mod_maps
        return features
    
    ### Save and load the classifier and its training data

    def save(self):
        """ Save the classifier at self.clf to self.path. """

        with open(self.path, 'w') as f:
            cPickle.dump(self.clf, f)

    def save_training(self, X, y):
        """ Save the training data with which self.clf.fit has been called. """
        
        path = os.path.join(os.path.dirname(self.path), 
                           'X_train_'+os.path.basename(self.path))
        with open(path, 'w') as f:
            cPickle.dump(X, f)        
        with open(path.replace('X_train','y_train'), 'w') as f:
            cPickle.dump(y, f)
    
    def load(self):
        """ Load the classifier written at self.path. 
        
        Throws an error if self.path doesnt exist or if the
        classifier at this path isnt valid for this instance. """
        
        if not os.path.exists(self.path):

            err = 'Path doesnt exists!'
            raise ValueError(err)

        with open(self.path) as f:
            clf = cPickle.load(f)
        self._check_valid_clf(clf)
        
        self.clf = clf
        self.labels = self.clf.classes_   
        self.nb_modalities = self.clf.n_features_ 
        
    @staticmethod
    def get_supervoxels(mod_maps, mask):
        """ Calculate supervoxels for this list of modality images.
        
        """
        
        mask = oitk.get_itk_array(mask)
        mod_maps = [oitk.get_itk_array(arr) for arr in mod_maps]
        
        for i, im in enumerate(mod_maps):
            im[mask==0] = 0
            mod_maps[i] = inorm.window_rescale(im, mask)
        
        print 'Calculating supervoxels'
        supervoxels = sup.get_supervoxels(mod_maps) 
        
        return supervoxels
    
def get_segmentation(mod_maps, mask, supervoxels=None):
    """ Get a rough estimation of probabilistic tumor segmentation maps using 
    a supervoxel classifier. 
    
    Parameters
    ----------
    mod_maps : dict (str : imagelike)
        dictionary with keys  't1', 't2', 't1c', 'flair' mapping to the 
        corresponding image paths (or np.ndarrays)
    mask : imagelike
        the brain mask of this case
    supervoxels : imagelike, optional
        the supervoxels calculated on the modality images, defaults to
        None, in which case they will be calculated
        
    Returns
    -------
    soft_maps : dict (str : imagelike)
        dictionary with keys 'gm', 'wm', 'csf''enhanced', 'nonactive', 'necrotic'
        AND 'edema',  mapping to the corresponding probabilistic segmentations
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
    save_path = os.path.join(this_dir, 'pickles', 'supervoxel_classifier_init.pickle')
    clf = SlicMeansClassifier(save_path)

    # create input data
    mod_list = [mod_maps[mod] for mod in modality_order]
    if supervoxels is None:
        supervoxels = SlicMeansClassifier.get_supervoxels(mod_list, mask)

    soft_segmentations, _ = clf.predict_proba(mod_list,
                                              supervoxels,
                                              mask,
                                              None)

    soft_maps = {} 
    for i, label in enumerate(SLIC_LABELS):
        soft_maps[label] = soft_segmentations[i]
    
    return soft_maps, supervoxels
    
    
if __name__ == '__main__':
    
    # assemble the paths for your patient
    this_dir = os.path.dirname(os.path.realpath(__file__))
    mod_maps = {'t1':os.path.join(this_dir, 'example', 't1.nii.gz'),
                't1c':os.path.join(this_dir, 'example', 't1c.nii.gz'),
                't2':os.path.join(this_dir, 'example', 't2.nii.gz'),
                'flair':os.path.join(this_dir, 'example', 'flair.nii.gz')}
    mask = os.path.join(this_dir, 'example', 'brain_mask.nii.gz')
    
    # load supervoxels if they exist, and segment
    supervoxel_path = os.path.join(this_dir, 'example', 'init_supervoxels.nii.gz')
    loaded_supervoxels = None
    if os.path.exists(supervoxel_path):
        loaded_supervoxels = oitk.get_itk_array(supervoxel_path)
    soft_maps, supervoxels = get_segmentation(mod_maps, mask, loaded_supervoxels)
    
    # write results
    proto = oitk.get_itk_image(mod_maps['t1'])
    if loaded_supervoxels is None:
        im = oitk.make_itk_image(supervoxels, proto, verbose=False)
        oitk.write_itk_image(im, supervoxel_path)
    for label, prob_map in soft_maps.iteritems():
        dirname = os.path.join(this_dir, 'example', 'init_results')
        if not os.path.exists(dirname):
            os.mkdir(dirname) 
        path = os.path.join(dirname, label+'.nii.gz')
        im = oitk.make_itk_image(soft_maps[label], proto, verbose=False)
        oitk.write_itk_image(im, path)
    
