# -*- coding: UTF-8 -*-
"""Module allowing to write data and metadata resulting from the em
algorithm.

"""
__version__ = '1.0'
__author__ = 'Esther Alberts'

import numpy as np
import os

from ..utilities import own_itk as oitk

class WriterEM():
    """Class to deal with visualisation adn (over)writing data and
    results from the algorithm.

    An instance must be created to create an ems_initializer.Initializer()
    instance, which is needed to initialize the
    ems_tumor_original.IteraterEM(), which allows to run the em.

    Attributes
    ----------
    overwrite : boolean
        If True, results are to be overwritten if they already exist,
        if False, abort is set to True.
    abort : boolean
        If True, the em should not be allowed to run.
    save_dir : str
        Path to directory were tissue and tumor segmentations are to be
        written.
    tissue_paths : list of str
        Paths where tissue segmentations will be saved.
    tumor_paths : list of str
        Paths where tumor segmentations will be saved.
    metadata_path : str
        Path were metadata is to be written.
    metadata : dict
        Dictionary containing metadata information.
    r : ems_read_input.ReaderEM
        Reader containing information about input data and run parameters.

    """

    def __init__(self,
                 save_dir=None,
                 overwrite=False,
                 write_interval=None):

        # Parameters concerning result path names
        self.overwrite = overwrite
        self.save_dir = save_dir

        # Parameters concerning saving nii of plot results
        self.write_interval = write_interval

    def connect_to_read(self, read_instance):
        """Integrate information from reading instance

        parameters
        ----------
        read_instance : ems_read_input.ReadInput
            the read_instance allows to have access to the input data

        """

        self.r = read_instance

        # Further refine WriterEM instance
        self._set_directories()
        self._set_result_paths()
        self._set_metadata()
        self._create_metadata_file()

        if not self.abort:

            self._create_visualiser()

        print ''

    ####################################################################

    def _set_directories(self):
        """Set the directory where results will be saved.

        1. Get the subdir whose name is based on the availability of a tumor
        box and the use of inclusion constraints.
        2. Then create a directory to store the result for this run.
        3. In this directory, create a directory to save the figures.

        """

        sub_dir = self._get_sub_dir()

        self._set_abort(sub_dir)

        self._set_dir(sub_dir)

    def _get_sub_dir(self):
        """ Get the sub directory (middle directory), in which a directory
        will be created to save the results of this case. """

        if self.save_dir is None:
            dirname = os.path.dirname(os.path.commonprefix(self.r.data_files))
            ref_dir = os.path.join(dirname, 'ems_results')
            if not os.path.exists(ref_dir):
                os.mkdir(ref_dir)
        else:
            ref_dir = self.save_dir
            if not os.path.exists(ref_dir):
                os.mkdir(ref_dir)

        return ref_dir

    def _set_abort(self, nii_dir):
        """See if results already exist in this directory, and if so, if we have 
        to recalculate or instead, abort.

        """

        def _fully_processed():
            """See if a run has already fully been calculated previously."""
            label_paths = self._get_result_paths(nii_dir)

            label_exist = [os.path.exists(f) for f in label_paths]
            return False not in label_exist

        if _fully_processed():
            print 'Run already fully processed'
            if self.overwrite:
                print 'However, you wish to recalculate anyways'
                print 'Recalculating now ...'
                self.abort = False
            else:
                print 'Ok, aborting ...'
                self.abort = True

        else:
            print 'Run has been started but calculations hadnt finished'
            print 'Recalculating now ...'
            self.abort = False

    def _set_dir(self, new_dir_path):
        """Set the directory name to save results and save the metadata
        there.

        parameters
        ----------
        sub_dir : str
            path where to create a new directory to save results for this run.

        """

        metadata_path = os.path.join(new_dir_path, 'metadata.dat')
        self.metadata_path = metadata_path

        self.nii_dir = new_dir_path
        print 'SAVE_DIR: ' + print_path(self.nii_dir, 3)

    ##########################################################################

    def _set_result_paths(self):
        """Set the paths where tissue and tumor segmentations should
        be written. Assume `self.nii_dir` has already been set.

        """

        label_paths = self._get_result_paths(self.nii_dir)
        self.label_paths = label_paths

    def _get_result_paths(self, nii_dir):
        return get_result_paths(nii_dir, self.r.labels)
    
    ##########################################################################
    
    def _set_metadata(self, intensity_model=True):
        """ Set the dictionary for metadata: `self.metadata`. """
        
        self.metadata = {}
        
        input_data = {}
        atlas_tmp = [f if not isinstance(f, np.ndarray)
                     else 'array given'
                     for f in self.r.atlas_input]
        input_data['atlas_input'] = atlas_tmp
        mask_tmp = 'array given' if isinstance(self.r.mask_input, np.ndarray) \
                    else self.r.mask_input
        input_data['mask_file'] = mask_tmp
        input_data['data_files'] = self.r.data_files
        input_data['modalities'] = self.r.modalities
        input_data['labels'] = self.r.labels
        self.metadata['input_data'] = input_data

        tumor_box = {}
        tumor_box['tumor_lower_co'] = self.r.tlc
        tumor_box['tumor_upper_co'] = self.r.tuc
        self.metadata['tumor_box'] = tumor_box

        param = {}
        param['max_iteration'] = self.r.p.max_iteration
        param['sub_sample'] = self.r.p.sub_sample
        param['seed'] = self.r.p.seed
        self.metadata['param'] = param

        if intensity_model:
            int_param = {}
            int_param['gaussian_norm'] = self.r.p.gaussian_norm
            self.metadata['intensity_param'] = int_param

    def _create_metadata_file(self):
        """Create parameter file based on metadata of instance.

        Parameter file is temporarily saved under Downloads. The metadata
        is set as an instance attribute `self.metadata`.

        """

        # open metadata
        fid = open(self.metadata_path, 'w')
        fid.write('// \n')
        
        # Write metadata
        fid.write('// ****** Input Data ****** \n')
        for key, value in self.metadata['input_data'].iteritems():
            fid.write('%s: %s \n' % (key, value))
            
        fid.write('// ****** Tumor Box ****** \n')
        for key, value in self.metadata['tumor_box'].iteritems():
            fid.write('%s: %s \n' % (key, value))

        fid.write('// ****** Parameters ****** \n')
        for key, value in self.metadata['param'].iteritems():
            fid.write('%s: %s \n' % (key, value))

        if 'intensity_param' in self.metadata:
            fid.write('// ****** Intensity parameters ****** \n')
            for key, value in self.metadata['param'].iteritems():
                fid.write('%s: %s \n' % (key, value))

        fid.close()    

    ####################################################################

    def _create_visualiser(self):
        """Create a visualisation instance for this data."""

        pass
            
    def set_show(self, show):
        
        pass

    ####################################################################

    def visualise_input(self):
        """Visualise input data."""

        pass

    ##########################################################################

    def save_iteration(self,
                       label_maps,
                       iteration):

        if self.write_interval is not None:
            if (iteration - 1) % self.write_interval == 0:
                self.write_label_maps(label_maps, iteration)

    def save_on_convergence(self,
                            label_maps,
                            log_llh):

        self.visualise_iteration(label_maps, 'final')
        self.write_label_maps(label_maps)
        self.plot_logLLH(log_llh)

    def visualise_iteration(self,
                            label_maps,
                            iteration):
        """Visualise results of current iteration."""

        pass

    def visualise_volumes(self, arrs, title):

        pass

    def plot_logLLH(self, logLLHArray):
        """Plot the log likelihoods up to the current iteration."""

        pass

    ##########################################################################

    def write_label_maps(self, label_maps):
        """Write the tissue segmentations."""

        for label_ind in range(self.r.nr_labels):

            label_map = self.r.to_matrix(label_maps[label_ind, :])
            image = oitk.make_itk_image(label_map,
                                        self.r.data_itk_image)

            path = self.label_paths[label_ind]
            oitk.write_itk_image(image, path)

            print self.r.labels[label_ind] + \
                    ' segmentation written at ' + path[50:]

    def write_average_label_maps(self, label_maps):
        """Write the tissue segmentations."""

        av_sum = np.sum(label_maps, axis=0)
        label_maps = label_maps / np.fmax(av_sum, np.finfo(float).tiny)
        self.visualise_iteration(label_maps , 'averaged')
        self.write_label_maps(label_maps, 'average')

    def write_intermediate(self, arr, basename):

        path = os.path.join(self.nii_dir, basename)
        full_arr = self.r.to_matrix(arr)
        im = oitk.make_itk_image(full_arr)
        oitk.write_itk_image(im, path)

        print basename + ' written in ./ems_results'

    def write_recognized_labels(self, unknown_to_known):
        """ Write a little document specifying how the labels map to known
        labels. """

        fid = open(os.path.join(self.nii_dir, 'recognized_labels.txt'),
                   'w')

        fid.write('// ****** Recognized labels ****** \n')

        for label in unknown_to_known:
            fid.write('%s : %s \n' % (label, unknown_to_known[label]))

        fid.write('// \n')
        fid.close()

##########################################################################

def get_result_paths(nii_dir, labels):
    label_paths = []
    for i in range(len(labels)):
        label = labels[i]
        path = os.path.join(nii_dir, label + '.nii.gz')
        label_paths.append(path)

    return label_paths

def all_result_paths_exist(nii_dir, labels):
    result_paths = get_result_paths(nii_dir, labels)
    if False in [os.path.exists(path) for path in result_paths]:
        return False
    return True

def print_path(path, layers=3):
    """ Print a number of last directories in the directory hierarchy
    of the path, specified by layers. """

    list_of_dirs = path.split('/')
    if list_of_dirs[-1] == '':
        list_of_dirs = list_of_dirs[:-1]

    layer = layers
    sub_path = ''
    while layer > 0:
        sub_path += '/' + list_of_dirs[-layer]
        layer -= 1

    return sub_path
