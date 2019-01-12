#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Module responsible for loading images contained in image files in numpy
arrays and setting variables needed to run the EM on

"""
__version__ = '1.0'
__author__ = 'Esther Alberts'

import numpy as np
import os
import scipy.ndimage as scim

import ems_initializer as ei

from ..utilities import own_itk as oitk

############################################################################################

class InputData():
    """Class to read input data and initialise variables based on
    run parameters

    An instance must be created to create an ems_initializer.Initializer()
    instance, which is needed to initialize the
    ems_tumor_original.IteraterEM(), which allows to run the EM.

    """

    def __init__(self,
                 data_input,
                 param_instance,
                 tumor_location):
        """Constructor

        Parameters:
        -----------
        data_files : dict of (str, str)
            Paths to images in which tumor appears hyperintense
            keys are in ['t1','t1c', 'flair', 't2'] (cfr. mr.MR_MODALITIES)
        param_instance : ems_initializer.Param()
            Parameter instance
        tumor_location : ems_initializer.TumorLocation()
            Tumor location instance

        .. note:: make sure that all the images referenced in
            images in data_files are in the same
            reference space and have the same pixel resolution and
            image dimensions.

        """

        self.set_data_files(data_input)

        if param_instance is None:
            param_instance = ei.Param()
        if not isinstance(param_instance, ei.Param):
            raise ValueError('param_instance is not a Param() instance')
        else:
            self.p = param_instance

        if tumor_location is None:
            tumor_location = ei.TumorLocation()
        if not isinstance(tumor_location, ei.TumorLocation):
            err = '`tumor_location_instance` is no `TumorLocation()` instance'
            raise ValueError(err)
        else:
            if tumor_location.tumor_lower_co is None:
                self.tlc = np.zeros(3, dtype='int')
            else:
                self.tlc = tumor_location.tumor_lower_co
                print 'TUMOR LOCATION: lower bounded by given coordinates' + \
                        str(self.tlc)
            if tumor_location.tumor_upper_co is None:
                self.tuc = self.dim - np.ones(3, dtype='int')
            else:
                self.tuc = tumor_location.tumor_upper_co
                print 'TUMOR LOCATION: upper bounded by given coordinates' + \
                        str(self.tuc)

    ########################################################################################

    def set_data_files(self, data_input):
        ''' Define paths to input brain tumor images

        Parameters:
        -----------
        data_input : dict of (str, str)
            Paths to images in which tumor appears hypointense
            keys are in mr.MR_MODALITIES

        Set self.data_itk_image, self.dim, self.data_spaces,
        self.data_type and self.full_data

        '''

        if not isinstance(data_input, dict):
            raise AttributeError('data_input must be supplied as a dict!')

        if False in [isinstance(path, str) for path in data_input.values()]:
            err = 'Please supplyy data_input as a dictionary of paths!'
            raise ValueError(err)
        self.modalities = sorted(data_input.keys())
        self.data_files = [data_input[key] for key in self.modalities]
        self.nr_channels = len(data_input)
        print 'DATA: channels ' + str(self.modalities)

        # Read in the data
        data, dim, spacing = [], [], []
        for channel in range(self.nr_channels):

            image, tmp_dim, tmp_spacing = \
                    oitk.get_itk_data(self.data_files[channel])
            data.append(image.flatten())
            dim.append(tuple(tmp_dim))
            tmp_spacing = np.around(np.asarray(tmp_spacing), 4)
            spacing.append(tuple(tmp_spacing))

        # Check dimensions
        if False in [this_dim == dim[0] for this_dim in dim]:
            print 'Different dimensions: %s' % (str(dim))
            err = 'Please make sure data dimensions are the same for all channels'
            raise ValueError(err)
        # Check spacing
        if False in [this_sp == spacing[0] for this_sp in spacing]:
            print 'Different pixel dimensions: %s' % (str(spacing))
            err = 'Please make sure pixel dimensions are the same for all channels'
            raise ValueError(err)
        # Store data in np array
        if True in [len(np.unique(im)) == 1 for im in data]:
                err = 'Empty channel for %s! Remove this and try again!' \
                        % (self.modalities[channel])
                raise ValueError(err)
        data = np.asarray(data, dtype=np.float)

        self.data_itk_image = oitk.get_itk_image(self.data_files[0])
        self.dim = dim[0]
        self.data_spaces = spacing[0]
        self.full_data = data

        print '\t shape ' + str(self.dim)
        print '\t spacing ' + str(self.data_spaces)
        print ''

############################################################################################

class ReaderEM(InputData):
    '''
    This class reads the input files for the em algorithm
    '''

    def __init__(self, data_input,
                        atlas_input,
                        mask_input,
                        distr_for_label=None,
                        param_instance=None,
                        tumor_location=None):
        '''
        Segments a brain image in tumor tissue and in brain tissues (white
        matter, grey matter and cerebrospinal fluid).

        Parameters
        ----------
        - data_input: dict (str: str)
            dictionary mapping modality to its corresponding image path
            or numpy array (these are the input channels)
        - atlas_input: dict (str: str)
            dictionary mapping label to its corresponding image path or
            numpy array (these are initial label maps)
        - mask_input: str
            path to binary mask valid over all images in data_files,
            can be None (atlas files will define ROI)
        - distr_for_label: dict (str:str)
            dictionary mapping each label to ('gauss', 'gumbel', 'gamma', 
            'frechet', 'logistic', 'auto', 'multivariate' or 'gmm') 
            indicating with which distribution this label should be modelled.
            Should have the same keys as atlas_input.

        .. NOTE:: All files should contain preprocessed images, in the sense that,
            all images should have the same resolution, the same isotropic pixel
            dimensions and they should be in the same reference space.
            Accepted file extensions are: *TIFF, JPEG, PNG, BMP, dicom_load, GIPL,
            Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate), Nrrd or VTK images*.
        '''

        InputData.__init__(self, data_input,
                                 param_instance,
                                 tumor_location)

        # Quick check mask input.
        if isinstance(mask_input, list):
            err = 'Please give me a single mask path ' + \
                    'instead of ' + self.mask_input
            raise AttributeError(err)
        self.mask_input = mask_input

        # Quick check atlas input.
        if not isinstance(atlas_input, dict):
            err = 'Please supply atlas files as a dictionary of ' + \
                    'label strings to image paths, arrays or floats'
            raise AttributeError(err)

        # Read in the labels.
        self.labels = sorted(atlas_input.keys())
        for tissue in ei.TISSUES:
            if tissue in self.labels:
                self.labels.remove(tissue)
                self.labels.append(tissue)
        self.nr_labels = len(self.labels)
        self.atlas_input = [atlas_input[key] for key in self.labels]

        # Set distribution for each label.
        if distr_for_label is None:
            distr_for_label = {key:'gauss' for key in self.labels}
        for label in self.labels:
            if label not in distr_for_label:
                err = 'Distribution for label %s not specified' % label
                raise AttributeError(err)
        distr = [distr_for_label[label] for label in self.labels]

        # load the respective distribution modules
        #CHANGED - get_distribution
        #self.distr_for_label = [ReaderEM.get_distribution(distr_str) for distr_str in distr]
        #CHANGED
        self.is_read = False

        print 'LABELS: ' + str(self.labels)
        print ''

    ########################################################################################

    def read(self):
        ''' Function reading in all data and setting most of the important instance variables.

        Typically called by ems_initializer.Initializer() '''

        self._set_mask()
        self._set_atlas()
        self._set_data()

        self._initialise_maps()

        self.is_read = True

    ########################################################################################

    def _set_mask(self):
        """ Set the mask

        Set self.playing, self.nr_brain_voxels

        """

        # Read in the mask
        mask = oitk.get_itk_array(self.mask_input)
        mask_dim = mask.shape
        if mask_dim != self.dim:
            err = 'Mask dimension %s != data dimension %s' \
                    % (str(mask_dim),str(self.dim))
            raise ValueError(err)

        # set mask
        playing = np.ones(shape=(np.prod(self.dim)), dtype='?')
        playing = np.logical_and(playing, mask.flatten())

        self.playing = playing
        self.nr_brain_voxels = np.count_nonzero(self.playing)

        volume = self.nr_brain_voxels * np.prod(self.data_spaces) / 1000000.
        print 'MASK: ' + str(self.nr_brain_voxels) + ' voxels'
        print 'MASK: ' + str(volume) + ' l'
        print ''

        return playing

    def _set_atlas(self):
        """ Normalise and mask the atlas.

        Set self.atlas

        """

        # Set self.atlas for the brain voxels
        self.atlas = np.zeros(shape=(self.nr_labels,
                                     self.nr_brain_voxels))

        for label in range(self.nr_labels):

            # atlas given as a flat prior, fill in later
            if isinstance(self.atlas_input[label],
                          (float, long, int)):

                # sub_sample self.p.sub_sample of the brain voxels
                np.random.seed(self.p.seed + label)
                samples = np.random.rand(self.nr_brain_voxels)
                indices = np.where(samples <= self.p.sub_sample)

                # set the selected voxels to the flat prior
                flat_prior = self.atlas_input[label]
                prior = np.zeros(self.nr_brain_voxels,)
                prior[indices] = flat_prior
                
                # Smooth 3d prior to avoid pepper salt
                prior_3d = self.to_matrix(prior)
                smoothed_3d = scim.filters.gaussian_filter(prior_3d, 2)
                smoothed_flat = smoothed_3d.flatten()
                smoothed_prior = np.fmax(smoothed_flat[self.playing], 0)
                
                self.atlas[label, :] = smoothed_prior

                # printout
                percent = np.around(100 * np.count_nonzero(prior) \
                                        / np.float(prior.size),
                                    decimals=2)
                print 'ATLAS: ' + self.labels[label] + \
                        ' flat prior ' + str(flat_prior) + \
                        ' subsample factor ' + str(self.p.sub_sample) + \
                        ' subsampled ' + str(percent) + '%'

            # atlas given as an array or a path to an array
            else:

                # check shape
                atlas_raw = oitk.get_itk_array(self.atlas_input[label])
                dim_atlas = atlas_raw.shape
                if dim_atlas != self.dim:
                    err = 'Please make sure atlas dimensions ' + \
                            'correspond to data dimensions'
                    raise ValueError(err)

                # fill atlas in for brain voxels
                full_atlas = atlas_raw.flatten()
                self.atlas[label, :] = full_atlas[self.playing]

                # printouts and adapt self.atlas_input if array was given
                if not isinstance(self.atlas_input[label], str):
                    self.atlas_input[label] = 'array given!'
                    print 'ATLAS: ' + self.labels[label] + \
                            ' array given'
                else:
                    print 'ATLAS: ' + self.labels[label] + ' path given ' + \
                            os.path.basename(self.atlas_input[label])

        # Normalize
        sum_atlas = np.fmax(np.sum(self.atlas, axis=0), ei.TINY)
        for label in range(self.nr_labels):
            self.atlas[label] = self.atlas[label] / sum_atlas

        print ''

    ########################################################################################

    def _set_data(self):
        """ Flatten and mask the data.

        Set self.data

        """
        self.data = np.zeros(shape=(self.nr_channels,
                                    self.nr_brain_voxels),
                             dtype=np.float)
        for channel in range(self.nr_channels):
            channel_data = self.full_data[channel, :]
            self.data[channel, :] = channel_data[self.playing]
            if len(np.unique(self.data[channel])) == 1:
                err = 'Data channel ' + self.modalities[channel] + \
                    ' has no information, remove it!'
                raise ValueError(err)

    ########################################################################################

    def _initialise_maps(self):
        ''' Initialise tumor classification.

        declared instance variables:
        ----------------------------
            - self.prior_maps
        '''
        # Initialise healthy tissue classification.
        prior_maps = np.zeros(shape=(self.nr_labels,
                                     self.nr_brain_voxels))

        for label_ind in range(self.nr_labels):
            prior_maps[label_ind, :] = self.atlas[label_ind, :]

        self.prior_maps = prior_maps

    ############################################################################
    # Helper functions (to be called after read() has been executed)
    ############################################################################

    def to_matrix(self, flat_array):
        ''' Rescale flat array, or the array of flat arrays,
        back into a matrix with dimensions self.dim.
        Each flat array should should either be of length brain voxels,
        or of the length of all voxels present in the 3d volume.'''

        def is_playing(arr):
            ''' Return weather the array contains only the brain voxels
            or all voxels of the entire 3d volume '''
            if arr.size == self.nr_brain_voxels:
                return True
            elif arr.size == np.prod(dim):
                return False
            else:
                err = 'this array does not have an interpretable size'
                raise RuntimeError(err)

        dim = self.dim

        if len(flat_array.shape) == 1:
            if is_playing(flat_array):
                flat_array_full = np.zeros((np.prod(dim)),)
                flat_array_full[self.playing] = flat_array
            else:
                flat_array_full = flat_array

            matrix = flat_array_full.reshape(dim)

        elif len(flat_array.shape) == 2:
            flat_arrays = flat_array
            shape = (flat_array.shape[0], dim[0], dim[1], dim[2])
            matrix = np.zeros(shape=shape)

            for i in range(flat_array.shape[0]):
                flat_array = flat_arrays[i]
                matrix[i, :, :, :] = self.to_matrix(flat_array)

        return matrix

    ########################################################################################
    
