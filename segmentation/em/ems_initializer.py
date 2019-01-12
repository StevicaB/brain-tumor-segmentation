#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Module containing
- two classes containing parameters: Param() and TumorLocation()
- class for initialisation of EM brain tumor segmentation

"""
__version__ = '1.0'
__author__ = 'Esther Alberts'

import numpy as np

import ems_read_input as ri
import ems_write_output as wo

########################################################################################

GM = 0
WM = 1
CSF = 2
TISSUES = ['gm', 'wm', 'csf']

EPS = np.finfo(np.double).eps
TINY = np.finfo(np.double).tiny

TUMOR = 1
NO_TUMOR = 0

########################################################################################

class Param():

    def __init__(self,
                 sub_sample=0.75,
                 seed=1401,
                 max_iteration=15,
                 gaussian_norm=True):
        """Constructor

        Parameters
        ----------
        sub_sample: np.float (0,1]
            percentage of voxels selected randomly to initialize the
            class maps for classes for which a flat prior is specified
            instead of a full map.
        seed : np.int
            a seed for the random number generator
        max_iteration: int [15]
            how many iterations the EM has to do
        guassian_norm: boolean [True]
            whether the gaussian evaluations have to be normalised

        """
        if sub_sample <= 0 or sub_sample > 1:
            err = 'sub_sample factor should be in (0,1]'
            raise ValueError(err)
        self.sub_sample = sub_sample
        self.seed = seed
        self.max_iteration = max_iteration
        self.gaussian_norm = gaussian_norm

########################################################################################

class TumorLocation():
    """Class to keep information about outer tumor coordinatees

    """

    def __init__(self, tumor_lower_co=None, tumor_upper_co=None):

        if isinstance(tumor_lower_co, list):
            tumor_lower_co = np.asarray(tumor_lower_co)
        if isinstance(tumor_upper_co, list):
            tumor_lower_co = np.asarray(tumor_upper_co)
        self.tumor_lower_co = tumor_lower_co
        self.tumor_upper_co = tumor_upper_co


########################################################################################

class Initializer():
    """Class integrating information from ems_read_input.py and
    ems_write_output.py to generate a valid initializer for
    ems_tumor_original.py. Sets directories where results will be
    written in ems_write_output.WriteOutput.

    """

    def __init__(self, read_instance, write_instance=wo.WriterEM()):
        """
        Connects reading and writing instance to create
        valid em initializer.

        Parameters
        ----------
        read_instance: ems_read_input.ReaderEM
        write_instance: ems_write_output.WriteOuput

        """

        if not isinstance(write_instance, wo.WriterEM):
            raise ValueError('write_instance ' + \
                             ' is not a wo.WriterEM() instance')
        self.w = write_instance
        if not isinstance(read_instance, ri.InputData):
            raise ValueError('read_instance ' + \
                             'is not a ri.ReaderEM() instance')
        self.r = read_instance

        self.w.connect_to_read(self.r)

        self.p = self.w.r.p

        self.invalid = self.w.abort

        self.initialize()

    def initialize(self):
        """ Initialise read instance, if it hasn't been read yet.

        """

        if not self.is_valid_for_start():
            label_paths = self.w.label_paths

            # print existing paths
            print 'em Already calculated for these paths'
            print 'Segmentation paths ' + str(label_paths)

        elif not self.r.is_read:
            self.r.read()
            self.w.visualise_input()

    def get_metadata_path(self):
        """Get the path where metadata should be written."""

        return self.w.metadata_path

    def is_valid_for_start(self):
        """Ask if object is valid to use for
        ems_tumor_original.IteraterEM."""

        return not self.invalid
