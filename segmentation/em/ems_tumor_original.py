# -*- coding: UTF-8 -*-
"""Module containing the iterative EM implementation

"""
__version__ = '1.0'
__author__ = 'Esther Alberts'

import numpy as np
import scipy.stats._multivariate as m

import ems_initializer as ei

###############################################################################

class IteraterEM():
    """Class to segment a brain image in tumor tissue and in brain tissues
    (white matter, grey matter and cerebrospinal fluid).

    Results are written to files in a subdirectory
    of the first input image (as supplied to InputData()).

    Attributes
    ----------
    invalid : boolean
        If true, the initializer is invalid and the EM cannot be run
    r : ems_read_input.ReaderEM
        Reader containing information about input data and run parameters.
    p : ems_initializer.Param
        Parameter instance
    w : ems_write_output.WriterEM
        Writer instance
    label_maps : 2D array
        Posterior label maps
        (nr_gaussians, nr_brain_voxels)
    means : 2D array
        means per gaussian used to model tissue intensity distribution
        per channel (nr_gaussians, nr_channels)
    variances : ndarray
        variances per gaussian used to model tissue intensity distribution
        per channel (nr_gaussians, nr_channels)
    minimum
        minimum of the gamma distribution used to model the tumor intensity
        distribution per channel (nr_gaussians, nr_channels)
    shape
        shape of the gamma distribution used to model the tumor intensity
        distribution per channel (nr_gaussians, nr_channels)
    scale
        scale of the gamma distribution used to model the tumor intensity
        distribution per channel (nr_gaussians, nr_channels)
    converged : boolean
        If true, algorithm has finished its iterations
    iteration : int
        Iteration number
    log_llh_arr : 1D array
        Log likelihood values per iteration
    log_llh : float
        Most recent Log likelihood
    rel_log_llh : float
        Relative increase in Log likelihood to previous iteration

    """

    def __init__(self, em_initializer):
        """
        Initialises all variables needed to call self.run()

        Parameters
        ----------
        em_initializer : ems_initializer.Intializer
            Initialisation instance

        """

        if not isinstance(em_initializer, ei.Initializer):
            raise ValueError()

        self.invalid = em_initializer.invalid

        self.r = em_initializer.r
        self.p = em_initializer.p
        self.w = em_initializer.w

        if not self.invalid:
            self._initialisation()

    def _initialisation(self):
        """ Initialise all instance variables for the EM iterations,
        called by self.__init__(). """

        self._maps_initialisation()
        self._distr_initialisation()
        self._em_initialisation()

    def _maps_initialisation(self):
        """ Initialise map instance variables. """

        # Initialise the maps
        self.label_maps = np.copy(self.r.atlas)
        
        self.atlas_maps = np.copy(self.r.atlas)

        self.set_stable_prior()

    def _distr_initialisation(self):
        """ Initialise gaussian instance variables. """

        self.invalid_label_indices = []

        # Initialisations for intensity guassians used to model
        # healthy tissue intensity distribution.
        self.univariates = [None for __ in range(self.r.nr_labels)]

    def _em_initialisation(self):
        """ Initialise instance variables related to the EM itself. """

        # Initialisations for iterations.
        self.converged = 0
        self.iteration = 0

        # Log-likelihood initialisations.
        self.log_llh_arr = np.zeros((self.p.max_iteration,))
        self.log_llh = 0
        self.rel_log_llh = 1
        self.previousLogLLH = self.log_llh
        
    def set_stable_prior(self, labels=None):
        """For these labels, assume the input maps are highly reliable.
        """

        if labels is None:
            # Set all classes stable
            labels = self.r.labels
        self.stable_prior = [True if self.r.labels[i] in labels else False
                             for i in range(self.r.nr_labels)]

    def get_result_paths(self):
        """ Get dictionary of labels towards the result paths. """
        
        labels = self.r.labels
        paths = self.w.label_paths
        result_paths = dict(zip(labels, paths))
        
        return result_paths

    def run(self):
        """ Start EM iterations """

        if not self._check_ready_to_run():
            return

        self._print_start_message()

        # Start iterating.
        while self.converged == False:

            self.previousLogLLH = self.log_llh

            self._print_start_iteration()

            ##########################
            # 1. Calculate distribution parameters.
            self._update_distr_param()
            # 3. Calculate classification.
            self._evaluate_configurations()
            ###########################

            # Convergence criterion.
            self.converged = self.iteration == self.p.max_iteration - 1

            # visualisation.
            if not self.converged:
                self.w.save_iteration(self.label_maps,
                                      self.iteration)
            self.iteration = self.iteration + 1

        # Write and visualise the average of all segmentations
        self.w.save_on_convergence(self.label_maps,
                                   self.log_llh_arr)

        # Return the segmentations, probabilistic and binary, in a
        # dictionary format so that the labels are explicit
        label_maps_dict_proba = {}
        for label_ind in range(self.r.nr_labels):
            label = self.r.labels[label_ind]
            label_maps_dict_proba[label] = \
                 self.r.to_matrix(self.label_maps[label_ind])

        return label_maps_dict_proba

    def _check_ready_to_run(self):
        """ Return whether EM algorithm has already been run with this
        em initializer. """

        if self.invalid:
            err = 'Initialisation object is invalid, ' + \
                        'results cannot be overwritten'
            print(err)
            return False

        if self.converged:
            err = 'This instance has already been run, ' + \
                    'reset before running it again'
            print(err)
            return False

        return True

    ####################################################################

    def _update_distr_param(self):
        ''' Function calculating distribution parameters for each label
        for each channel.

        Instance variables:
        -------------------
         - Called: self.label_maps, self.r.data
         - Modified: self.univariates

        '''

        print('1. PARAMETER UPDATE: '),
        print('Calculating distribution parameters')

        for label_ind in range(self.r.nr_labels):

            if label_ind in self.invalid_label_indices:
                continue

            # get the probability map for this label
            weights = self.label_maps[label_ind, :]
            self._print_label(self.r.labels[label_ind], weights)

            # if no voxels are likeli to belong to this distribution, remove the label
            if np.sum(weights) == 0:
		self.invalid_label_indices.append(label_ind)
                #self._make_label_invalid(label_ind)
                continue
            
            # just do a multivariate gaussian
            print '\t - Multivariate Gaussian over the channels'
            data = self.r.data.astype(np.float)
            means = np.average(data, axis=1, weights=self.label_maps[label_ind])
            cov = np.cov(data, ddof=0, aweights=self.label_maps[label_ind])
            self.univariates[label_ind] = means, cov
            
            print ''
                        
    def _evaluate_configurations(self):
        ''' Function evaluating configurations incorporating prior and
            current parameters.

            Instance variables:
            -------------------
            Called: self.r.data,
                    self.means, self.variances,
                    self.minimum, self.shape, self.scale
            Modified: self.label_maps, self.log_llh, self.log_llh_arr,
                    self.rel_log_llh

        '''
        print('2. CALCULATING POSTERIOR'),
        print('')

        joint_label_data = np.zeros(shape=(self.r.nr_labels,
                                           self.r.nr_brain_voxels))

        # add label maps
        print('\t Calculating prior')
        prior_maps = self.get_prior_maps()
        print('\t Calculating marginals and copula for each label')
        for label_ind in range(self.r.nr_labels):
            if label_ind in self.invalid_label_indices:
                continue
            joint_label_data[label_ind, :] = \
                    self._get_joint_data_given_label(label_ind) \
                    * prior_maps[label_ind, :]

        # Calculate data likelihood by summing joint_label_data over labels.
        data_likelihood = np.sum(joint_label_data, axis=0)
        self._print_round_err(data_likelihood, ei.TINY,
                              'Normalizing likelihood (over all labels)')
        data_likelihood = np.fmax(data_likelihood, ei.TINY)

        # Get label posterior by dividing joint_label_data by data_likelihood.
        label_maps = joint_label_data / data_likelihood

        self._print_round_err(label_maps, ei.EPS,
                              'Lower bounding label maps (over all labels)',
                              zeros_expected=True)
        if not np.all(np.isfinite(label_maps)):
            err = 'NaNs in label maps...'
            raise RuntimeError(err)
        label_maps = np.fmax(label_maps, ei.EPS)
        label_maps = np.fmin(label_maps, 1)
        for label_ind in self.invalid_label_indices:
            label_maps[label_ind][:] = 0
        self.label_maps = label_maps

        # Calculate log-likelihood over all labels
        self.log_llh = np.sum(np.log(data_likelihood));
        self.log_llh_arr[self.iteration] = self.log_llh
        self.rel_log_llh = \
            (self.log_llh - self.previousLogLLH) / self.previousLogLLH

        self._print_llh()

        print ''

    def _get_joint_data_given_label(self, label_index):
        ''' Get likelihood for this label.


        Instance variables:
        -------------------
        Called: self.r.atlas_prior, self.tumor_atlas

        '''
        # for initialisation, one might only want to evaluate over a
        # subset of the channels

        # Calculate multivariate gaussian
        means, cov = self.univariates[label_index]
        data = self.r.data.astype(np.float)
        
        likeli_product = \
            evaluate_gaussian_multivariate(data, means, cov)
            
        return np.fmax(likeli_product, 0)

    def get_prior_maps(self):
        """ Set prior label probabilities for the next iteration. """

        # get posteriors of previous iteration
        prior_maps = np.zeros_like(self.label_maps)
        self._get_prior_unchanged(prior_maps)
        
        # normalize
        sum_norm = np.sum(prior_maps, axis=0)
        prior_maps = prior_maps / np.fmax(sum_norm, ei.TINY) # not /= !!
        self._print_round_err(sum_norm, ei.TINY,
                              'Normalizing prior maps over all labels',
                              zeros_expected=True) 

        # visualise if wished
        if self.iteration == self.p.max_iteration - 1:
            self.w.visualise_volumes(prior_maps, 'Prior maps')
            
        return prior_maps
    
    def _get_prior_unchanged(self, prior_maps):
        """ Set the prior maps as flat prior or as the atlas
        input for the stable prior labels."""     
        
        # print message
        self._print_stable_prior()

        # get priors 
        for label_ind in range(self.r.nr_labels):
            
            if label_ind in self.invalid_label_indices:
                continue
            
            if not self.stable_prior[label_ind]:
                # get posteriors of previous iteration
                perc = np.sum(self.label_maps[label_ind]) / \
                                        np.float(self.r.nr_brain_voxels)
                prior_maps[label_ind] = np.max((0.02, perc))
            else:
                # get atlas prior
                prior_maps[label_ind] = self.atlas_maps[label_ind]   

    ####################################################################

    def _print_start_message(self):
        """Print a start message."""

        print('')
        print('EM starting')
        print('Remember: ')
        print('\t no biasfield correction')
        print('\t no MRF spatial regularisator')
        print('\t Input data is supposed to be biasfield corrected')
        print('\t All input images should be in the same reference space')
        print('')

        self._print_class_start_message()

    def _print_class_start_message(self):
        """ Print a start message specific to this class. """

        print('MULTIVARIATE VERSION WITH COPULAS HAS BEEN STARTED!!!')

    def _print_start_iteration(self):

        print('---------------')
        print('Iteration %d' % self.iteration)
#         memory = float(awk(ps('u', '-p', os.getpid()),
#                                '{sum=sum+$6}; END {print sum/1024}'))
#         print('Memory used: ' + str(memory) + ' MB')
        print('')

    def _print_label(self, label, weights):

        print '\t LABEL %s:' % label,
        print '%.2f %% voxels with maximum probability %.2f %%' % (\
                 self._print_procent(np.count_nonzero(weights > ei.EPS)),
                 np.max(weights) * 100.)
        if np.sum(weights) == 0:
            print '\t\t Label map is empty!!!'
            print '\t\t Sorry... %s is not considered...' % label

    def _print_llh(self):

        print('\t logLikelihood = %f' % (self.log_llh))
        print('\t loglikelihood increase = %.2f %%' % (self.rel_log_llh * 100.))
        print('')

    def _print_round_err(self,
                         variable,
                         rounding_value,
                         case_string='',
                         zeros_expected=False):

        to_print = ''
        total = variable.size
        neg_values = np.count_nonzero(variable < 0)
        if neg_values > 0:
            to_print += '\t\tAlgorithmic error, '
            to_print += 'voxels with neg probabilities: %.2f %%\n' % (\
                            self._print_procent(neg_values, total))

        lost_zeros = np.count_nonzero(variable == 0)
        if not zeros_expected and lost_zeros > 0:
            to_print += '\t\tVoxels set to min due to multiplying '
            to_print += 'too small values: %.2f %%\n' % (\
                            self._print_procent(lost_zeros, total))

        if rounding_value > 0:
            lost = np.count_nonzero(variable <= rounding_value) \
                    - lost_zeros - neg_values
            if lost > 0:
                to_print += '\t\tVoxels set to min due to forcing '
                to_print += 'a lower positive value: %.2f %%\n' % (\
                                self._print_procent(lost, total))

        if len(to_print) > 0:
            print '\t %s' % case_string
            print to_print

    def _print_procent(self, value, total=None):

        if total is None:
            total = self.r.nr_brain_voxels
        procent = value / np.float(total)
        procent = procent * 100.

        return procent
    
    def _print_stable_prior(self):
        
        unstables = [self.r.labels[ind] for ind in range(self.r.nr_labels)
                   if self.stable_prior[ind] is False]
        stables = [self.r.labels[ind] for ind in range(self.r.nr_labels)
                   if self.stable_prior[ind] is True]
        msg = ''
        print('\t  - Adding %s flat prior for %s' % (msg, unstables))
        print('\t  - Adding %s atlas prior for %s' % (msg, stables))

########################################################################

def evaluate_gaussian_multivariate(data, means, cov):
    """ Evaluate gaussian multivariate using scipy package. 
    
    Parameters
    ----------
    data : arraylike, shape (n,s)
        data to be evaluated under a gaussian
    mean : arraylike, shape (n,)
        gaussian mean
    var : arraylike, shape (n,n)
        gaussian covariance matrix
    normalized : bool
        True if gaussian should integrate to 1, False if gaussian max
        should be at 1.
    """
    
    err = ''
    if len(data.shape) !=2 or np.any(data.shape == 1):
        err = 'Please give a 2D data array (n, s)'
    elif means.shape[0] != data.shape[0]:
        err = 'Please supply %d means' % data.shape[0]
    elif len(cov.shape) != 2 or cov.shape[0] != cov.shape[1]:
        err = 'Please supply a square covariance of size %d' % data.shape[0]
    if len(err) > 0:
        raise ValueError(err)
    
    if len(means.shape) == 1:
        means = means[:,np.newaxis]
        
    diff = data.astype(np.float) - means
      
    psd = m._PSD(cov, allow_singular=True)
    maha = np.sum(np.square(np.dot(diff.T, psd.U)), axis=-1)
    log_pdf = -0.5 * (psd.rank * m._LOG_2PI + psd.log_pdet + maha)
          
    density = np.exp(log_pdf)
    
    return density
