"""
This module provides and interface to skull strip using given modalities.

Usage:
    Initialize an instance.
    e.g:
        skull_stripper = SkullStripper()

    Run skull stripping method:
    e.g:
        mask_path = skull.stripper.strip_skull(modalities, T1_id=1)
"""

import subprocess
import shlex
import os
import helpers as utils
import numpy as np
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.image import load_img, math_img, threshold_img
import time
import registration as reg
import paths

class SkullStripper():

    #initialize the paths for the atlas folder, output folder and sh scripts.
    # in our case id_folder is the patient name
    def __init__(self, output_folder="temp", id_folder=""):

        self.T1_REG_PREFIX = "t1_reg_"
        self.BRATS_REG_PREFIX = "brats_reg_"
        self.files = []
        self.atlas = utils.get_relative_path("Atlas")
        self.ss_sh_path = utils.get_relative_path(os.path.join("sh", "skull_strip.sh"))
        self.ra_sh_path = utils.get_relative_path(os.path.join("sh", "rigid_affine.sh")) 
        self.brats_transform_path = utils.get_relative_path(os.path.join("sh", "brats_transform.sh"))
        self.brats_ra_path = utils.get_relative_path(os.path.join("sh", "ra_brats.sh"))
        self.output_folder = output_folder
        self.id_folder = id_folder
        self.BRATS_SPACE_PATH = utils.get_relative_path(os.path.join("Atlas","t1_brats_space.nii"))
        self.STRIPPED_BRATS = utils.get_relative_path(os.path.join("Atlas","t1_skullstripped_brats_space.nii"))

    def register(self, fix_image, mov_image, prefix):
        """ Perform rigid registration between two images
        Parameters
        ----------
        fixed_image : String
            path of the image which will be the fixed image
        mov_image: String
            path of the image which will be the moving image
     
        Returns
        -------
        registered_image : String
            path for the registered image
        """

        modality_name = os.path.splitext(os.path.basename(mov_image))[0]
        registered_image = utils.get_relative_path(os.path.join(self.output_folder, prefix + os.path.basename(mov_image)))

       	command = shlex.split("%s %s %s %s %s" % (self.ra_sh_path, fix_image, mov_image, registered_image, modality_name))
        registration = subprocess.call(command)


        return registered_image

    def calculate_brats_transform(self, fix_image, mov_image, prefix):

        modality_name = os.path.splitext(os.path.basename(mov_image))[0]
        registered_image = utils.get_relative_path(os.path.join(self.output_folder, "brats_transform"))

        command = shlex.split("%s %s %s %s %s" % (self.brats_transform_path, fix_image, mov_image, registered_image, modality_name))
        registration = subprocess.call(command)


        return registered_image

    def register_brats(self, fix_image, mov_image, prefix, transform_prefix):

        modality_name = os.path.splitext(os.path.basename(mov_image))[0]
        registered_image = utils.get_relative_path(os.path.join(self.output_folder, prefix + os.path.basename(mov_image)))
        transform_path = utils.get_relative_path(os.path.join(self.output_folder, transform_prefix))
        command = shlex.split("%s %s %s %s %s" % (self.brats_ra_path, fix_image, mov_image, transform_path, registered_image))
        registration = subprocess.call(command)

        return registered_image

    def register_modalities(self, modality_files, t1_id=0):
        """ Perform rigid registration for all modalities in order to put them into the same coordinate system
        Parameters
        ----------
        modality_files : String array
            paths of the files to register
        selected_modality: int
             index of the modality to use a a base image in registration. If not specified(=-1), the max resolution one will be selected
        Returns
        -------
        paths : String Array
            path for the registered images
        """

        t = time.time()

        print("Modalities to register to T1 space:")
        print(modality_files)
        print("T1 modality as registration space:")
        print(modality_files[t1_id])

        print("------------- \n Registration to the T1 space started.")
        #First do the registration to the T1 space
        registration_space = modality_files[t1_id]
        registration_to_t1_paths = []

        for i in range(len(modality_files)):
            if i == t1_id:
                registration_to_t1_paths.append(modality_files[t1_id])
                continue
            moving_image = modality_files[i]
            registered_image = self.register(registration_space, moving_image, self.T1_REG_PREFIX)
            registration_to_t1_paths.append(registered_image)

        elapsed_time = time.time() - t
        print("Time for T1 Registration: %s" % (elapsed_time))

        print("Registration to the T1 space finished. Result files:")
        print(registration_to_t1_paths)

        print("Modalities to register to BRATS Space: ")
        print(registration_to_t1_paths)
        print("BRATS as registration space:")
        print(self.BRATS_SPACE_PATH)
        print("------------- \n Registration to the BRATS space started.")

        registration_space = self.BRATS_SPACE_PATH
        registration_to_brats_paths = []

        t = time.time()

        self.calculate_brats_transform(registration_space, registration_to_t1_paths[t1_id], "brats_transform")
        #Register all to the brats space
        for modality in registration_to_t1_paths:
            moving_image = modality
            registered_image = self.register_brats(registration_space, moving_image, self.BRATS_REG_PREFIX, "brats_transform0GenericAffine.mat")
            registration_to_brats_paths.append(registered_image)
        
        elapsed_time = time.time() - t

        print("Time for BRATS Registration: %s" % (elapsed_time))

        print("Registration to the BRATS space finished. Result files:")
        print(registration_to_brats_paths)

        for i in range(len(registration_to_t1_paths)):
            if i != t1_id:
                os.remove(registration_to_t1_paths[i])

        return registration_to_brats_paths

    def deformable_registration(self, atlas_path, anatomy_path, name):
        aff_reg = os.path.join(self.output_folder, 't1_atlas_aff_reg.nii.gz')
        aff_trans = os.path.join(self.output_folder, 't1_atlas_aff_transformation.txt')
        f3d_cpp = os.path.join(self.output_folder, 't1_atlas_f3d_cpp.nii.gz')
        f3d_reg = os.path.join(self.output_folder, name + '_atlas_reg_deform.nii.gz')
        tissue_reg = os.path.join(self.output_folder, name)
        tissueAtlas = os.path.join(self.output_folder, name + '_')

        reg.niftireg_affine_registration(atlas_path, anatomy_path, transform_path=aff_trans, result_path=aff_reg)
        reg.niftireg_nonrigid_registration(atlas_path, anatomy_path, transform_path = aff_trans, cpp_path=f3d_cpp, result_path=f3d_reg)
        for tissue in ["csf", "gm", "wm"]:
            reg.niftireg_transform(tissueAtlas + tissue + ".nii.gz",anatomy_path,f3d_cpp,result_path=tissue_reg + "_" + tissue + "temp.nii.gz",cpp=True)
            img = nib.load(tissue_reg + "_" + tissue + "temp.nii.gz")
            mask = math_img('(img - np.min(img))/(np.max(img)-np.min(img))',img=img)
            nib.save(mask, tissue_reg + "_" + tissue + "temp.nii.gz")
            os.remove(tissueAtlas + tissue + ".nii.gz")
            os.rename(tissueAtlas + tissue + "temp.nii.gz", tissueAtlas + tissue + ".nii.gz")
        os.remove(aff_reg)
        os.remove(aff_trans)
        os.remove(f3d_cpp)

    def apply_mask(self, anatomy_path, mask_path, name):
        """Apply calculated mask to the brain image
        Parameters
        ----------
        anatomy_path: String
            Path of the brain to mask
        mask_path: String
            Path of the mask
        name: String
            Name of the modality
        """
        mask = nib.load(os.path.join(self.output_folder, mask_path))
        mask = math_img('img > 0.9', img=mask)
        nib.save(mask, os.path.join(self.output_folder, mask_path))
        patient = nib.load(anatomy_path)

        masked_data = np.multiply(patient.get_data(), mask.get_data())
        masked_data = nib.Nifti1Image(masked_data, patient.affine, patient.header)

        path_to_save = utils.get_relative_path(os.path.join(self.output_folder,
                                                            "stripped_" + \
                                                            name + \
                                                            "_" + \
                                                            self.id_folder+ \
                                                            ".nii.gz"))

        nib.save(masked_data, path_to_save)

        return path_to_save


    def strip_skull(self, modality_files, t1_id):
        """ Perform skull stripping
        Parameters
        ----------
        modality_files : String array
            paths of the files to register
        register_modalities: boolean
            will be true if the modalities are already registered and there is no need to register before stripping
        selected_modality: int
            index of the modality to use a a base image in registration. If not specified(=-1), the max resolution one will be selected
        """

        print(utils.get_relative_path(os.path.join("Atlas", "atlas_" + "wm" + ".nii")))

        print("Input Modality Files: ------------------")
        print(modality_files)


        print("Rigid registration started.\n --------------------")
        #Register modalities first to T1 and to BRATS space
        self.files = self.register_modalities(modality_files, t1_id)
        print("Rigid registration finished.\n --------------------")

        t = time.time()

        print("Skull stripiing started. \n -------------")
        moving_image = self.atlas
        print("Moving image: %s" % moving_image)

        #Skull strip with T1 modality
        fixed_image = self.files[t1_id]
        print("Fixed image: %s" % fixed_image)
        modality_name = os.path.basename(fixed_image).split('.')[0]
        #run the sh script to strip the skull
        command = shlex.split("%s %s %s %s %s" % (self.ss_sh_path, fixed_image, moving_image, self.output_folder, self.id_folder))
        stripping = subprocess.call(command)

        stripped_images=[]
        print()
        #Apply masks
        for modality in self.files:
            fixed_image = modality;
            file_name = os.path.splitext(os.path.basename(modality))[0].split('_')[2]
            if file_name == "T1":
                modality_name = "T1"
            else:
                modality_name = os.path.splitext(os.path.basename(modality))[0].split('_')[4]
            stripped_image = self.apply_mask(fixed_image, self.id_folder+ "_mask.nii.gz", modality_name)
            stripped_images.append(stripped_image)

        #Apply mask to Atlas registered
       	atlas_reg_path = utils.get_relative_path(os.path.join(self.output_folder, self.id_folder+"_atlas_reg.nii.gz" ))
       	stripped_atlas = self.apply_mask(atlas_reg_path, self.id_folder+ "_mask.nii.gz", "atlas")
        self.deformable_registration(stripped_atlas, stripped_images[0], self.id_folder)
        elapsed_time = time.time() - t
        print("Time for Skull Stripping: %s" % (elapsed_time))

       	for file in self.files:
			os.remove(file)