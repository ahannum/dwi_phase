
import numpy as np
import os
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
import nrrd

import sys
sys.path.append("/Users/arielhannum/Documents/Stanford/CMR-Setsompop/Code/cDTI_python")
from mystic_mrpy.Data_Import.Diffusion   import *
from mystic_mrpy.Data_Sorting.Diffusion  import *
from mystic_mrpy.Diffusion.DWI  import *
from mystic_mrpy.Diffusion.Gibbs         import *
from mystic_mrpy.Diffusion.Registration  import *
from mystic_mrpy.Diffusion.Rejection     import *
from mystic_mrpy.Diffusion.Respiratory   import *
from mystic_mrpy.Diffusion.Averaging     import *
from mystic_mrpy.Diffusion.Denoising     import *
from mystic_mrpy.Diffusion.Interpolation import *
from mystic_mrpy.Diffusion.Segmentation_Matrix_DTI import *
from mystic_mrpy.Diffusion.DTI import *
from mystic_mrpy.Diffusion.cDTI import *

def get_edge(img):
    #define the vertical filter
    vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]

    #define the horizontal filter
    horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

    #get the dimensions of the image
    n,m = img.shape

    #initialize the edges image
    edges_img = img.copy()

    #loop over all pixels in the image
    for row in range(3, n-2):
        for col in range(3, m-2):

            #create little local 3x3 box
            local_pixels = img[row-1:row+2, col-1:col+2]

            #apply the vertical filter
            vertical_transformed_pixels = vertical_filter*local_pixels
            #remap the vertical score
            vertical_score = vertical_transformed_pixels.sum()/4

            #apply the horizontal filter
            horizontal_transformed_pixels = horizontal_filter*local_pixels
            #remap the horizontal score
            horizontal_score = horizontal_transformed_pixels.sum()/4

            #combine the horizontal and vertical scores into a total edge score
            edge_score = (vertical_score**2 + horizontal_score**2)**.5

            #insert this edge score into the edges image
            edges_img[row, col] = edge_score*2

    #remap the values in the 0-1 range in case they went out of bounds
    edges_img = edges_img/edges_img.max()
    edges_img[edges_img ==0] = 'nan'
    edges_img[edges_img >0] = 1
    return edges_img


def load_image(motion, timepoint, volunteer,diffusion,slice, directory):
    inpath = os.path.join(directory,'V00'+str(volunteer),'3_DWI')

    data,affine, voxsize = load_nifti(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'), return_voxsize=True)
    bvals = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvals')) 
    bvecs = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvecs'))

    mask,header =  nrrd.read(os.path.join(inpath,  'M'+str(motion)+'_mask_new.nrrd'))
    mask = mask.astype('float')
    mask[mask==0] = np.nan

    data1,bvals_sort,bvecs_sort = stacked2sorted(data,bvals,bvecs.T)
        
    ims = np.stack((data1[:,:,:,:,:5],data1[:,:,:,:,5:10],data1[:,:,:,:,10:15],
                        data1[:,:,:,:,15:20],data1[:,:,:,:,20:25],data1[:,:,:,:,25:30],
                        data1[:,:,:,:,30:35],data1[:,:,:,:,35:40]),axis = -1)  

    disp_im = ims[:,:,slice,:,:,timepoint]
    del ims 

    mag = np.abs(disp_im)
    phs_stds, phs_diffs = get_phs_diff(disp_im)

    return phs_stds[:,:,diffusion], phs_diffs[:,:,diffusion], np.nanmean(mag[:,:,diffusion,:],axis = -1),mask[:,:,slice]
    


def get_phs_diff(im):
    """
    Calculate phase difference between each image rep and average phase
    Input: complex image [x dim, y dim, diffusion dirs, reps]
    Output: phs difference [x dim, y dim, diffusion dirs, reps]
    
    """
    phs = np.angle(im)
    im_adj = (np.exp(1j*phs))/np.nanmean(np.exp(1j*phs),axis = -1)[:,:,:,np.newaxis]
    phs_diff = np.angle(im_adj / np.tile(np.nanmean(im_adj,axis = -1)[:,:,:,np.newaxis],(1,1,1,im_adj.shape[-1])))

    phs_std = np.sqrt(np.sum(phs_diff**2,axis = -1)/5)

    return phs_std,phs_diff
    
    
