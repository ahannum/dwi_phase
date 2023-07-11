from Diffusion import stacked2sorted
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
import nibabel as nib
import nrrd
import numpy as np
import os
import scikit_posthocs as sp
from scipy import stats
import sys


def get_edge(img):
    """
    Function to get outline of image mask 
    Input: image mask
    Output: image mask outline
    """
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

def load_image(motion,  volunteer,diffusion,slice, directory):
    """
    Load image data for a given volunteer, motion, diffusion direction and slice
    Input: motion, volunteer, diffusion direction, slice, directory
    Output: phase standard deviation msp, phase difference map, magnitude, mask
    """
    inpath = os.path.join(directory,'V00'+str(volunteer),'DWI')
    # Load only data for a given slice
    test = nib.load(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    #print(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    data = test.dataobj[:,:,slice,:] #load only the diffusion directions of interest from test variable
    # Load Bvals and Bvecs
    bvals = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvals')) 
    bvecs = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvecs'))
    # Load mask
    mask_LV,affine, voxsize = load_nifti(os.path.join(inpath, 'LV_M'+str(motion)+'.nii'), return_voxsize=True)
    mask_BP,affine, voxsize = load_nifti(os.path.join(inpath, 'BP_M'+str(motion)+'.nii'), return_voxsize=True)
    mask = (mask_LV-mask_BP).astype('float')
    mask[mask==0] = np.nan
    # Sort data
    data1,bvals_sort,bvecs_sort = stacked2sorted(data[:,:,np.newaxis,:],bvals,bvecs.T)

    if data1.shape[0] == 100:
        disp_im = np.flip(np.squeeze(data1.transpose(1,0,2,3,4)),axis = 1)
        mask = np.flip(mask.transpose(1,0,2),axis = 1)
    else:
        disp_im = np.squeeze(data1)
        

    # Get magnitude and phase temporal variation
    mag = np.abs(disp_im)
    phs_stds, phs_diffs = get_phs_diff(disp_im)

    return phs_stds[:,:,diffusion], phs_diffs[:,:,diffusion], np.nanmean(mag[:,:,diffusion,:],axis = -1),mask[:,:,slice]

def get_phs_diff(im):
    """
    Calculate phase difference between each image rep and average phase
    Input: complex image [x dim, y dim, diffusion dirs, reps]
    Output: phs difference [x dim, y dim, diffusion dirs, reps]
    
    """
    # Calculate the phase
    phs = np.angle(im)
    # Divide by the average backgrond ( b=0) phase
    im_adj = (np.exp(1j*phs))/np.nanmean(np.exp(1j*phs),axis = -1)[:,:,:,np.newaxis]
    # Take the difference between each image rep and the average phase
    phs_diff = np.angle(im_adj / np.tile(np.nanmean(im_adj,axis = -1)[:,:,:,np.newaxis],(1,1,1,im_adj.shape[-1])))
    # Calculate the standard deviation of the phase difference
    phs_std = np.sqrt(np.sum(phs_diff**2,axis = -1)/5)

    return phs_std,phs_diff


def get_phs_diff_wholeData(im):
    """
    Calculate phase difference between each image rep and average phase
    Input: complex image [x dim, y dim, diffusion dirs, reps]
    Output: phs difference [x dim, y dim, diffusion dirs, reps]
    
    """
    # Calculate the phase
    phs = np.angle(im)
    # Divide by the average backgrond ( b=0) phase
    im_adj = (np.exp(1j*phs))/np.nanmean(np.exp(1j*phs[:,:,:,0,:]),axis = -1)[:,:,:,np.newaxis,np.newaxis]
    # Take the difference between each image rep and the average phase
    phs_diff = im_adj / np.tile(np.nanmean(im_adj,axis = -1)[:,:,:,:,np.newaxis],(1,1,1,1,5))
    # Calculate the standard deviation of the phase difference
    phs_std = np.sqrt(np.sum(np.angle(phs_diff[:,:,:,:,:])**2,axis = -1)/5)

    return phs_std,phs_diff

def load_image_all(motion,  volunteer, directory):
    """
    Load image data for a given volunteer, motion, and filepath
    Input: motion, volunteer, diffusion direction, slice, directory
    Output: phase standard deviation maps for volunteer,  mask
    """

    inpath = os.path.join(directory,'V00'+str(volunteer),'DWI')
    # Load only data for a given slice
    test = nib.load(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    #print(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    data = test.dataobj #load only the diffusion directions of interest from test variable
    # Load Bvals and Bvecs
    bvals = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvals')) 
    bvecs = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvecs'))
    # Load mask
    mask_LV,affine, voxsize = load_nifti(os.path.join(inpath, 'LV_M'+str(motion)+'.nii'), return_voxsize=True)
    mask_BP,affine, voxsize = load_nifti(os.path.join(inpath, 'BP_M'+str(motion)+'.nii'), return_voxsize=True)
    mask = (mask_LV-mask_BP).astype('float')
    mask[mask==0] = np.nan
    # Sort data
    data1,bvals_sort,bvecs_sort = stacked2sorted(data,bvals,bvecs.T)

    if data1.shape[0] == 100:
        disp_im = np.flip(np.squeeze(data1.transpose(1,0,2,3,4)),axis = 1)
        mask = np.flip(mask.transpose(1,0,2),axis = 1)
    else:
        disp_im = np.squeeze(data1)
    # Get magnitude and phase temporal variation
    phs_std,__ = get_phs_diff_wholeData(disp_im)
    del disp_im,data1,data,bvals,bvecs
    return phs_std, mask

def get_tempPhs_mean_std(motion, directory,list_vols ):
    """
    Calculate the mean and standard deviation of the phase standard deviation
    Input: motion, td, diffusion, slice,directory
    Output: Mean and Standard deviation for 10 volunteers and 8 timepoints[timepoints, volunteers]  
    """
    mean = np.zeros((3,4,10))
    for vv in range(10):
        volunteer = list_vols[vv]
        image0,  mask  = load_image_all(motion,  volunteer, directory)
        image = image0*mask[:,:,:,np.newaxis]
        mean[:,:,vv] = np.nanmean(image,axis = (0,1))
        print('Finished volunteer '+str(vv+1)+'/10')

    return mean


def get_tempPhs_net_meanstd(directory,list_vols,motion):
    print('Calculating Net Temporal Phase Std. Deviation for Motion',motion)
    std_mean = get_tempPhs_mean_std(motion, directory,list_vols )
    return std_mean
    


def compute_statistics_motionComp(data,alpha):
    """
    Compute statistics for a given data set
    Input: data [# volunteers, levels of motion compensation]
    Output: hypothesis test results [groups,groups]
    """
    group1 = data[:,0]
    group2 = data[:,1]
    group3 = data[:,2]

    # Test for normality
    tests = [stats.shapiro(group1)[1] <alpha,stats.shapiro(group2)[1] <alpha, stats.shapiro(group3)[1] <alpha]
    if np.sum(tests) > 0:
        normal = 0
    else:
        normal = 1

    #normal = 0 if stats.shapiro(group1)[1] <alpha or stats.shapiro(group2)[1] <alpha or stats.shapiro(group1)[1] <alpha else 1

    # If not normal, use non-parametric test
    if normal ==0:
        result = stats.friedmanchisquare(group1, group2, group3)
        if result[1] < alpha:
            test = sp.posthoc_wilcoxon([group1,group2,group3],p_adjust = 'holm-sidak')
            hypothesis = test
        else:
            hypothesis = np.empty((3,3,))
            hypothesis[:] = np.nan  
            
    # If normal, use parametric test
    elif normal ==1:
        result = stats.f_oneway(group1, group2, group3)
        if result[1] < alpha:
            test = sp.posthoc_ttest([group1,group2,group3],p_adjust = 'holm-sidak')
            hypothesis = test 
        else:
            hypothesis = np.empty((3,3,))
            hypothesis[:] = np.nan  

    return hypothesis

def compute_statistics_slices(data,alpha):
    """
    Compute statistics for a given data set
    Input: data [# volunteers, levels of motion compensation]
    Output: hypothesis test results [groups,groups]
    """
    group1 = data[0,:]
    group2 = data[1,:]
    group3 = data[2,:]


    # Test for normality
    tests = [stats.shapiro(group1)[1] <alpha,stats.shapiro(group2)[1] <alpha, stats.shapiro(group3)[1] <alpha, ]
    if np.sum(tests) > 0:
        normal = 0
    else:
        normal = 1

    #normal = 0 if stats.shapiro(group1)[1] <alpha or stats.shapiro(group2)[1] <alpha or stats.shapiro(group1)[1] <alpha or stats.shapiro(group4)[1] <alpha or \
    #    stats.shapiro(group5)[1] <alpha or stats.shapiro(group6)[1] <alpha else 1

    # If not normal, use non-parametric test
    if normal ==0:
        result = stats.friedmanchisquare(group1, group2, group3)
        if result[1] < alpha:
            test = sp.posthoc_wilcoxon([group1,group2,group3],p_adjust = 'holm-sidak')
            hypothesis = test
        else:
            hypothesis = np.empty((3,3))
            hypothesis[:] = np.nan  
            
    # If normal, use parametric test
    elif normal ==1:
        result = stats.f_oneway(group1, group2, group3)
        if result[1] < alpha:
            test = sp.posthoc_ttest([group1,group2,group3],p_adjust = 'holm-sidak')
            hypothesis = test 
        else:
            hypothesis = np.empty((3,3,))
            hypothesis[:] = np.nan  

    return hypothesis


def load_spatial_map(directory,volunteer,motion,slice,diffusion):
    """
    Load spatial map from file
    Input: filepath
    Output: spatial phase map
    """
    # Setup up filepath
    inpath = os.path.join(directory,'V00'+str(volunteer),'DWI')
    # Load spatial phase map ||phi||
    filename = 'M'+str(motion)+'_slope.npy'
    load_name = os.path.join(inpath,filename)
    spatial_map = np.load(load_name)[:,:,slice,diffusion,:]

    # Load data to check original shape
    test = nib.load(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    data = test.dataobj[:,:,slice,:] #load only the diffusion directions of interest from test variable


    # Get mean and standard deviation maps 
    mean_map = np.nanmean(spatial_map,axis = -1)
    std_map = np.nanstd(spatial_map,axis = -1)

    if data.shape[0] == 100:
        mean_map = np.flip(mean_map,axis = 1)
        std_map = np.flip(std_map,axis = 1)

    #Load magnitude image
    mag = abs(load_data(directory,volunteer,motion,slice,diffusion))

    return mean_map, std_map,  np.nanmean(mag,axis = -1),

def load_data(directory,volunteer,motion,slice,diffusion):
    """
    Load all data from file
    Input: directory, volunteer, motion slice,diffusion,timepoint
    Output: data [complex]
    """
    inpath = os.path.join(directory,'V00'+str(volunteer),'DWI')
    # Load only data for a given slice
    test = nib.load(os.path.join(inpath, 'M'+str(motion)+'_registered.nii'))
    data = test.dataobj[:,:,slice,:] #load only the diffusion directions of interest from test variable
    # Load Bvals and Bvecs
    bvals = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvals')) 
    bvecs = np.loadtxt(os.path.join(inpath, 'M'+str(motion)+'_registered.bvecs'))

    # Sort data
    data1,bvals_sort,bvecs_sort = stacked2sorted(data[:,:,np.newaxis,:],bvals,bvecs.T)

    if data1.shape[0] == 100:
        disp_im1 = np.flip(np.squeeze(data1.transpose(1,0,2,3,4)),axis = 1)
    else:
        disp_im1 = np.squeeze(data1)


    # Get magnitude and phase temporal variation
    disp_im = np.squeeze(disp_im1[:,:,:,:,])[:,:,diffusion,:]

    return disp_im


def  get_spatialPhs_mean_std( directory,list_vols,motion):
    """
    Get net mean spatial phase [mean and standard deviation] across all volunteers for a given motion compensation level
    Input: filepath, list of volunteers, motion compensation 
    Output: net spatial phase mean and standard deviation for a given slice, diffusion direction, and volunteer [3 slices, 4 directions, 10 volunteers]
    """
    mean = np.zeros((3,4,10))
    std = np.zeros((3,4,10))
    for vv in range(10):
        volunteer = list_vols[vv]
        mean_map, std_map, mask  = load_spatial_map_all(directory,volunteer,motion)
        image = mean_map#*mask[:,:,:,np.newaxis]
        mean[:,:,vv] = np.nanmean(image,axis = (0,1))

        image_std = std_map#*mask[:,:,:,np.newaxis]
        std[:,:,vv,3] = np.nanmean(image_std,axis = (0,1))
        print('Finished volunteer '+str(vv+1)+'/10')

    return mean,std

def load_spatial_map_all(directory,volunteer,motion):
    """
    Load spatial map from file
    Input: filepath
    Output: spatial phase map
    """
    # Setup up filepath
    inpath = os.path.join(directory,'V00'+str(volunteer),'DWI')
    # Load spatial phase map ||phi||
    filename = 'M'+str(motion)+'_slope.npy'
    load_name = os.path.join(inpath,filename)
    spatial_map = np.load(load_name)
    
    # Load mask
    mask_LV,affine, voxsize = load_nifti(os.path.join(inpath, 'LV_M'+str(motion)+'.nii'), return_voxsize=True)
    mask_BP,affine, voxsize = load_nifti(os.path.join(inpath, 'BP_M'+str(motion)+'.nii'), return_voxsize=True)
    mask = (mask_LV-mask_BP).astype('float')
    mask[mask==0] = np.nan 

    if mask.shape[0] == 100:
        mask = np.flip(np.squeeze(mask.transpose(1,0,2)),axis = 1)
        spatial_map = np.flip(spatial_map,axis = 1)
    else:
        mask = np.squeeze(mask)

    # Get mean and standard deviation maps 
    mean_map = np.nanmean(spatial_map,axis = -1)
    std_map = np.nanstd(spatial_map,axis = -1)

    return mean_map, std_map, mask,