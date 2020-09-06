import math
import pprint
import scipy.misc
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from video_utils import draw_stroke
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import random

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def save_images(images, grid_size, image_path, invert=True, channels=3,angle=None):
    if invert:
        images = inverse_transform(images)
    return imsave(images, grid_size, image_path, channels,angle)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, grid_size, path, channels=1,angle=None):
    h, w = int(images.shape[1]), int(images.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))

    for idx, image in enumerate(images):
        i = int(idx % grid_size[1])
        j = int(idx // grid_size[1])

        if channels == 1:
            img[j*h:j*h+h, i*w:i*w+w, 0] = image 
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
    # Flatten third dimension if only grayscale image (necessary for scipy image save method).
    if channels == 1:
        img = img.reshape(img.shape[0:2])
    
    return scipy.misc.imsave(path, img)

def save_img(images,ref,images_gen,grid,path,boundary=True,channels=1):
    images = inverse_transform(images)
    images_gen = inverse_transform(images_gen)
    ref = inverse_transform(ref)

    h,w = (images.shape[1],images.shape[2])
    img = np.zeros((int(h*grid[0]*2),int(w*grid[1])))

    boundary_value = 1
    if boundary:
        ref[:,0,:,:] = images[:,0,:,:] = images_gen[:,0,:,:] = boundary_value
        ref[:,:,0,:] = images[:,:,0,:] = images_gen[:,:,0,:] = boundary_value
        ref[:,-1,:,:] = images[:,-1,:,:] = images_gen[:,-1,:,:] = boundary_value
        ref[:,:,-1,:] = images[:,:,-1,:] = images_gen[:,:,-1,:] = boundary_value

    for i in xrange(grid[0]):
        img[2*i*h:(2*i+1)*h,0*w:(0+1)*w] = images[i,:,:,0*channels]

        traj = np.minimum(images[i,:,:,1],images[i,:,:,2])
        for ii in xrange(3,grid[1]-1):
            traj = np.minimum(traj,images[i,:,:,ii])

        img[2*i*h:(2*i+1)*h,w:2*w] = traj

        for j in xrange(2,grid[1]):
            img[2*i*h:(2*i+1)*h,j*w:(j+1)*w] = ref[i,:,:,(j-2)*channels]
            img[(2*i+1)*h:(2*i+2)*h,j*w:(j+1)*w] = images_gen[i,:,:,(j-2)*channels]

    return scipy.misc.imsave(path, img)

def save_img_video(ref,images_gen,grid,path,boundary=True,channels=3):
    
    h = grid[2]
    img = np.zeros((int(h*grid[0]*2),int(h*grid[1]),channels))

    for i in xrange(grid[0]):
        for j in xrange(0,grid[1]):
            images_gen_tmp = inverse_transform(images_gen[j][i,:,:,:])
            ref_tmp = inverse_transform(ref[j][i,:,:,:])
            img[2*i*h:(2*i+1)*h,j*h:(j+1)*h,:] = images_gen_tmp
            img[(2*i+1)*h:(2*i+2)*h,j*h:(j+1)*h,:] = ref_tmp
            
    return scipy.misc.imsave(path, img)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_img(frame_path,input_size,ch=3):
    if ch ==3:
        img = scipy.misc.imread(frame_path.strip())
        img = scipy.misc.imresize(img,[input_size,input_size])
    if ch ==1:
        img = scipy.misc.imread(frame_path.strip())
        img = scipy.misc.imresize(img,[input_size,input_size])
        img = np.expand_dims(img,2)
        a = np.zeros_like(img)
        a = img
        img = np.concatenate((img,img),axis=2)
        img = np.concatenate((img,a),axis=2)

    return img*2./255 - 1 

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length



def get_ms_img_robot(point_list,input_size,c):
    img = np.ones((input_size,input_size,1))*255
    # c=0 if starts from the first frame
    for x in point_list:
        img[x[0],x[1],:] = c*8
        c = c+1

    return img*2./255 - 1



def get_local_mask_robot(ms_t,input_size,local_size,ch=3):

    s = int(local_size/2)
    c = int(ms_t[0][0])+32
    r = int(ms_t[0][1])+32

    mask_img = np.zeros((input_size,input_size,ch),dtype=np.int16)

    offset = random.randint(-64,64)
    c = c+offset

    if r-s<0:
        r = s
    if c-s<0:
        c = s
    if r+s>input_size:
        r = input_size-s
    if c+s>input_size:
        c = input_size-s
        
    mask_img[r-s:r+s,c-s:c+s,:] = 1

    ref = np.ones_like(mask_img,dtype=np.int16)

    return mask_img==ref

def inverse_image_new(img):
    img = (img + 0.5) * 255.
    img[img > 255] = 255
    img[img < 0] = 0
    img = img[..., ::-1] # bgr to rgb
    return img

