import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


def img2vid(img_path, vid_path, fps=30):
    """A function which converts a directory of images into a video at the given framerate."""
    dir = os.listdir(img_path)
    dir.sort()
    img_array = []
    set_size = 1
    for filename in dir:
        img = cv2.imread(img_path + '/' + filename)
        height, width, layers = img.shape
        if set_size == 1:
            size = (width, height)
            set_size = 0
            
        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
        img_array.append(img)
    out = cv2.VideoWriter(vid_path, 0x7634706d, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    
def stich_video(img_path1,img_path2,vid_path,fps=15):
    dir = os.listdir(img_path1)
    dir.sort()
    img_array = []
    count = 0
    for filename in dir:
        if count % 3 == 0:
            img_path = img_path1
        else:
            img_path = img_path2
        img = cv2.imread(img_path + '/' + filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        count += 1
    out = cv2.VideoWriter(vid_path, 0x7634706d, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
def depth2color(img_path, out_path):
    
    dir = os.listdir(img_path)
    dir.sort()    
    dep_array = []
    for filename in dir:
        
        depth = depth_read(img_path + '/' + filename)
        c_depth = depth_colorize(depth)
        # dep_array.append(c_depth)
        depth_write(out_path + '/' + filename, c_depth)
     
def depth_write(filename, img):
    img[img < 0] = 0  # negative depth is like 0 depth
    img = img * 256
    if np.max(img) >= 2 ** 16:
        print('Warning: {} pixels in {} have depth >= 2**16 (max is: {}).Truncating before saving.'.format(img[img >= 2**16].shape[0], "/".join(filename.split('/')[-5:]), np.max(img)))
        img = np.minimum(img, 2 ** 16 - 1)

    img = img.astype('uint16')
    cv2.imwrite(filename, img)

cmap = plt.cm.RdBu
# np.seterr(divide='ignore', invalid='ignore')  # so the "RuntimeWarning: invalid value encountered in true_divide" in depth_colorize function will stop showing

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')

"""
Loads depth map D from png file and returns it as a numpy array,
"""
def depth_read(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    if np.max(depth_png) > 65536:  # not relevant for debug (when NNs don't predict good depth), leading to 0.9m in all of the image, resulting this error OR when we insert black image to the NN
        print("warning: max depth {} in while reading image{} in depth_read".format(np.max(depth_png), filename))

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def stich_imgs(img_path1,img_path2,out_path):
    """Take two directories of images and combine them into a single directory with half the images from each directory."""
    imgs = os.listdir(img_path1)
    for img in imgs:
        if int(Path(img_path1 + '/' + img).stem) % 4 == 0:
            shutil.copyfile(img_path1 + '/' + img, out_path + '/' + img)
        else:
            shutil.copyfile(img_path2 + '/' + img, out_path + '/' + img)
    
if __name__ == "__main__":
    # img2vid("/home/tony/github/Adaptive Lidar/Adaptive-LiDAR-Sampling/stich", "stich4.mp4", 15)
    # depth2color("/home/tony/github/Adaptive Lidar/outputs/var_final_NN/var.test.mode=dense.input=d.resnet18.time=2022-12-15@10-51/dense_depth_images/data_depth_velodyne/test/2011_09_26_drive_0002_sync/proj_depth/velodyne_raw/image_02","depth")
    img2vid("gt", "color_gt.mp4", 15)
    # stich_video("/home/tony/github/Adaptive Lidar/Adaptive-LiDAR-Sampling/color_raw","/home/tony/github/Adaptive Lidar/Adaptive-LiDAR-Sampling/color_adaptive","stich2.mp4",fps=15)
    # stich_imgs('/home/tony/github/Adaptive Lidar/Adaptive-LiDAR-Sampling/color_raw','/home/tony/github/Adaptive Lidar/Adaptive-LiDAR-Sampling/color_adaptive','stich')
    
