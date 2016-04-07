import numpy as np
import glob
import cv2
import time
import yaml
from matplotlib import pyplot as plt
import pcl as pcl

plt.ion()

def readPmatrix():
    f = open('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/calib.txt')
    txt = f.readlines()
    p1_txt = txt[0]
    p2_txt = txt[1]
    p1_txt2 = p1_txt.split(' ')
    p1 = p1_txt2[1::]
    p1[-1] = p1[-1][0:-1]
    p1 = map(float,p1)
    p1_mat = np.reshape(p1,(3,4))


    p2_txt2 = p2_txt.split(' ')
    p2 = p2_txt2[1::]
    p2[-1] = p2[-1][0:-1]
    p2 = map(float,p2)
    p2_mat = np.reshape(p2,(3,4))

    return(p1_mat, p2_mat)



def loadfromtuner(d_obj,filename):
    fname = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/gen_data/tes' 
    attr_list = ['numDisparities','preFilterSize','speckleRange','uniquenessRatio', 'blockSize', 'minDisparity','speckleWindowSize', 'textureThreshold', 'preFilterCap','disp12MaxDiff', 'preFilterType','P1','P2']
    att_dict = yaml.load(open(fname,'r'))['my_object']
    d_obj.setNumDisparities(att_dict[attr_list[0]])
    #d_obj.setPreFilterSize(att_dict[attr_list[1]])
    d_obj.setSpeckleRange(att_dict[attr_list[2]])
    d_obj.setUniquenessRatio(att_dict[attr_list[3]])
    d_obj.setBlockSize(att_dict[attr_list[4]])
    d_obj.setMinDisparity(att_dict[attr_list[5]])
    d_obj.setSpeckleWindowSize(att_dict[attr_list[6]])
    #d_obj.setTextureThreshold(att_dict[attr_list[7]])
    d_obj.setPreFilterCap(att_dict[attr_list[8]])
    d_obj.setDisp12MaxDiff(att_dict[attr_list[9]])
    d_obj.setP1(att_dict[attr_list[11]])
    d_obj.setP2(att_dict[attr_list[12]])
    #d_obj.setPreFilterType(att_dict[attr_list[10]])
    return d_obj

imgL = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_0/000300.png',0)
imgR = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_1/000300.png',0)

fname_left = sorted(glob.glob('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_0/*.png'))
fname_right = sorted(glob.glob('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_1/*.png'))


window_size = 11
min_disp = 0
num_disp = 160
#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=window_size)

stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
numDisparities = num_disp,
blockSize = window_size,
uniquenessRatio = 1,
speckleWindowSize = 50,
speckleRange = 2,
disp12MaxDiff = 10,
P1 = 8*3*window_size**2,
P2 = 32*3*window_size**2)


stereo = loadfromtuner(stereo, 'tes')

def kitti_test():

    #plt.set_cmap('gray')
    fig = plt.figure( 1 )
    ax = fig.add_subplot( 111 )
    ax.set_title("My Title")

    im = ax.imshow(cv2.imread(fname_left[0]))# Blank starting image
    fig.show()
    fig.canvas.draw()


    for i in range(len(fname_left)):
        img_l = cv2.cvtColor(cv2.imread(fname_left[i]), cv2.COLOR_BGR2GRAY)
        img_r = cv2.cvtColor(cv2.imread(fname_right[i]), cv2.COLOR_BGR2GRAY)
        disp = stereo.compute(img_l, img_r)
        im.set_data(disp)
        fig.canvas.draw()

        # cv2.imshow('frame',disp)
        # cv2.waitKey(10)
        print('fno:' + str(i))

def testfromkitti():
    disp = stereo.compute(imgL,imgR)
    plt.imshow(disp)
    plt.show()

def disptodepth(stereo_obj,imgL,imgR,disp):
    mindisp = stereo_obj.getMinDisparity()
    maxdisp = stereo_obj.getNumDisparities()
    (p1,p2) = readPmatrix()
    bf = 0-p2[0,-1]
    print(bf)
    disp = disp - disp.min() #making all positive values
    disp = np.copy(disp/disp.max()) #Converting to disparity between one and zero
    disp = disp + 1
    
    depth = float(bf)/disp; #Converting to depth vales
    #depth = depth*100
    print('exiting disptodepth') 
    return depth

#def depthtopcl(depth,stereo_obj,imgL,disp):
    
    #(h,l) = np.shape(imgL)
    #coordinates = np.mgrid[0:h,0:l,0:1].T.reshape(-1,3) #creating a 3d point array for storing x-y-z locations. as index progress, column changes first.
    #color = np.empty_like(coordinates)
    #print('Conversion started')
    ##colorimg = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    #for i in range(np.shape(coordinates)[0]/10):
        #a = coordinates[i,0]
        #b = coordinates[i,1]
        #coordinates[:,2] = depth[a,b]
        #color[i] = np.array([imgL[a,b]]*3)
        #print(i)
    #np.save('co',coordinates)
    #np.save('co',color)
    #print('conversion completed')
    #pcl_obj = pcl.PointCloud(coordinates, color)
    #pcl_obj.write_ply('test.ply')

def depthtopcl2(depth,stereo_obj,imgL,disp):
    print('Conversion started')
    (h,l) = np.shape(imgL)
    coordinates = np.mgrid[0:h,0:l].T.reshape(-1,2) #creating a 2d point array for storing x-y locations. as index progress, column changes first.
    deptharray = (depth).T.reshape(-1,1) 
    deptharray = deptharray - deptharray.min() + 1 #Converting depth to positive values only.
    
    cors_mat = np.hstack((coordinates, deptharray))
    print(cors_mat.shape)
     
    
    #colorimg = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAYnp.save('co',coordinates)
    color_mat = np.copy(imgL)
    color_mat = color_mat.T.reshape(-1) #array of colors according to co-ordinate indexing
    color_mat = np.vstack((color_mat,color_mat,color_mat))
    color_mat = color_mat.T.reshape(-1,3)
    
        
    
    np.save('cors', cors_mat)
    np.save('co',color_mat)
    print('conversion completed')
    pcl_obj = pcl.PointCloud(cors_mat, color_mat)
    pcl_obj.filter_infinity()
    pcl_obj.write_ply('test.ply')
    
            
        

testfromkitti()
disp = stereo.compute(imgL,imgR)
depth = disptodepth(stereo, imgL, imgR, disp)
depthtopcl2(depth, stereo, imgL, disp)
# disparity = stereo.compute(imgL,imgR)
# plt.figure()
# plt.set_cmap('gray')
# plt.subplot(311), plt.imshow(imgL)
# plt.subplot(312), plt.imshow(imgR)
# plt.subplot(313), plt.imshow(disparity,'gray')
# plt.show()
