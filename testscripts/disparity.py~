import numpy as np
import glob
import cv2
import time
import yaml
from matplotlib import pyplot as plt

plt.set_cmap('gray')
def loadfromtuner(d_obj,filename):
	fname = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/gen_data/'+str(filename) 
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



imgL = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_0/000100.png',0)
imgR = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_1/000100.png',0)

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

def kitti_test(stereo):
	
	#plt.set_cmap('gray')
	fig = plt.figure( 1 )
	ax = fig.add_subplot( 111 )
	ax.set_title("My Title")
	img_l = cv2.cvtColor(cv2.imread(fname_left[0]), cv2.COLOR_BGR2GRAY)
	img_r = cv2.cvtColor(cv2.imread(fname_right[0]), cv2.COLOR_BGR2GRAY)
	global disp 
	disp = stereo.compute(img_l, img_r)
	disp = disp - disp.min()
	disp = (disp/16).astype('uint8')	
	im = ax.imshow(disp, vmin=disp.min(),vmax=disp.max())# Blank starting image
	fig.show()
	fig.canvas.draw()
	stereo = loadfromtuner(stereo,'smoothandgood')
	
	
	for i in range(len(fname_left)):
		img_l = cv2.cvtColor(cv2.imread(fname_left[i]), cv2.COLOR_BGR2GRAY)
		img_r = cv2.cvtColor(cv2.imread(fname_right[i]), cv2.COLOR_BGR2GRAY)
		disp = stereo.compute(img_l, img_r)
		disp = disp - disp.min() #Converting to all positive values for succeding conversion to unit8
		disp = (disp/16).astype('uint8') #Converting to unit8 for histogram equalization
		im.set_data(cv2.equalizeHist(disp))  #Histogram equalization
		fig.canvas.draw()
	
		# cv2.imshow('frame',disp)
		# cv2.waitKey(10)
		print('fno:' + str(i))
	#return disp

def testfromkitti():
	
	disp = stereo.compute(imgL,imgR)
	plt.imshow(disp)
	plt.show()


	

# disparity = stereo.compute(imgL,imgR)
# plt.figure()
# plt.set_cmap('gray')
# plt.subplot(311), plt.imshow(imgL)
# plt.subplot(312), plt.imshow(imgR)
# plt.subplot(313), plt.imshow(disparity,'gray')
# plt.show()
