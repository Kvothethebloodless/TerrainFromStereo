import numpy as np
import glob
import cv2
import time
from matplotlib import pyplot as plt
from Tkinter import *
import pickle

plt.set_cmap('gray')

class stereo_compute():
    def __init__(self):
	self.imgL = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_0/000300.png',0)
	self.imgR = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_1/000300.png',0)
	self.fname_left = sorted(glob.glob('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_0/*.png'))
	self.fname_right = sorted(glob.glob('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/00/image_1/*.png'))
	
	self.window_size = 11
	self.min_disp = -64
	self.num_disp = 192
	self.uniquenessRatio = 1
	self.speckleWindowSize = 50
	self.speckleRange = 2
	self.disp12MaxDiff = 10
	self.P1 = 8*3*self.window_size**2
	self.P2 = 32*3*self.window_size**2	
	self.stereo = cv2.StereoSGBM_create(minDisparity = self.min_disp, numDisparities = self.num_disp,blockSize = self.window_size,uniquenessRatio = self.uniquenessRatio,speckleWindowSize = self.speckleWindowSize,speckleRange=self.speckleRange,disp12MaxDiff = self.disp12MaxDiff,P1 = self.P1,P2 = self.P2)
	self.disp = self.stereo.compute(self.imgL, self.imgR)
	self.fig = plt.figure()
	self.ax = self.fig.add_subplot(211)
	self.ax.set_title("My Title")
	self.im = self.ax.imshow(self.imgL, cmap='gray')# Blank starting image
	
	self.ax2 = self.fig.add_subplot(212)
	self.im2 = self.ax2.imshow(self.disp,vmin=self.disp.min(),vmax=self.disp.max())
	self.fig.show()
	self.fig.canvas.draw()	

	
	
    def compute(self):
	#print('compute called \n')
	#print(str(np.sum(self.disp == self.stereo.compute(self.imgL, self.imgR))))
	self.disp = self.stereo.compute(self.imgL, self.imgR)
	self.plot_disparity()
	
    def plot_disparity(self):
	#print('plot called \n')
	self.im2.set_data(self.disp)
	self.fig.canvas.draw()
	
	
str_obj = stereo_compute()
 
def cmd_windowsize(value):
    #print('windsize'+str((int(value)*2)+1))
    str_obj.stereo.setBlockSize((int(value)*2)+1)
    str_obj.compute()
       
def cmd_mindisp(value):
    #print('mindisparity'+str(value))
    str_obj.stereo.setMinDisparity(int(value))
    str_obj.compute()
def cmd_numdisp(value):
    value = int(value)*16
    str_obj.stereo.setNumDisparities(value)
    str_obj.compute()
        
def cmd_ur(value):
    value = int(value)
    str_obj.stereo.setUniquenessRatio(value)
    str_obj.compute()
    
def cmd_spwindowsize(value):
    value = int(value)
    str_obj.stereo.setSpeckleWindowSize(value)
    str_obj.compute()
    
def cmd_sprange(value):
    value = int(value)
    str_obj.stereo.setSpeckleRange(value)
    str_obj.compute()
    
def cmd_dispmaxdiff(value):
    value = int(value)
    str_obj.stereo.setDisp12MaxDiff(value)
    str_obj.compute()
    
def cmd_p1(value):
    value = int(value)
    str_obj.stereo.setP1(value)
    str_obj.compute()
    
def cmd_p2(value):
    value = int(value)
    str_obj.stereo.setP2(value)
    str_obj.compute()

def writetofile():
    fname = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/gen_data/' + s_filename.get()
    f = open(fname,'w')
    str_obj.stereo.save(fname)
    print "saved as" + fname
   

root = Tk()   
s_windowsize = Scale(root, orient = 'horizontal', from_ =5, to =60, command = cmd_windowsize, label = 'windowsize',length = 1500)
s_mindisp = Scale(root, orient = 'horizontal', from_ =-300, to= 300, command = cmd_mindisp, label = 'minimum disparity',length = 1500)
s_numdisp = Scale(root, orient = 'horizontal', from_= 1, to =30, command = cmd_numdisp, label= 'Disparity Range as multiple of 16',length = 1500)
s_ur = Scale(root, orient = 'horizontal', from_= 1, to =100, command = cmd_ur, label= 'Unique ratio' , length = 1500)
s_spwindowsize = Scale(root, orient = 'horizontal', from_= 1, to =100, command = cmd_spwindowsize, label= 'Speckle Window Size',length = 1500)
s_sprange = Scale(root, orient = 'horizontal', from_= 1, to =300, command = cmd_sprange, label= 'Speckle Range',length = 1500)
s_dmaxdiff = Scale(root, orient = 'horizontal', from_= 1, to =50, command = cmd_dispmaxdiff, label= 'Disp 12 Max Diff',length = 1500)
s_p1 = Scale(root, orient = 'horizontal', from_= 1, to =30000, command = cmd_p1, label= 'P1',length = 1500)
s_p2 = Scale(root, orient = 'horizontal', from_= 1, to =100000, command = cmd_p2, label= 'P2',length = 1500)
s_filename = Entry(root)
s_filename.insert(0,'tes')
b = Button(root, text="SAVE", command=writetofile)


s_windowsize.pack()
s_mindisp.pack()
s_numdisp.pack()
s_ur.pack()
s_spwindowsize.pack()
s_sprange.pack()
s_dmaxdiff.pack()
s_p1.pack()
s_p2.pack()
s_filename.pack()
b.pack()

root.mainloop()
#stereo = cv2.StereoBM()


def nothing(x):
    pass


