import numpy as np
import cv2
import matplotlib as plt


def readPmatrix():
	f = open('datasets/video_seq/dataset/sequences/00/calib.txt')
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


