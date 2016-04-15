import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2



def loadimage_kitti(seqno,side,image_num, rgb):
    foldername = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/'
    if int(seqno) > 21:
        raise Exception('Sequence not available. ')
    
    seqdir= foldername+seqno+'/'
    #print(seqdir)
    
    if side == 'l':
        sidedir = seqdir + 'image_0/'
    elif side == 'r':
        sidedir = seqdir + 'image_1/'
    else:
        raise Exception('Exception. Enter valid side (l or r)')
    
    #print(sidedir)
    fname = sorted(glob.glob(sidedir+'*.png'))
    
    if image_num>len(fname):
        raise Exception('Image number exceeded limit in the directory')
    
    else:
        imgpath = fname[image_num]
        print('Reading image from '  + str(imgpath))
        img = cv2.imread(imgpath)
        if rgb:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def loadstereopair_kitti(seqno,image_num, rgb):
    imgL = loadimage_kitti(seqno, 'l', image_num, rgb)
    imgR = loadimage_kitti(seqno, 'r', image_num, rgb)
    return (imgL, imgR)

