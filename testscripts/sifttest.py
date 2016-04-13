import cv2
import numpy as np
import itertools
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt


img = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_0/000100.png',0)
template = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_1/000101.png',0)



def getfeatures(img,template,no_points,onsides):
    
    
    detector = cv2.xfeatures2d.SIFT_create()
    descriptor = detector
   
    ####
    #
    #
    #Finding Features
    #
    #
    ####
    
    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)
    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)
    
    #kdt=KDTree(sd,leaf_size=30,metric='euclidean')
    #idx, dist = flann.knnSearch(td, 1, params={})
    
    
    ####
    #
    #
    #Matching Features in i1 with those in i2
    #
    #
    #####
    
    nn1 = NearestNeighbors(n_neighbours=1,algorithm='brute',metric='euclidean')
    nn1.fit(sd)
    #dist, idx = kdt.query(td, k=1)
    dist, idx = nn1.kneighbors(td,1,return_distance=True)
    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    
   
   

   
    distance = .020

    #skp_final = []
    #if onsides:
        #for i, dis in itertools.izip(idx, dist):
            #if dis < distance:
                #skp_final.append(skp[i])
            #else:
                #break
    
    
    
    
    
    #kdt2=KDTree(td,leaf_size=30,metric='euclidean')
    #dist, idx = kdt2.query(sd, k=1)
    ####### Matching features in i2 with i1.

    nn2 = NearestNeighbors(n_neighbours=1,algorithm='brute',metric='euclidean')
    nn2.fit(td)
    dist, idx = nn2.kneighbors(sd,1,return_distance=True)
    
    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    
    #### Distance thresholding to find the best matches only.
    #for i, dis in itertools.izip(idx, dist):
        #if dis < distance:
            #tkp_final.append(tkp[i])
        #else:
            #break
    



    #### Code to show matched features side by side.
    #h1, w1 = img.shape[:2]
    #h2, w2 = template.shape[:2]
    #nWidth = w1+w2
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

    #nHeight = max(h1, h2)
    #hdif = (h1-h2)/2
    #newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    #newimg[hdif:hdif+h2, :w2] = template
    #newimg[:h1, w2:w1+w2] = img
    
    #tkp = tkp_final
    #skp = skp_final
    
    pts_left = []
    pts_right = []
    
    for i in range(min(len(tkp), len(skp))):
        pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
        pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
        
        #cv2.line(newimg, pt_a, pt_b, (255-10*i, 10*i, 5*i),thickness=3)
        
        pts_1.append(pt_a)
        pts_2.append(pt_b)
        
    return (pt_a, pt_b)







