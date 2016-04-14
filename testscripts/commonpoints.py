import cv2
import numpy as np
import itertools
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
import utilites.helper as hp

img = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_0/000100.png',0)
template = cv2.imread('/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/datasets/video_seq/dataset/sequences/02/image_1/000101.png',0)



def featurepoint_toworldtransform(points,(h,l),cors):
    
    points = np.array(points, dtype='int')
    points[:,1] = h-1 - points[:,1] #Doing this as SIFT already gives in x,yform
    flipped_points = points
       
    #cors = np.load('cors.npy')
    index =  l*flipped_points[:,1] + flipped_points[:,0]
    print(index)
    return cors[index]
    
    


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
    
    indices1 = range(len(dist)) 
    indices1.sort(key=lambda i: dist[i]) #Getting the order of distances among tk<->sk pair. indices will contain index of the ordered pairs arranged according
    #to the ascending order of distance between them
    
    dist1 = [dist[i] for i in indices1]
    idx1 = [idx[i] for i in indices1] #Will contain the index of the sk descriptor corresponding to the tk descriptor in the arranged indices1 vector.
    
   
   

   
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
    indices2 = range(len(dist))
    indices2.sort(key=lambda i: dist[i]) #Refer to step 1 comments above.
    dist2 = [dist[i] for i in indices2]
    idx2 = [idx[i] for i in indices2]
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
    
    pts_1 = []
    pts_2 = []
    
    for i in range(min(no_points,len(skp),len(tkp))):
        #pt_a = (int(tkp[i].pt[0]),int(tkp[i].pt[1]))
        #pt_b = (int(skp[i].pt[0]),int(skp[i].pt[1]))
        
        pt_a = tkp[indices1[i]].pt  #Retrieveing tk corresponding to lowest distances
        pt_b = skp[idx1[i]].pt #Retreieveing the correspoding sk to tk.
        
        #cv2.line(newimg, pt_a, pt_b, (255-10*i, 10*i, 5*i),thickness=3)
        
        pts_1.append(pt_a)
        pts_2.append(pt_b)
        
    return (pts_1, pts_2)



def neighbour3dpoints(seqno,f1,f2,no_sets,pointsperset):    
    pcl1name = 'seq'+seqno+'frame'+str(f1)
    pcl2name = 'seq'+seqno+'frame'+str(f2)
    path1 = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/gen_data/coordinates/'+ str(pcl1name)+'.npy'
    path2 = '/home/manish/Awesomestuff/Subjects/IVP/Project_stereo/gen_data/coordinates/'+ str(pcl2name)+'.npy'
    cords1 = np.load(path1)
    cords2 = np.load(path2)
    i1 = hp.loadimage_kitti(seqno,'l',f1,0)
    i2 = hp.loadimage_kitti(seqno,'l',f2,0)
    (h,l) = i1.shape
    (pts_1,pts_2) = getfeatures(img, template, no_sets, 0)
    pts3d_1 = featurepoint_toworldtransform(pts_1, (h,l), cords1)
    pts3d_2 = featurepoint_toworldtransform(pts_2, (h,l), cords2)
    
    mask1_1 = np.abs(pts3d_1[:,2])<100;
    mask1_2 = pts3d_1[:,2]>0
    mask1 = np.logical_and(mask1_1,mask1_2)
    
    mask2_1 = np.abs(pts3d_2[:,2])<100;
    mask2_2 = pts3d_2[:,2]>0
    mask2 = np.logical_and(mask2_1,mask2_2)
    
    mask = np.logical_and(mask1,mask2)
    
    pts3d_1 = pts3d_1[mask]
    pts3d_2 = pts3d_2[mask]
    
    n_keypoints = len(pts3d_1)
    print('Total of ' + str(n_keypoints) + ' keypoints are found')
    
    kdt1=KDTree(cords1,leaf_size=30,metric='euclidean')
    dist1, idx1 = kdt1.query(pts3d_1, k=pointsperset, return_distance=True) #Gives in sorted order.
    
    pset1 = []
    
    n_sets = min(n_keypoints,no_sets) #Checking if we have given number of keypoint matches as the sets or not.
    print('Total of ' + str(n_sets)+ ' sets are found')
    for i in range(n_sets):
        pset1.append(pts3d_1[i])
        for j in range(pointsperset):
            pset1.append(cords1[idx1[i][j]])
    pset1 = np.array(pset1)
    
    kdt2 = KDTree(cords2, leaf_size=30, metric='euclidean')
    dist2, idx2 = kdt2.query(pts3d_2, k=pointsperset, return_distance= True)
    
    pset2 = []
    
    for i in range(n_sets):
        pset2.append(pts3d_2[i])
        for j in range(pointsperset):
            pset2.append(cords2[idx2[i][j]])
    pset2 = np.array(pset2)    
    
    return(pset1,pset2)
    
    
    
    
#neighbour3dpoints('00', 1,2, 10, 20)

    
    


