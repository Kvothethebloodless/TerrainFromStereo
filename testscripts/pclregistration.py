import numpy as np
import matplotlib.pyplot as plt
import cv2
from utilites import psosolver as psosolver
import commonpoints as cps
from utilites import helper as hp
from utilites import statelogger as stlg
import pcl 




def rotationmatrix(alpha,beta,gamma):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    rx = np.matrix(np.array([[1, 0, 0],[0, ca, -sa],[0, sa, ca]]))
    ry = np.matrix(np.array([[cb, 0, sb],[0,1,0],[-sb, 0, cb]]))
    rz = np.matrix(np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]]))
    return np.dot(rx,np.dot(ry,rz))

def transformationmatrix(rot_params):
    (alpha,beta,gamma,tx,ty,tz) = rot_params
    rotmatrix= rotationmatrix(alpha, beta, gamma)
    T = np.array([tx,ty,tz]).T.reshape(3,1)
    return np.hstack((rotmatrix,T))

def score_func(rot_params):
    t_vector = transformationmatrix(rot_params).reshape(12,1)
    transformed_points = transform_points(t_vector,tp2)
    diff = tp1-transformed_points
    return(np.sum(np.power(diff,2)))


    
(tp1, tp2) = cps.neighbour3dpoints('00',1,2,100,100)
no_points = np.shape(tp1)[0]
prepoints = np.hstack((tp2,np.ones((no_points,1))))
pi = np.pi
searchspaceboundaries = np.vstack((np.array([-pi,-pi,-pi,-2,-2,-2]),-1*np.array([-pi,-pi,-pi,-2,-2,-2])))




def findtransformation():
    
    
    psol = psosolver.PSO(score_func, 63, 6, 1.6319, .6239, 1, np.ones(6),
                     searchspaceboundaries)
    
    score_func(np.random.randint(1,10,6))
    
    
    point = np.ones(6)
    rec_obj_pso = stlg.statelogger('gen_data/pclregistraion','pclreglog',psol.curr_score,psol.globalmin,psol.current_pos)
            
    for i in range(100):
        psol.update_pos()
        psol.update_currscores()
        psol.update_selfmin()
        psol.update_globalmin()
        psol.update_velocities()
        #(centroid, spread) = psol.calc_swarm_props()
    
        #print('\n\n\n')
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \n')
        #print('PARTICLE SWARM OPTIMIZER REPORT')
    
    
        #psol.report()
    
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
        ## print(psol.globalmin,psol.curr_score)
        rec_obj_pso.add_state(psol.curr_score,psol.globalmin,psol.current_pos)
    
        #print('\n\n\n')
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  \n')
       # print('GRADIENT DESCENT SOLVER REPORT')
       # curr_score = currscript_score(point)
       # point = gdssol.get_descented_point(point)
       # descented_score = currscript_score(point)
       # deltascore = descented_score - curr_score
       # curr_score = descented_score
       # gdssol.report()
       # rec_obj_gds.add_state(gdssol.score_func(point))
    return (transformationmatrix(psol.globalminlocation),psol)


def transform_points(t_vector,cords):
    no_points = len(cords)
    prepoints = np.hstack((cords,np.ones((no_points,1))))
    transform_matrix = t_vector.reshape(3,4)
    transformed_points = np.dot(transform_matrix,prepoints.T).T
    return transformed_points


def findBestTransform():
    n_pso_iters = 15
    results = []
    for i in range(n_pso_iters):
        (t_matrix,psol) = findtransformation()
        score = psol.globalmin
        results.append((score,t_matrix))
        
    
    results_sorted = sorted(results, cmp=None, key=lambda sol:sol[0], reverse=False)
    print(results_sorted)
    return (results_sorted[0][1])
    
    

    
def stitchPointClouds(seqno,f1,f2):
    cords1,cols1 = hp.readfromply(seqno,f1)
    cords2,cols2 = hp.readfromply(seqno,f2)
    t_matrix = findBestTransform()
    cords2_transformed = transform_points(t_matrix.reshape((12,1)), cords2)
    newcords = np.vstack((cords1,cords2_transformed))
    newcolors = np.vstack((cols1,cols2))
    print('length of the new stack is : ' + str(len(newcords)))
    pcl_obj = pcl.PointCloud(newcords, newcolors)
    pcl_obj.filter_sky()
    
    pcl_obj.write_ply('stitched.ply')
    
    
def stitchpointclouds_arbit(t_vector):
    seqno = '00'
    f1 = 1
    f2 = 2
    cords1,cols1 = hp.readfromply(seqno,f1)
    cords2,cols2 = hp.readfromply(seqno,f2)
    t_matrix = t_vector.reshape(3,4)
    cords2_transformed = transform_points(t_matrix.reshape((12,1)), cords2)
    newcords = np.vstack((cords1,cords2_transformed))
    newcolors = np.vstack((cols1,cols2))
    print('length of the new stack is : ' + str(len(newcords)))
    pcl_obj = pcl.PointCloud(newcords, newcolors)
    pcl_obj.filter_sky()
    pcl_obj.write_ply('stitched.ply')
    