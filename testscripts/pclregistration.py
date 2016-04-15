import numpy as np
import matplotlib.pyplot as plt
import cv2
from utilites import psosolver as psol
import commonpoints as cps
from utilites import helper as hp


(tp1, tp2) = cps.neighbour3dpoints('00',1,2,10,20)
no_points = np.shape(tp1)[0]
searchspaceboundaries = np.hstack((-1*np.ones(1,10),np.ones(1,10))
transformation_matrix = psol.pso(self, score_func, 63, 12, 1.6319, .6239, dt, np.ones(12),
                 searchspaceboundaries)