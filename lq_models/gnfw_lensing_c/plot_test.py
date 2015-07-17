import numpy as np
from pylab import *
from mycosmology import *
import alens_arr as aa
import subprocess as sp
import sys
#--------------------------------------------------------------------
#cmd = '/home/nan/.local/lib/python2.7/site-packages/gnfw_lensing_c/a.out'
x_in = 0.3
cmd = './a.out '+str(x_in)
print cmd
pixels_tmp=sp.check_output(cmd,shell=True)
print pixels_tmp.split()

#x1,y1 = np.loadtxt("lq1.dat",usecols = (0,1),unpack=True)
#x2,y2 = np.loadtxt("lq2.dat",usecols = (0,1),unpack=True)
#x3,y3 = np.loadtxt("lq3.dat",usecols = (0,1),unpack=True)
#plot(x1,y1,'r-')
#plot(x2,y2,'r--')
#plot(x3,y3,'r.')
#grid()
#show()
