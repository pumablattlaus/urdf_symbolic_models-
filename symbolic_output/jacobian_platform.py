#!/usr/bin/python3

from math import sin, cos, pi
import numpy as np

def getJacobianPlatform(theta):
    J = np.zeros((6, 2))
    J[0,0]=cos(theta)
    J[1,0]=sin(theta)
    J[5,1]=1
    return J

if __name__ == "__main__":
    theta = pi/2
    print(getJacobianPlatform(theta))