import numpy as np
from jacobian_ur_16_eef import getJacobianUr16_base_link_inertiaUr16_wrist_3_link as getJacobianManipulator
from jacobian_platform import getJacobianPlatform
import numdifftools as nd
from math import pi


def manipulability(q):
    jacob = getJacobianManipulator(q)
    m = np.linalg.det(jacob @ jacob.T)
    return m


def calc_gradient(func, q):
    return nd.Gradient(func)(q)


def calcPlatformOptimal(j_m, j_p, dH):
    optMat = np.zeros((8, 8))
    optMat[:, 0:6] = -(np.linalg.inv(j_m) @ j_p).T
    optMat[:, 6:] = np.eye(2)
    u_p = optMat @ dH
    return u_p


if __name__ == "__main__":
    q = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    theta = pi / 2
    gradient = calc_gradient(manipulability, q)
    j_m = getJacobianManipulator(q)
    j_p = getJacobianPlatform(theta)

    optMat = -(np.linalg.inv(j_m) @ j_p).T
    u_p = optMat @ gradient
    print(u_p)
