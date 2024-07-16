import numpy as np


def rot2D(ang):
    """
    Return the 2D rotation matrix of given angle in degrees
    """
    alpha = np.pi * ang / 180
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])


def princ_angle(ang):
    """
    Return the principal angle in the 0° to 180° quadrant as PA is always
    defined at p/m 180°.
    """
    if not isinstance(ang, np.ndarray):
        A = np.array([ang])
    else:
        A = np.array(ang)
    while np.any(A < 0.0):
        A[A < 0.0] = A[A < 0.0] + 360.0
    while np.any(A >= 180.0):
        A[A >= 180.0] = A[A >= 180.0] - 180.0
    if type(ang) is type(A):
        return A
    else:
        return A[0]


def sci_not(v, err, rnd=1, out=str):
    """
    Return the scientifque error notation as a string.
    """
    power = -int(("%E" % v)[-3:]) + 1
    output = [r"({0}".format(round(v * 10**power, rnd)), round(v * 10**power, rnd)]
    if isinstance(err, list):
        for error in err:
            output[0] += r" $\pm$ {0}".format(round(error * 10**power, rnd))
            output.append(round(error * 10**power, rnd))
    else:
        output[0] += r" $\pm$ {0}".format(round(err * 10**power, rnd))
        output.append(round(err * 10**power, rnd))
    if out is str:
        return output[0] + r")e{0}".format(-power)
    else:
        return *output[1:], -power

def wcs_PA(PC21, PC22):
    """
    Return the position angle in degrees to the North direction of a wcs
    from the values of coefficient of its transformation matrix.
    """
    if (abs(PC21) > abs(PC22)) and (PC21 >= 0):
        orient = -np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) > abs(PC22)) and (PC21 < 0):
        orient = np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) < abs(PC22)) and (PC22 >= 0):
        orient = np.arccos(PC22) * 180.0 / np.pi
    elif (abs(PC21) < abs(PC22)) and (PC22 < 0):
        orient = -np.arccos(PC22) * 180.0 / np.pi
    return orient
