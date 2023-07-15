#import torch
import math
import mpmath
import scipy
from scipy import special

solidAngle = 0.0

# takes 3 inputs: r0, rm, and L from the paper
def solidAngleCalc(r0, rm, L):

    Rmax = math.sqrt( L**2 + (r0+rm)**2 )
    R1 = math.sqrt( L**2 + (r0-rm)**2 )
    k = 2*math.sqrt( r0 * rm / (L * L + ( r0 + rm )**2) )
    alpha = (4*r0*rm) / (r0+rm)**2 #alpha^2 in the paper

    if (r0 < rm): 
        solidAngle = 2*math.pi - 2*(L/Rmax)*(scipy.special.ellipk (k**2) + math.sqrt(1 - alpha) * mpmath.ellippi(alpha, k**2))

    if (r0 == rm):
        solidAngle = math.pi - (2*L / Rmax) * scipy.special.ellipk (k**2)

    if (r0 > rm): 
        solidAngle = (2*L / Rmax) * ( ((r0-rm) / (r0+rm))*mpmath.ellippi(alpha, k**2) - scipy.special.ellipk (k**2))

    return solidAngle

# test
print("solidAngleCalc:", solidAngleCalc(0.2,1,1) )