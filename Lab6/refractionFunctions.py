import numpy as np
from math import acos, asin, pi

def refractVector(normal, incident, n1, n2):
	# Snell's Law		
	c1 = np.dot(normal, incident)
	
	if c1 < 0:
		c1 = -c1
	else:
		normal = np.array(normal) * -1
		n1, n2 = n2, n1

	n = n1 / n2
	
	T = n * (incident + c1 * normal) - normal * (1 - n**2 * (1 - c1**2 )) ** 0.5
	
	return T / np.linalg.norm(T)


def totalInternalReflection(normal, incident, n1, n2):
	c1 = np.dot(normal, incident)
	if c1 < 0:
		c1 = -c1
	else:
		n1, n2 = n2, n1
		
	if n1 < n2:
		return False
	
	theta1 = acos(max(-1, min(1, c1)))
	thetaC = asin(n2/n1)
	
	return theta1 >= thetaC


def fresnel(normal, incident, n1, n2):
	c1 = np.dot(normal, incident)
	if c1 < 0:
		c1 = -c1
	else:
		n1, n2 = n2, n1

	# Clamp c1 to prevent domain errors
	c1 = max(0, min(1, abs(c1)))
	
	# Ensure the expression under square root is non-negative
	sqrt_term = max(0, 1 - c1**2)
	s2 = (n1 * sqrt_term**0.5) / n2
	
	# Clamp s2 and ensure c2 calculation is valid
	s2 = max(0, min(1, s2))
	c2_sqrt_term = max(0, 1 - s2 ** 2)
	c2 = c2_sqrt_term ** 0.5
	
	# Prevent division by zero
	denominator1 = (n2 * c1) + (n1 * c2)
	denominator2 = (n1 * c2) + (n2 * c1)
	
	if abs(denominator1) < 1e-10 or abs(denominator2) < 1e-10:
		return 1.0, 0.0  # Total reflection
	
	F1 = (((n2 * c1) - (n1 * c2)) / denominator1) ** 2
	F2 = (((n1 * c2) - (n2 * c1)) / denominator2) ** 2

	Kr = (F1 + F2) / 2
	Kr = max(0, min(1, Kr))  # Clamp Kr to [0,1]
	Kt = 1 - Kr
	return Kr, Kt