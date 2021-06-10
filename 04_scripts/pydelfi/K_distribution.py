# %%

# ------- 
# This script for extracting 'K' values to use within the ParFlow simulation 
# K values are in meters per second 

# References: 
# https://numpy.org/doc/stable/reference/generated/numpy.flip.html
# https://stackoverflow.com/questions/41000668/log-x-axis-for-histogram
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
# https://images.app.goo.gl/2eoptUANtmNF5igQ7

# %%
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from numpy.random import shuffle
from numpy.random import randn, rand, lognormal, logseries, multivariate_normal, normal, uniform
import matplotlib.pyplot as plt

# %%
# K information
# range -> https://images.app.goo.gl/2eoptUANtmNF5igQ7

# # --- Turn the knob on 'k' to get the desired values for PARFLOW Simulation

# create a range of K values
k = 20 # size of output
k_min, k_max, u = 14, 0, 7 # upper lower, u bound of K values (nb of form 10**-val)
sig = 3 # sigma of normal distribution (nb of form 10**-val)

# gaussian distribution
k_gauss = normal(loc=u,scale=sig,size=k)
k_gauss[k_gauss > k_min] = k_min
k_gauss[k_gauss < k_max] = k_max
# print(k_gauss)
bins = np.arange(k_max,k_min+1)
plt.hist(k_gauss, bins)
plt.title('Gauss - Raw')
plt.show()

k_gauss = 10**-k_gauss
bins = np.flip(1/(10**(np.arange(k_max,k_min+1))))
# print(k_gauss)
plt.xscale('log')
plt.hist(k_gauss, bins=bins)
plt.title('Gauss - True K')
plt.show()


# %%
# # uniform distribution
k_uni = uniform(low=k_max,high=k_min,size=k)
bins = np.arange(k_max,k_min+1)
plt.hist(k_uni, bins)
plt.title('Uniform - Raw')
plt.show()

k_uni = 10**-k_uni
bins = np.flip(1/(10**(np.arange(k_max,k_min+1))))
plt.xscale('log')
plt.hist(k_uni, bins=bins)
plt.title('Uniform - True K')
plt.show()
# %%
