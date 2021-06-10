
# ---
# from Peters lecture course 
# https://sml505.pmelchior.net/14-ApproximateInference.html#simulation-based-inference
# Peter reference for GMM 
# https://github.com/pmelchior/pygmmis
# SKLearn reference for GMM
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

# %%
import numpy as np
import matplotlib.pyplot as plt
from seaborn import pairplot
import pandas as pd
import pygmmis
from sklearn.mixture import GaussianMixture as GMM


# %%
# --- Define forward simulation model
# aka a linear model with Gaussian noise 



np.random.seed(0)

def simulate(theta, x=None):
    sigma = 0.1
    a, b = theta
    model = a + b*x
    noise = np.random.normal(size=len(x), scale=sigma)
    
    # # let's be nasty later on ...
    # from scipy.stats import halfcauchy
    # noise = halfcauchy.rvs(size=len(x), scale=sigma)

    return model + noise

theta = (0,1)
N = 30
x = np.random.rand(N)
y_o = simulate(theta, x)

plt.scatter(x, y_o, c='C3', label='Observation')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend(frameon=False)

# %%
# -- define compressed statistic 
# 'the derivative of the adopted log-likelihood wrt theta at fiducial position 
def summary(x, y, mu, C_1):
    grad_ab = np.vander(x, N=2, increasing=True)
    return grad_ab.T @ C_1 @ (y - mu) # matrix multiplication


# %%
# --- Generate a bunch of simulations with different parameters and compute compmressed statistics 
# using the true parameters that were used to generate the "observed data":
mu = theta[0] + theta[1]*x
# inverse covariance of the simulation at fixed parameters
C_1 = np.eye(N) / 0.1**2

n_sims = 100
d = 2
thetas = np.empty((n_sims, d))
ts = np.empty((n_sims, d))

for n in range(n_sims):
    # draw prior is simply 2D Gaussian with some suitable std
    thetas[n] = np.random.normal(theta, scale=2)
    # simulate with parameter
    y = simulate(thetas[n], x)
    # compress to summaries
    ts[n] = summary(x, y, mu, C_1)

data = np.concatenate((thetas, ts), axis=1)
df = pd.DataFrame(data, columns=(r'$\theta_0$', r'$\theta_1$', '$t_0$', '$t_1$'))
pairplot(df, corner=True)

# reference:
# t_0 is summary statistic 0 (corresponing with parameter value theta_0)
# t_1 is summary statistic 1 (corresponding with parameter value theta_1)
# the pairplot generated shows the empirical distribution p(t, D) is nicely Gaussian. 

# %%
# --- 
# final steps

# fit the 4-Dimensional Gaussian dataset (df) with a GMM fitter. 
# GMM stands for Gaussian Mixture Model
# https://scikit-learn.org/stable/modules/mixture.html
# enables one to learn Gaussian Mixture Models (diagonal, spherical, tied, and full covariance), sample them, and estimate from data
# A two component Gaussian Mixture Model is basically just data points and the equi probability surfaces of a model
# GMMs generalize K-means clustering to incorporate information about covariance structure of the data as well as the centers of latent Gaussians

# # this method didn't seem to work very well.
# gmm = pygmmis.GMM(K=1, D=2*d)
# pygmmis.fit(gmm, data, init_method='kmeans')

# # Backup from sklearn
gmm = GMM(n_components=2,random_state=0)
gmm.fit(data)

print(gmm.means_)
print(gmm.covariances_.shape)





# %%
# get sub-vectors/matrices for t and theta
k = 0

# # using pygmm
# mu_theta = gmm.mean[k,:2]
# mu_t = gmm.mean[k,2:]
# C_theta = gmm.covar[k,:2,:2]
# C_t = gmm.covar[k,2:,2:]
# C_x = gmm.covar[k,:2, 2:]

# using sklearn 
mu_theta = gmm.means_[k,:2]
mu_t = gmm.means_[k,2:]
C_theta = gmm.covariances_[k,:2,:2]
C_t = gmm.covariances_[k,2:,2:]
C_x = gmm.covariances_[k,:2, 2:]

# evaluate at t_ = t(observation)
mu_t_ = summary(x, y_o, mu, C_1)
mu_theta_ = mu_theta + C_x @ np.linalg.inv(C_t) @ (mu_t_ - mu_t)
C_theta_ = C_theta - C_x @ np.linalg.inv(C_t) @ C_x.T

# compute MLE
sigma = 0.1
X = np.vander(x, N=2, increasing=True)
Sigma_1 = 1/sigma**2 * np.eye(len(x))
C_mle = np.linalg.inv(X.T @ Sigma_1 @ X)
mle = C_mle @ X.T @ Sigma_1 @ y_o

# plot the results
from matplotlib.patches import Ellipse
def add_ellipses(ax, pos, C, sigmas=[1,2], labels=None, set_axis=True, **kwargs):
    eigval, eigvec = np.linalg.eig(C)
    angle = np.degrees(np.arctan2(eigvec[1,0], eigvec[0,0]))
    w, h = 2 * np.sqrt(eigval)
    for i,n in enumerate(sigmas):
        if labels is not None:
            label = labels[i]
        else:
            label = None
        ax.add_patch(Ellipse(pos, width=n*w, height=n*h, angle=angle, fill=False, label=label, **kwargs))
    dist = max(w,h) * 1.1*max(sigmas)
    if set_axis is True:
        ax.set_xlim(pos[0] - dist/2, pos[0] + dist/2)
        ax.set_ylim(pos[1] - dist/2, pos[1] + dist/2)
        

fig = plt.figure()
ax = fig.gca()
ax.scatter(*(0,1), c='k', label='Truth', zorder=10)

ax.scatter(*mle, c='C0', label='MLE', zorder=10)
add_ellipses(ax, mle, C_mle, edgecolor='C0')

ax.scatter(*mu_theta_, c='C3', label='DELFI')
add_ellipses(ax, mu_theta_, C_theta_, edgecolor='C3')
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$b$')
ax.legend(frameon=False)

# %%
# readme on Gaussian Mixture Algorithm from SKLearn 
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
import numpy as np
from sklearn.mixture import GaussianMixture
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
gm.means_
gm.predict([[0, 0], [12, 3]])

# %%
