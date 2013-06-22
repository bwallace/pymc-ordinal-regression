import pymc 
from pymc import Normal, Uniform, TruncatedNormal, deterministic, stochastic, observed

import pylab
import numpy

num_providers = 41
num_sessions = 371
n_clusters = 4
sample_count = 0
data = pylab.csv2rec('ml-cluster-data.csv')
## note that we drop the fourth cluster to avoid 
# over-determination (c0-c3 are mutually exclusive & exhaustive)
X_design = numpy.rec.fromarrays(
        [data.intercept, data.c_0, data.c_1, data.c_2],
        names=["intercept", "c0", "c1", "c2"])

X_mat = numpy.mat(X_design.tolist())

#--------------- hyper-parameters -------------#
inv_var = 1e-4
n = 0 

# overall ('high level') params
beta = pymc.Normal('beta', mu=[0]*n_clusters, tau=inv_var)

# and now make the beta matrix, which stacks beta
# coefficients (one per doctor) -- the n_clusters is 
# because this is how many parameters we have (we
# substract one, but then add an intercept)
Bdr = []
for provider_i in xrange(num_providers):
    Bdr.append(pymc.Normal('beta-dr-%s' % provider_i, mu=beta, tau=inv_var))

# construct a vector of betas |sessions| long
session_betas = []
for session_num, session_provider in enumerate(data.dr_id):
    session_betas.append(Bdr[int(session_provider)])

# the Betas to use for each session (which correspond to the 
# dr that participated in them).
SB = pymc.Container(session_betas)   

###
# setup the cut-off point parameters (lambda's)
# for this we will use truncated normals
#lambda_inv_var = 1e-5
lambdas = [pymc.Normal('lambda_0', 0, inv_var)]
for i in xrange(3):
    lambdas.append(pymc.TruncatedNormal('lambda_%s' % (i+1), (i+1), inv_var, lambdas[i], numpy.inf))
lambdas = pymc.Container(lambdas)

#-------------------- model ------------------#
@deterministic()
def y_hat(X=X_mat, session_betas=SB):
    # y_hat = x_i * beta_i 
    # where beta_i are coefficients corresponding
    # to the dr who participated in session i. 
    out = numpy.zeros(num_sessions)
    for i, x_i in enumerate(X):
        beta_i = session_betas[i]
        #out = out + numpy.dot(x_i, beta_i)
        #print numpy.dot(x_i, beta_i)[0][0]
        out[i] = numpy.dot(x_i, beta_i)[0]

    return out
  
# relatively tight variance
y_h = Normal("Y_h", mu=y_hat, tau=1e-2)

#@stochastic()
@deterministic
def y_hat_to_y(y_h=y_h, lambdas=lambdas):
    #return Normal("Y", mu=y_h, tau=1e5)
    def map_y_hat_i(y_h_i):
        if y_h_i < lambdas[0]:
            return 1
        elif y_h_i < lambdas[1]:
            return 2
        elif y_h_i < lambdas[2]:
            return 3
        elif y_h_i < lambdas[3]:
            return 4
        else:
            return 5

    mapper = numpy.vectorize(map_y_hat_i)

    #print mapper(y_h)
    return mapper(y_h)

# hacky way of getting a point mass (note the variance)
Y = Normal("Y", mu=y_hat_to_y, tau=1e5, observed=True, value=data.y)


