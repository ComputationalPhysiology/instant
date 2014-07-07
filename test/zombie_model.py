from __future__ import print_function
# Zombie model as described by 
# WHEN ZOMBIES ATTACK !: MATHEMATICAL MODELLING OF AN OUTBREAK OF ZOMBIE INFECTION
# by   Philip Munz, Ioan Hudea, Joe Imad, Robert J. Smith.
# In: Infectious Disease Modelling Research Progress, 
# Editors: J.M. Tchuenche and C. Chiyaka, pp. 133-150 2009 Nova Science Publishers, Inc.


from instant import inline_with_numpy
from numpy import *





    
c_code = """
void time_loop(int nS, double* S, 
               int nI, double* I, 
               int nZ, double* Z, 
               int nR, double* R, 
               double PI, double alpha, double beta, double delta, double rho, double zeta,  
               double dt, int N, double S0, double I0, double Z0, double R0)
{
    if ( nS != nI || nS != nZ || nS != nR || nS != N ) {
        printf("Arrays must be of same size!");  
        return; 
    }

    S[0] = S0; 
    I[0] = I0; 
    Z[0] = Z0; 
    R[0] = R0; 
    for (int i=0; i<N-1; i++) { 
        S[i+1] = S[i] + dt*(PI - beta*S[i]*Z[i] - delta*S[i]);
        I[i+1] = I[i] + dt*(beta*S[i]*Z[i] - (rho + delta)*I[i]); 
        Z[i+1] = Z[i] + dt*(rho*I[i] + zeta*R[i] - alpha*S[i]*Z[i]); 
        R[i+1] = R[i] + dt*(delta*S[i] + delta*I[i] + alpha*S[i]*Z[i] - zeta*R[i]); 
    }
}
"""

N = 100
time_loop = inline_with_numpy(c_code, arrays = [['nS', 'S'], ['nI', 'I'], ['nZ', 'Z'], ['nR', 'R']], cache_dir = "zombie_cache")
S = zeros(N)
I = zeros(N)
Z = zeros(N)
R = zeros(N)

T = 100 
dt = T/(N-1)
PI = 0.0 
alpha = 0.05 
beta = 0.095
zeta = 0.01 
delta = 0.01 
rho = 0.001 

S0 = 50
Z0 = 1 
R0 = 0 
I0 = 0 

time_loop(S, I, Z, R, PI, alpha, beta, delta, rho, zeta, dt, N, S0, I0, Z0, R0)


tt = arange(0, 1, 1.0/N) 

try: 
    import pylab
    pylab.plot(tt, S)
#    pylab.plot(tt, I)
    pylab.plot(tt, Z)
#    pylab.plot(tt, R)
    #pylab.savefig('pressure_plot')
    pylab.show()
except: 
    print("To get a plot of the solution install pylab")




