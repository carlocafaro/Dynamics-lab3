# Dynamics-lab3

import numpy as np
import pylab
from scipy import integrate
from scipy.integrate import tplquad
from data import getdata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def vectorfield(x, t, sigma, rho, beta):
    """
    Function to return the value of dx/dt, given x.
    
    Inputs:
    x - the value of x
    t - the value of t
    alpha - a parameter
    beta - another parameter

    Outputs:
    dxdt - the value of dx/dt
    """
    
    return np.array([sigma*(x[1]-x[0]),
                        x[0]*(rho-x[2])-x[1],
                        x[0]*x[1]-beta*x[2]])




def getdata(y0,T,Deltat):
    """
    Function to return simulated observational data from a dynamical
    system, for testing out forecasting tools with.

    Inputs:
    y0 - the initial condition of the system state
    T - The final time to simulate until
    Deltat - the time intervals to obtain observations from. Note
    that the numerical integration method is time-adaptive and
    chooses it's own timestep size, but will interpolate to obtain
    values at these intervals.
    """
    t = np.arange(0.,T,Deltat)
    data = integrate.odeint(vectorfield, y0, t=t,args=(10,28,8/3))
    return data, t  






def ensemble_plot(Ens,title=''):
    """
    Function to plot the locations of an ensemble of points.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Ens[:,0],Ens[:,1],Ens[:,2],'.')    
    pylab.title(title)

def ensemble(M,T):
    """
    Function to integrate an ensemble of trajectories of the Lorenz system.
    Inputs:
    M - the number of ensemble members
    T - the time to integrate for
    """
    ens = 0.1*np.random.randn(M,3)

    for m in range(M):
        t = np.array([0.,T])
        data = integrate.odeint(vectorfield, 
                                y0=ens[m,:], t=t, args=(10.0,28.,8./3.),
                                mxstep=100000)

        ens[m,:] = data[1,:]

    return ens

def f1(d):
    """
    Function to apply ergodic average to.
    inputs:
    X - array, first dimension time, second dimension component
    outputs:
    f: - 1d- array of f values
    """
    
    return d[:,0]**2 + d[:,1]**2 + d[:,2]**2


def f2(d):
    "Function to apply ergodic average to"
    
    return d[:,0]**3+d[:,1]**3+d[:,2]**3

def spatialaverageMC(f,ens):
    
    "Function to compute the Monte-Carlo Approximation of the spatial average" 
    
    
    return np.average(f(ens))
    


def exercise2part1():

    "Choosing a function f investigate the variance in the approximation with respect to this random initial condition"
    "Here we want to see that the error in the MC approximation decays as M^(-0.5)"
    
    simulations=100
    f_bar=np.zeros(simulations)

    for i in range(simulations):
        f_bar[i]= spatialaverageMC(f1,ensemble(100,100))
    Var1 = np.var(f_bar)
    print ('Variance for 100 simulations, 100 ensemble members at time T=100', Var1)
    
    f_bar1=np.zeros(simulations)
    for i in range(simulations):
        f_bar1[i]= spatialaverageMC(f1,ensemble(200,100))
    Var2 = np.var(f_bar1)
    print 'Variance for 100 simulations, 200 ensemble members at time T=100', Var2
    
    f_bar2=np.zeros(simulations)
    for i in range(simulations):
        f_bar2[i]= spatialaverageMC(f1,ensemble(300,100))
    Var3 = np.var(f_bar2)
    print 'Variance for 100 simulations, 300 ensemble members at time T=100', Var3






def exercise2part2():
    "Here we want to compute the temporal average, running the Lorenz system at each time N*Deltat, with Deltat fixed and N varying between 50 and 2000"
    "Then we take the average at each time-step and we perform this calculation four times in order to demonstrate the indipendence of the average"
    "with respect to the random initial conditions"
    
    Deltat=0.1
    
    timearray=np.linspace(50.0, 2000.0, 50)
    temporalaverage=np.zeros((len(timearray),4))
    
    "To demonstrate the indipendence wrt to the initial conditions here we are generating four different time averages"
    for j in range(4):
        for i in range(len(timearray)):
            t = np.arange(0.,timearray[i]*Deltat,Deltat)
            data = integrate.odeint(vectorfield, np.random.randn(3), t=t,args=(10,28,8/3))
            temporalaverage[i,j]=np.average(f1(data))
        
    plt.plot(timearray, temporalaverage[:,:])
    plt.show()
    print 'From the plot we can see that the rate of convergence is approximately exponentially with the time'
    print 'Furthemore for the maximum time limit the four different plots for different random initial conditions are very close, suggesting the'
    print 'indipendence of the time average wrt to the initial conditions'      
    

def exercise3part1(T=200, Deltat=0.1, epsilon=0.5):
    
    "In this exercise we want to perturb the parameters of the Lorenz system to see how the shape of the Lorenz attractor change"
    
    y0=np.random.randn(3)
    t = np.arange(0.,T, Deltat)
    data = integrate.odeint(vectorfield, y0, t=t,args=(10,28,8/3))
    data_sigma=integrate.odeint(vectorfield, y0, t=t,args=(10+epsilon,28,8/3))
    data_rho=integrate.odeint(vectorfield, y0, t=t,args=(10,28+epsilon,8/3))
    data_beta=integrate.odeint(vectorfield, y0, t=t,args=(10,28,8/3+epsilon))
    fig = plt.figure()
    ax0 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    
    ax0.plot(data[:,0],data[:,1],data[:,2])
    ax1.plot(data_sigma[:,0],data_sigma[:,1],data_sigma[:,2])
    ax2.plot(data_rho[:,0],data_rho[:,1],data_rho[:,2])
    ax3.plot(data_beta[:,0],data_beta[:,1],data_beta[:,2])

    ax0.set_title('Standard parameters')
    ax1.set_title('Sigma perturbed')
    ax2.set_title('Rho perturbed')
    ax3.set_title('Beta perturbed')

    pylab.show()
    print 'From the plot, the system seems to be more sensibile in beta perturbations'

def exercise3part2(f,N=2000.0, Deltat=0.1):

    "In this second part of the exercise 3 we want to calculate the linear response to the small changes in parameters"
    "In order to do this we calculate the expectation value for each parameter value and then we average in time, using the ergodic property"
    
    rho_0=28.0
    sigma_0=10.0
    beta_0=8/3
    Delta=0.5
    steps=100
    rho = np.linspace(rho_0-Delta, rho_0+Delta, steps)
    sigma=np.linspace(sigma_0-Delta, sigma_0+Delta, steps)
    beta=np.linspace(beta_0-Delta, beta_0+Delta, steps)
    averages = np.zeros(len(rho))
    averages1= np.zeros(len(sigma))
    averages2=np.zeros(len(beta))
    partials = np.zeros_like(rho, dtype = float)
    y_0 = np.random.randn(3) 
    # We create an array of times, using Deltat,
    t = np.arange(0.,N*Deltat,Deltat)
    standard_data = integrate.odeint(vectorfield, y_0, t=t, args=(10.0,rho_0,8./3.))
    standard_average = np.average(f(standard_data))
    for i in range(len(rho)):
        # and integrate the vectorfield with initial condition y_0 taken from a Gaussian.
        data = integrate.odeint(vectorfield, y_0, t=t, args=(10.0,rho[i],8./3.))
        # We then apply our function 'f' to each data point (spread over time) and store the average.
        averages[i] = np.average(f(data))
    
    for i in range(len(sigma)):
        # and integrate the vectorfield with initial condition y_0 taken from a Gaussian.
        data = integrate.odeint(vectorfield, y_0, t=t, args=(sigma[i],28,8./3.))
        # We then apply our function 'f' to each data point (spread over time) and store the average.
        averages1[i] = np.average(f(data))
    
    for i in range(len(beta)):
        # and integrate the vectorfield with initial condition y_0 taken from a Gaussian.
        data = integrate.odeint(vectorfield, y_0, t=t, args=(10.0,28.0,beta[i]))
        # We then apply our function 'f' to each data point (spread over time) and store the average.
        averages2[i] = np.average(f(data))
        
        
        
    "Uncomment what you want to show and print"
    plt.plot(rho,averages)
    #plt.plot(sigma,averages1)
    #plt.plot(beta,averages2)
    plt.show()
    if f==f1:    
        print ('The response in rho is approximately linear with some noise and sharp points, where the response is not differentiable')
        print ('The response in sigma does not look like linear. There are many sharp points')
        print ('The response in beta is quite linear, with few sharp points')
    else:
        print('The response in rho is quite linear with some noise and sharp points')
        print('The response in sigma is not linear at all, with many sharp points')
        print ('The response in beta is roughly linear, but no so obvious')
    


    
if __name__ == '__main__':
    
    
    
    
   
    
       
    #exercise2part1()
    #exercise2part2()
    #exercise3part1()
   
    #exercise3part2(f2, 2000.0, 0.1)
   
