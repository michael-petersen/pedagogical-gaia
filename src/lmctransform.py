"""
transformations from EDR3 LMC paper

michael petersen, january 2021

"""
import numpy as np


# now to make the on-sky rotated coordinates...

# fit parameters from EDR3 paper
alpha_c = 81.28
delta_c = -69.78
inclination = 34.08
omega = 309.92
mux0 = 1.858
muy0 = 0.385
muz0 = 1.115
v0 = 75.9
r0 = 2.94
alpharc = 5.306
DLMC = 49.5 # in kpc


def ortho_x(delta,alpha,alpha_c=81.28,delta_c=-69.78):
    # eq 1
    delta*=np.pi/180.;alpha*=np.pi/180.;alpha_c*=np.pi/180.
    return np.cos(delta)*np.sin(alpha-alpha_c)

def ortho_y(delta,alpha,alpha_c=81.28,delta_c=-69.78):
    # eq 1
    delta*=np.pi/180.;alpha*=np.pi/180.;alpha_c*=np.pi/180.;delta_c*=np.pi/180.
    return np.sin(delta)*np.cos(delta_c) - np.cos(delta)*np.sin(delta_c)*np.cos(alpha-alpha_c)

def Mmatrix(delta,alpha,alpha_c=81.28,delta_c=-69.78):
    delta*=np.pi/180.;alpha*=np.pi/180.;alpha_c*=np.pi/180.;delta_c*=np.pi/180.
    return np.array([ [np.cos(alpha-alpha_c),-np.sin(delta)*np.sin(alpha-alpha_c)],
                      [np.sin(delta_c)*np.sin(alpha-alpha_c),np.cos(delta)*np.cos(delta_c)+np.sin(delta)*np.sin(delta_c)*np.cos(alpha-alpha_c)]])


def mu_xy(mu_alpha,mu_delta,emu_alpha,emu_delta,ecorr,delta,alpha,alpha_c=81.28,delta_c=-69.78):
    # eq 2
    mmatrix = Mmatrix(delta,alpha,alpha_c,delta_c)
    mu  = np.array([mu_alpha,mu_delta]).T
    C = np.array([[emu_alpha*emu_alpha,ecorr*emu_delta*emu_alpha],[ecorr*emu_delta*emu_alpha,emu_delta*emu_delta]])
    outmu = np.zeros([mu.shape[0],2])
    Cmu = np.zeros([mu.shape[0],2,2])
    for indx in range(0,mu.shape[0]):
        outmu[indx] = np.dot(mmatrix[:,:,indx],mu[indx])
        Cmu[indx] = np.dot(mmatrix[:,:,indx],np.dot(C[:,:,indx],mmatrix[:,:,indx].T))
    return outmu,Cmu


def only_mu_xy(mu_alpha,mu_delta,emu_alpha,emu_delta,ecorr,delta,alpha,alpha_c=81.28,delta_c=-69.78):
    delta*=np.pi/180.;alpha*=np.pi/180.;alpha_c*=np.pi/180.;delta_c*=np.pi/180.
    mu_x = mu_alpha*np.cos(alpha-alpha_c) - mu_delta*np.sin(delta)*np.sin(alpha-alpha_c)
    mu_y = mu_alpha*np.sin(delta_c)*np.sin(alpha-alpha_c) + mu_delta*(np.cos(delta)*np.cos(delta_c) + np.sin(delta)*np.sin(delta_c)*np.cos(alpha-alpha_c))
    return mu_x,mu_y





def return_z(x,y):
    # eq B4 of Gaia 2018
    return np.sqrt(1.-x*x-y*y)

def return_vz(x,y,vx,vy):
    # eq B4 of Gaia 2018
    z = return_z(x,y)
    return -(x*vx + y*vy)/z
    

def return_zeta(x,y,inclination=34.08,omega=309.92):
    # eq 5
    inclination*=np.pi/180.;omega*=np.pi/180.
    z = return_z(x,y)
    a = np.tan(inclination)*np.cos(omega)
    b = -np.tan(inclination)*np.sin(omega)
    lx,ly = np.sin(omega),np.cos(omega)
    
    numerator = lx*x + ly*y
    denominator = z + a*x + b*y
    return numerator/denominator

def return_eta(x,y,inclination=34.08,omega=309.92):
    # eq 5
    # inclination=0,omega=90 returns the same projection
    inclination*=np.pi/180.;omega*=np.pi/180.
    z = return_z(x,y)
    a = np.tan(inclination)*np.cos(omega)
    b = -np.tan(inclination)*np.sin(omega)
    lx,ly = np.sin(omega),np.cos(omega)
    mx,my,mz = -np.cos(inclination)*np.cos(omega),np.cos(inclination)*np.sin(omega),np.sin(inclination)
    
    numerator = (mx - a*mz)*x + (my-b*mz)*y
    denominator = z + a*x + b*y
    
    return numerator/denominator


def return_zetaeta_dot(x,y,mu_x,mu_y,mu_x0=1.858,mu_y0=0.385,mu_z0=1.115,inclination=34.08,omega=309.92):
    inclination*=np.pi/180.;omega*=np.pi/180.
    z = return_z(x,y)
    a = np.tan(inclination)*np.cos(omega)
    b = -np.tan(inclination)*np.sin(omega)
    lx,ly = np.sin(omega),np.cos(omega)
    mx,my,mz = -np.cos(inclination)*np.cos(omega),np.cos(inclination)*np.sin(omega),np.sin(inclination)
    
    # first equation
    rhs1 = -mu_x0 + x*(mu_x0*x + mu_y0*y + mu_z0*z) + mu_x/(a*x + b*y + z)
    lhsf11 = lx - x*(lx*x + ly*y)
    lhsf12 = mx - x*(mx*x + my*y + mz*z)
    
    # second equation
    rhs2 = -mu_y0 + y*(mu_x0*x + mu_y0*y + mu_z0*z) + mu_y/(a*x + b*y + z)
    lhsf21 = ly - y*(lx*x + ly*y)
    lhsf22 = my - y*(mx*x + my*y + mz*z)
    
    # now the equations are
    # lhsf11*vzeta + lhsf12*veta = rhs1
    # lhsf21*vzeta + lhsf22*veta = rhs2
    
    # so multiply first equation by -lhsf21 and second by lhsf11:
    # -lhsf21*lhsf11*vzeta + -lhsf21*lhsf12*veta = -lhsf21*rhs1
    #  lhsf21*lhsf11*vzeta +  lhsf22*lhsf11*veta =  lhsf11*rhs2
    # and add together, rearrange for veta
    # 
    veta = (-lhsf21*rhs1 + lhsf11*rhs2)/(-lhsf21*lhsf12 + lhsf22*lhsf11)
    vzeta = (rhs1 - lhsf12*veta)/lhsf11
    return veta,vzeta
    

def planar_vtan(zeta,eta,vzeta,veta):
    r = np.sqrt(zeta*zeta+eta*eta)
    return (zeta*veta - eta*vzeta)/r

def planar_vr(zeta,eta,vzeta,veta):
    r = np.sqrt(zeta*zeta+eta*eta)
    return (zeta*vzeta + eta*veta)/r



