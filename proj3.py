import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import wofz

raw_data = np.loadtxt('P3data06.txt',skiprows=3) #Import the raw data

def Sort(Data,upper,lower): #Sort the data inbetween an upper and lower bound for, in the case, the wavelenght
    Data=np.asarray(Data)
    new_data=[]
    for i in range(len(Data)):
        if 10**Data[i][1]>=lower and 10**Data[i][1]<=upper:
            new_data.append(Data[i])
    return np.asarray(new_data)

def Gaussian(x): #The gaussian line profile
    return 1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2)
    
def Lorentzian(x): #The Lorentzian line profile
    return (1)/(np.pi*(1+x**2))

def Voigt(x): #The Voigt line profile
    return np.real(wofz((x + 1j)/np.sqrt(2))) /np.sqrt(2*np.pi)         

def ChisqFunc(para,dist,wavelenght,reflmbda,flux,ivar): #Chi-squared function. Takes in the sorted data and one of the line profiles. Also the parameter which are to be optimized
    if dist=='Gaussian':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Gaussian((wavelenght-para[3])/(para[4]))
    elif dist=='Lorentzian':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Lorentzian((wavelenght-para[3])/(para[4]))
    elif dist=='Voigt':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Voigt((wavelenght-para[3])/(para[4]))
    else:
        print('No or invalid model selected')
    return np.sum((flux-model)**2*ivar)

def Model(para,dist,wavelenght,reflmbda): # The fit model as a sperate function (one could have merged this and the previous function but i felt it was clearer to work with the code this way)
    if dist=='Gaussian':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Gaussian((wavelenght-para[3])/(para[4]))
    elif dist=='Lorentzian':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Lorentzian((wavelenght-para[3])/(para[4]))
    elif dist=='Voigt':
        model = para[0]+para[1]*(wavelenght-reflmbda)+para[2]*Voigt((wavelenght-para[3])/(para[4]))
    return np.asarray(model)

data=Sort(raw_data,6440,5900) #Getting the sortted data organized
flux = data[:,0]
wavelenght = 10**data[:,1]
ivar = data[:,2]
sigma = 1/(np.sqrt(data[:,2]))
reflmbda = 6175
initguess=[0,0,10,6175,50]
lambda_lab=2800.3

G_args=('Gaussian', wavelenght, reflmbda, flux, ivar) #Arguments as a sperate tuple for the scipy minimize function
L_args=('Lorentzian', wavelenght, reflmbda, flux, ivar)
V_args=('Voigt', wavelenght, reflmbda, flux, ivar)

G=optimize.minimize(ChisqFunc, initguess, args=G_args) #scipy minimize function to optimize the parameter for the chi-squared fucntion
L=optimize.minimize(ChisqFunc, initguess, args=L_args)
V=optimize.minimize(ChisqFunc, initguess, args=V_args)

initguessG=G.x #Saving the estimated parameters, G.x, for the later when we want to create monte carlo syntethic data
initguessL=L.x
initguessV=V.x

plt.figure(figsize=(6,5)) #plotting the gaussian fit
plt.plot(wavelenght,flux,color='b')
plt.plot(wavelenght,Model(G.x,'Gaussian',wavelenght,reflmbda),color='r',label='Gaussian fit',linewidth=3.0)
plt.xlim(5900,6440)
plt.ylim(6,15)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$]',fontsize=10)
plt.legend()
plt.show()

plt.figure(figsize=(6,5)) #plotting the Lorentzian fit
plt.plot(wavelenght,flux,color='b')
plt.plot(wavelenght,Model(L.x,'Lorentzian',wavelenght,reflmbda),color='lime',label='Lorentzian fit',linewidth=3.0)
plt.xlim(5900,6440)
plt.ylim(6,15)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$]',fontsize=10)
plt.legend()
plt.show()

plt.figure(figsize=(6,5)) #plotting the Voigt fit
plt.plot(wavelenght,flux,color='b')
plt.plot(wavelenght,Model(V.x,'Voigt',wavelenght,reflmbda),color='yellow',label='Voigt fit',linewidth=3.0)
plt.xlim(5900,6440)
plt.ylim(6,15)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$]',fontsize=10)
plt.legend()
plt.show()

plt.figure(figsize=(6,5)) #plotting the gaussian residue
plt.scatter(wavelenght,flux-Model(G.x,'Gaussian',wavelenght,reflmbda))
plt.hlines(np.polyfit(wavelenght,flux-Model(G.x,'Gaussian',wavelenght,reflmbda),0),5900,6440,color='r',label=str(np.polyfit(wavelenght,flux-Model(G.x,'Gaussian',wavelenght,reflmbda),0)),linewidth=2.0)
plt.ylabel(' ',fontsize=10)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.xlim(5900,6440)
plt.legend()
plt.show()

plt.figure(figsize=(6,5)) #plotting the Lorentzianresidue
plt.scatter(wavelenght,flux-Model(L.x,'Lorentzian',wavelenght,reflmbda))
plt.hlines(np.polyfit(wavelenght,flux-Model(L.x,'Lorentzian',wavelenght,reflmbda),0),5900,6440,color='lime',label=str(np.polyfit(wavelenght,flux-Model(L.x,'Lorentzian',wavelenght,reflmbda),0)),linewidth=2.0)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.xlim(5900,6440)
plt.legend()
plt.show()

plt.figure(figsize=(6,5)) #plotting the Voigt residue
plt.scatter(wavelenght,flux-Model(V.x,'Voigt',wavelenght,reflmbda))
plt.hlines(np.polyfit(wavelenght,flux-Model(V.x,'Voigt',wavelenght,reflmbda),0),5900,6440,color='yellow',label=str(np.polyfit(wavelenght,flux-Model(V.x,'Voigt',wavelenght,reflmbda),0)),linewidth=2.0)
plt.ylabel(' ',fontsize=10)
plt.xlabel('Wavelenght [Å]',fontsize=10)
plt.xlim(5900,6440)
plt.legend()
plt.show()

print('reduced chi-square Gaussian',ChisqFunc(G.x,'Gaussian',wavelenght,reflmbda,flux,ivar)/len(wavelenght)) #printing the reduced chi-squared value
print('reduced chi-square Lorentzian',ChisqFunc(L.x,'Lorentzian',wavelenght,reflmbda,flux,ivar)/len(wavelenght))
print('reduced chi-square Voigt',ChisqFunc(V.x,'Voigt',wavelenght,reflmbda,flux,ivar)/len(wavelenght))
print(' ')

synt_G=[]
synt_L=[]
synt_V=[]
for i in range(100): #getting synthetic data for the different distributions
    synt_g=Model(G.x,'Gaussian',wavelenght,reflmbda)+sigma*np.random.normal(0,1)
    synt_G.append(synt_g)
    synt_l=Model(L.x,'Lorentzian',wavelenght,reflmbda)+sigma*np.random.normal(0,1)
    synt_L.append(synt_l)
    synt_v=Model(V.x,'Voigt',wavelenght,reflmbda)+sigma*np.random.normal(0,1)
    synt_V.append(synt_v)
    
synt_para_G=[] #for each of the 100 diffrent sets of synthetics data, using a gaussian distribution, calculating the optimal parameters
for i in synt_G:
    Gsynt_args=('Gaussian', wavelenght, reflmbda, np.asarray(i), ivar)
    g=optimize.minimize(ChisqFunc, initguessG, args=Gsynt_args) #here we use the previous estimated parameters for the initail guess to, hopefully, speed up the processes
    synt_para_G.append(g.x)
synt_para_G=np.asarray(synt_para_G)

synt_para_L=[] #same as above but for a Lorentzian
for i in synt_L:
    syntL_args=('Lorentzian', wavelenght, reflmbda, np.asarray(i), ivar)
    l=optimize.minimize(ChisqFunc, initguessL, args=syntL_args)
    synt_para_L.append(l.x)
synt_para_L=np.asarray(synt_para_L)

synt_para_V=[] #same as above but for a Voigt
for i in synt_V:
    Vsynt_args=('Voigt', wavelenght, reflmbda, np.asarray(i), ivar)
    v=optimize.minimize(ChisqFunc, initguessV, args=Vsynt_args)
    synt_para_V.append(v.x)
synt_para_V=np.asarray(synt_para_V)


G_theta1=np.std(synt_para_G[:,0]) #getting the standard devivation of the different estimated paramters for each of the distributions
G_theta2=np.std(synt_para_G[:,1])
G_theta3=np.std(synt_para_G[:,2])
G_theta4=np.std(synt_para_G[:,3])
G_theta5=np.std(synt_para_G[:,4])

L_theta1=np.std(synt_para_L[:,0])
L_theta2=np.std(synt_para_L[:,1])
L_theta3=np.std(synt_para_L[:,2])
L_theta4=np.std(synt_para_L[:,3])
L_theta5=np.std(synt_para_L[:,4])

V_theta1=np.std(synt_para_V[:,0])
V_theta2=np.std(synt_para_V[:,1])
V_theta3=np.std(synt_para_V[:,2])
V_theta4=np.std(synt_para_V[:,3])
V_theta5=np.std(synt_para_V[:,4])

z_G=(synt_para_G[:,3]-lambda_lab)/lambda_lab #red shift estimate for each of the 100 different synthetic data sets
z_L=(synt_para_L[:,3]-lambda_lab)/lambda_lab
z_V=(synt_para_V[:,3]-lambda_lab)/lambda_lab

z_G_dev=np.std(z_G) #Getting the standard deviation for the red shift estimate
z_L_dev=np.std(z_L)
z_V_dev=np.std(z_V)

print('Gaussian parameter 1',G.x[0],' standard dev:', G_theta1) #printing different the different estimated paramters with thier corresponding standard deviation
print('Gaussian parameter 2',G.x[1],' standard dev:', G_theta2)
print('Gaussian parameter 3',G.x[2],' standard dev:', G_theta3)
print('Gaussian parameter 4',G.x[3],' standard dev:', G_theta4)
print('Gaussian parameter 5',G.x[4],' standard dev:', G_theta5)
print(' ')
print('Lorentzian parameter 1',L.x[0],' standard dev:', L_theta1)
print('Lorentzian parameter 2',L.x[1],' standard dev:', L_theta2)
print('Lorentzian parameter 3',L.x[2],' standard dev:', L_theta3)
print('Lorentzian parameter 4',L.x[3],' standard dev:', L_theta4)
print('Lorentzian parameter 5',L.x[4],' standard dev:', L_theta5)
print(' ')
print('Voigt parameter 1',V.x[0],' standard dev:', V_theta1)
print('Voigt parameter 2',V.x[1],' standard dev:', V_theta2)
print('Voigt parameter 3',V.x[2],' standard dev:', V_theta3)
print('Voigt parameter 4',V.x[3],' standard dev:', V_theta4)
print('Voigt parameter 5',V.x[4],' standard dev:', V_theta5)
print(' ')
print('redshift estimate Gaussian', (G.x[3]-lambda_lab)/(lambda_lab),' standard dev:',z_G_dev) #getting the red shift estimate from the orgininal data and the standard deviation from the synthetic data
print('redshift estimate Lorentzian', (L.x[3]-lambda_lab)/(lambda_lab),' standard dev:',z_L_dev)
print('redshift estimate Voigt', (V.x[3]-lambda_lab)/(lambda_lab),' standard dev:',z_V_dev)




















