import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys
from scipy.signal import savgol_filter
import seaborn as sns

sns.set_style('dark', {'axes.grid':True,'axes.edgecolor':'black', 'font.family':['serif'], 'font.serif':['Roman'],
                       'xtick.bottom':True, 'ytick.left':True})


#Normalized by number of spins
mat1 = np.loadtxt("L40_expec_values.txt",skiprows=1) / (40**2)
mat2 = np.loadtxt("L60_expec_values.txt",skiprows=1) / (60**2)
mat3 = np.loadtxt("L80_expec_values.txt",skiprows=1) / (80**2)
mat4 = np.loadtxt("L100_expec_values.txt",skiprows=1) / (100**2)

temp = mat1[:,0]*40**2 #Temp is not normalized

#Parameters for smoothing filter
poly_deg = 3
win_size =5

#Smoothing with Savitzky–Golay filter
Cv_smooth = np.zeros((len(temp),4))
Cv_smooth[:,0] = savgol_filter(mat1[:,3], win_size, poly_deg)
Cv_smooth[:,1] = savgol_filter(mat2[:,3], win_size, poly_deg)
Cv_smooth[:,2] = savgol_filter(mat3[:,3], win_size, poly_deg)
Cv_smooth[:,3] = savgol_filter(mat4[:,3], win_size, poly_deg)

chi_smooth = np.zeros((len(temp),4))
chi_smooth[:,0] = savgol_filter(mat1[:,4], win_size, poly_deg)
chi_smooth[:,1] = savgol_filter(mat2[:,4], win_size, poly_deg)
chi_smooth[:,2] = savgol_filter(mat3[:,4], win_size, poly_deg)
chi_smooth[:,3] = savgol_filter(mat4[:,4], win_size, poly_deg)

Tc1 = temp[np.argmax(Cv_smooth[:,0])]
Tc2 = temp[np.argmax(Cv_smooth[:,1])]
Tc3 = temp[np.argmax(Cv_smooth[:,2])]
Tc4 = temp[np.argmax(Cv_smooth[:,3])]

T_c_list = [Tc1,Tc2,Tc3,Tc4]
L_nu_list = [1./40, 1./60, 1./80, 1./100 ]

a,Tc_inf = np.polyfit(L_nu_list, T_c_list, deg=1, rcond=None, full=False, w=None, cov=False)
Tc_inf_analytic = 2./np.log(1+np.sqrt(2))
rel_err = np.abs((Tc_inf - Tc_inf_analytic ) / Tc_inf_analytic)

print('T_c_inf from C_v is: %e, which is an offset of %e from the analytical value, with an relative error of %e'  %(Tc_inf, np.abs(Tc_inf_analytic-Tc_inf), rel_err))

Tc1 = temp[np.argmax(mat1[:,3])]
Tc2 = temp[np.argmax(mat2[:,3])]
Tc3 = temp[np.argmax(mat3[:,3])]
Tc4 = temp[np.argmax(mat4[:,3])]

T_c_list = [Tc1,Tc2,Tc3,Tc4]
a,Tc_inf = np.polyfit(L_nu_list, T_c_list, deg=1, rcond=None, full=False, w=None, cov=False)
rel_err = np.abs((Tc_inf - Tc_inf_analytic ) / Tc_inf_analytic)

print('T_c_inf from not smooth C_v is: %e, which is an offset of %e from the analytical value, with an relative error of %e'  %(Tc_inf, np.abs(Tc_inf_analytic-Tc_inf), rel_err))

Tc1 = temp[np.argmax(chi_smooth[:,0])]
Tc2 = temp[np.argmax(chi_smooth[:,1])]
Tc3 = temp[np.argmax(chi_smooth[:,2])]
Tc4 = temp[np.argmax(chi_smooth[:,3])]

T_c_list = [Tc1,Tc2,Tc3,Tc4]
a,Tc_inf = np.polyfit(L_nu_list, T_c_list, deg=1, rcond=None, full=False, w=None, cov=False)
rel_err = np.abs((Tc_inf - Tc_inf_analytic ) / Tc_inf_analytic)

print('T_c_inf from chi is: %e, which is an offset of %e from the analytical value, with an relative error of %e'  %(Tc_inf, np.abs(Tc_inf_analytic-Tc_inf), rel_err))


Tc1 = temp[np.argmax(mat1[:,4])]
Tc2 = temp[np.argmax(mat2[:,4])]
Tc3 = temp[np.argmax(mat3[:,4])]
Tc4 = temp[np.argmax(mat4[:,4])]

T_c_list = [Tc1,Tc2,Tc3,Tc4]
a,Tc_inf = np.polyfit(L_nu_list, T_c_list, deg=1, rcond=None, full=False, w=None, cov=False)
rel_err = np.abs((Tc_inf - Tc_inf_analytic ) / Tc_inf_analytic)

print('T_c_inf from not smooth chi is: %e, which is an offset of %e from the analytical value, with an relative error of %e'  %(Tc_inf, np.abs(Tc_inf_analytic-Tc_inf), rel_err))


plt.figure()
plt.plot(temp,mat1[:,1], label="L = 40")
plt.plot(temp,mat2[:,1], label="L = 60")
plt.plot(temp,mat3[:,1], label="L = 80")
plt.plot(temp,mat4[:,1], label="L = 100")
plt.xlabel(r"Temperature, [$\frac{k_B T}{J}$]",)
plt.ylabel(r"$\frac{\langle E \rangle}{L^2}$", rotation = 0, labelpad = 11, fontsize = 16)
plt.legend()
plt.savefig('E_final.svg')
plt.show()


plt.figure()
plt.plot(temp,mat1[:,2], label="L = 40")
plt.plot(temp,mat2[:,2], label="L = 60")
plt.plot(temp,mat3[:,2], label="L = 80")
plt.plot(temp,mat4[:,2], label="L = 100")
plt.xlabel(r"Temperature, [$\frac{k_B T}{J}$]")
plt.ylabel(r"$\frac{\langle|M|\rangle}{L^2}$", rotation = 0, labelpad = 15, fontsize = 16)
plt.legend()
plt.savefig('M_final.svg')
plt.show()


plt.figure()
plt.plot(temp,Cv_smooth[:,0], label="L = 40, smoothed")
plt.plot(temp,Cv_smooth[:,1], label="L = 60, smoothed")
plt.plot(temp,Cv_smooth[:,2], label="L = 80, smoothed")
plt.plot(temp,Cv_smooth[:,3], label="L = 100, smoothed")
plt.xlabel(r"Temperature, [$\frac{k_B T}{J}$]")
plt.ylabel(r"$\frac{C_v}{L^2}$", rotation = 0, labelpad = 15, fontsize = 16)
plt.legend()
plt.savefig('Cv_final.svg')
plt.show()


plt.figure()
plt.plot(temp,mat1[:,4], label="L = 40")
plt.plot(temp,mat2[:,4], label="L = 60")
plt.plot(temp,mat3[:,4], label="L = 80")
plt.plot(temp,mat4[:,4], label="L = 100")
plt.xlabel(r"Temperature, [$\frac{k_B T}{J}$]")
plt.ylabel(r"$\frac{\chi}{L^2}$", rotation = 0, labelpad = 15, fontsize = 16)
plt.legend()
plt.savefig('Chi_final.svg')
plt.show()
