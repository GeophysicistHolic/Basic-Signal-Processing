import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

lt = 128
dt = 1
t = np.arange(0, lt-1, dt)


chirp = np.cos(2*np.pi*(10+(t/7))*(t/lt))



a = 0.01
ti = 8
ts = lt/16

fdr = []
fdn = []
fdm = []
fdb = []
fdg = []

i=0
while (ti+(ts*i)) <= lt :
    hn = np.zeros(len(t))
    hm = np.zeros(len(t))
    hb = np.zeros(len(t))
    N = 20
    if(N+(ts*i)<=len(t)):
        for j in range(0,N) :
            hn[j+int((ts*i))] = 0.5*(1 - np.cos(2*np.pi*((j)/N)))
    else:
        for j in range(0,N):
            if ((j+int(ts*i))<len(t)):
                hn[j+int(ts*i)] = 0.5*(1 - np.cos(2*np.pi*((j)/N)))
    
    if(N+(ts*i)<=len(t)):
        for j in range(0,N) :
            hm[j+int((ts*i))] = 0.54 - 0.46*np.cos(2*np.pi*((j)/N))
    else:
        for j in range(0,N):
            if ((j+int(ts*i))<len(t)):
                hm[j+int((ts*i))] = 0.54 - 0.46*np.cos(2*np.pi*((j)/N))
    
    if(N+(ts*i)<=len(t)):
        for j in range(0,N) :
            hb[j+int((ts*i))] = 0.42 - 0.5*np.cos(2*np.pi*((j)/N)) + 0.08*np.cos(4*np.pi*((j)/N))
    else:
        for j in range(0,N):
            if ((j+int(ts*i))<len(t)):
                hb[j+int((ts*i))] = 0.42 - 0.5*np.cos(2*np.pi*((j)/N)) + 0.08*np.cos(4*np.pi*((j)/N))
    
    hr = np.linspace(0-(ts*i), 127-(ts*i), 127)
    hr = np.where(abs(hr)<=20, 1, 0)
    
    gaus = np.exp(-1*a*((t-(((ts*i)+(ts*(i-1)))/2))**2)/2)
    
    ftr = chirp*hr
    ftg = chirp*gaus
    ftn = chirp*hn
    ftm = chirp*hm
    ftb = chirp*hb
    
    FTftn = np.fft.fft(ftn)
    FTftm = np.fft.fft(ftm)
    FTftb = np.fft.fft(ftb)
    FTftr = np.fft.fft(ftr)
    FTftg = np.fft.fft(ftg)
    
    fdn.append(FTftn)
    fdm.append(FTftm)
    fdb.append(FTftb)
    fdr.append(FTftr)
    fdg.append(FTftg)
    
    i+=1
    
fdn = np.array(fdn)
fdm = np.array(fdm)
fdb = np.array(fdb)
fdr = np.array(fdr)
fdg = np.array(fdg)

freq = np.fft.fftfreq(len(t), d=0.001)

Ar = np.empty((len(t),len(freq)))
An = np.empty((len(t),len(freq)))
Am = np.empty((len(t),len(freq)))
Ab = np.empty((len(t),len(freq)))
Ag = np.empty((len(t),len(freq)))

for i in range (len(fdn)):
    fdn[i] = abs(fdn[i])
    fdm[i] = abs(fdm[i])
    fdb[i] = abs(fdb[i])
    fdr[i] = abs(fdr[i])
    fdg[i] = abs(fdg[i])
    
a = 0
for i in range(len(t)):
    if i <=  (ti+(ts*a)):
        An[i] = fdn[a]
        Am[i] = fdm[a]
        Ab[i] = fdb[a]
        Ar[i] = fdr[a]
        Ag[i] = fdg[a]
    else :
        a+=1

An = np.transpose(An)
Am = np.transpose(Am)
Ab = np.transpose(Ab)
Ar = np.transpose(Ar)
Ag = np.transpose(Ag)         


fig, axs = plt.subplots(3,2)

axs[0,0].plot(t,chirp)
axs[0, 0].set_title('Original Signal Chirp')
axs[0,0].set(xlabel='Time(ms)', ylabel='Amplitude')
axs[0,1].pcolormesh(t, freq[:63], An[:][:63], cmap='RdBu', shading='gouraud')
axs[0, 1].set_title('Spectogram Hanning')
axs[0,1].set(xlabel='Time(ms)', ylabel='Frekuensi (Hz)')
axs[1,0].pcolormesh(t, freq[:63], Am[:][:63], cmap='RdBu', shading='gouraud')
axs[1, 0].set_title('Spectogram Hamming')
axs[1,0].set(xlabel='Time(ms)', ylabel='Frekuensi (Hz)')
axs[1,1].pcolormesh(t, freq[:63], Ab[:][:63], cmap='RdBu', shading='gouraud')
axs[1, 1].set_title('Spectogram Blackman')
axs[1,1].set(xlabel='Time(ms)', ylabel='Frekuensi (Hz)')
axs[2,0].pcolormesh(t, freq[:63], Ar[:][:63], cmap='RdBu', shading='gouraud')
axs[2, 0].set_title('Spectogram Rectangular')
axs[2,0].set(xlabel='Time(ms)', ylabel='Frekuensi (Hz)')
axs[2,1].pcolormesh(t, freq[:63], Ag[:][:63], cmap='RdBu', shading='gouraud')
axs[2, 1].set_title('Spectogram Gaussian')
axs[2,1].set(xlabel='Time(ms)', ylabel='Frekuensi (Hz)')


for i in range(len(fdg)):
    plt.figure(4)
    plt.plot(freq[:63], fdg[i][:63], label='Length'+str(i))
    plt.legend()

fig = plt.figure(6)
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t, freq[:63])
ax.plot_surface(X, Y, An[:][:63])
ax.set_xlabel('time (ms)')
ax.set_ylabel('frequencies (Hz)')
ax.set_zlabel('amplitude')
ax.set_title('Hanning')

fig = plt.figure(7)
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t, freq[:63])
ax.plot_surface(X, Y, Am[:][:63])
ax.set_xlabel('time (ms)')
ax.set_ylabel('frequencies (Hz)')
ax.set_zlabel('amplitude')
ax.set_title('Hamming')

fig = plt.figure(8)
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t, freq[:63])
ax.plot_surface(X, Y, Ab[:][:63])
ax.set_xlabel('time (ms)')
ax.set_ylabel('frequencies (Hz)')
ax.set_zlabel('amplitude')
ax.set_title('Blackman')

fig = plt.figure(9)
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t, freq[:63])
ax.plot_surface(X, Y, Ar[:][:63])
ax.set_xlabel('time (ms)')
ax.set_ylabel('frequencies (Hz)')
ax.set_zlabel('amplitude')
ax.set_title('Rectangular')

fig = plt.figure(10)
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(t, freq[:63])
ax.plot_surface(X, Y, Ag[:][:63])
ax.set_xlabel('time (ms)')
ax.set_ylabel('frequencies (Hz)')
ax.set_zlabel('amplitude')
ax.set_title('Gaussian')

plt.show()