import bisect
import antropy as ant
import numpy as np
import neurokit2 as nk
import pyhrv.time_domain as td
from matplotlib import *
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import hilbert

path = "/home/manef/tests/figures/figure_18_09_2021/unhealthy/"
ID = "14-2"

# == load the epileptic patient
with open(path + "App-PN" + ID + ".txt", 'r') as file1:
    seizure_app = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

with open(path + "Sample-PN" + ID + ".txt", 'r') as file1:
    seizure_sample = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

with open(path + "RRi-PN" + ID + ".txt", 'r') as file1:
    seizure_HRV = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

peaks_ep = []
with open(path + "peaks-PN" + ID + ".txt", 'r') as file1:
    peaks_ep = [float(i) for line in file1 for i in line.split('\n') if i.strip()]


# == load healthy subjects
path = "/home/manef/tests/data/fantasia-database-1.0.0(1)/young_to_try_in_test/f1y07/"

ID_healthy = "f1y07-0_60"

with open(path + "App-" + ID_healthy + ".txt", 'r') as file1:
    healthy_app = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

with open(path + "Sample-" + ID_healthy + ".txt", 'r') as file1:
    healthy_sample = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

with open(path + "RRi-" + ID_healthy + ".txt", 'r') as file1:
    healthy_HRV = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

peaks = []
with open(path + "peaks-" + ID_healthy + ".txt", 'r') as file1:
    peaks = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

# == healthy subjects
def features(peaks, fs):

    NN50, SDNN, approximate, sample, RRi, fft, welch, RMSSD = ([] for i in range(8))
    start = 0
    end = 120 * fs

    while True:

        go = bisect.bisect_left(peaks, start)
        out = bisect.bisect_left(peaks, end)

        RRi = []
        peaks_in = []
        ff = 0
        peaks_in = peaks[go:out]

        for i in range(len(peaks_in) - 1):
            new = peaks_in[i + 1] - peaks_in[i]
            new = (new / fs)
            RRi.append(new)
            if (new > 0.05):
                ff = ff+1

        # == NN50
        NN50.append(ff)

        # = SDNN
        SDNN.append(np.std(RRi))
        fft.append(ant.spectral_entropy(RRi, sf=fs, method='fft', normalize=False, nperseg=len(RRi)))
        welch.append(ant.spectral_entropy(RRi, sf=fs, method='welch', normalize=False, nperseg=len(RRi)))

        # == RSMMD
        RMSSD.append(np.sqrt(np.mean(np.power(RRi, 2))))


        # = reset parameters
        start = start + (10 * fs)
        end = end + (10 * fs)
        i += 1

        if (end > peaks[-1]):
            break

    return NN50, SDNN, welch, RMSSD

fs_ep = 512
XXX_ep = 0

NN50_ep, pNN50_ep, SDNN_ep, SDNN_test, nn50_test_sei, fft_ep, welch_ep, RMSSD_ep = ([] for i in range(8))

NN50_ep, SDNN_ep, welch_ep, RMSSD_ep = features(peaks_ep, fs_ep)

# == healthy subjects

fs = 250
XXX = 0

nn50_test_hea, SDNN, pNN50, NN50, welch, fft, RMSSD = ([] for i in range(7))

NN50, SDNN, welch, RMSSD = features(peaks, fs)

first = 0
tt = [first]
i = 1
while i <= len(seizure_app) - 1:
    first = first + 10
    tt.append(first)
    i = i + 1

first = 0
tt_healthy = [first]
i = 1
while i <= len(healthy_app) - 1:
    first = first + 10
    tt_healthy.append(first)
    i = i + 1


###########################################################
# == treating  app entropy of the epileptic patient
fs = 512

analytic_signal = hilbert(seizure_app)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

# == treating the sample entropy of the epileptic patient signal

analytic_signal_s = hilbert(seizure_sample)
amplitude_envelope_s = np.abs(analytic_signal_s)
instantaneous_phase_s = np.unwrap(np.angle(analytic_signal_s))
instantaneous_frequency_s = (np.diff(instantaneous_phase_s) / (2.0*np.pi) * fs)

# == treating the approximate entropy of the healthy signal
fs = 250
analytic_signal_h = hilbert(healthy_app)
amplitude_envelope_h = np.abs(analytic_signal_h)
instantaneous_phase_h = np.unwrap(np.angle(analytic_signal_h))
instantaneous_frequency_h = (np.diff(instantaneous_phase_h) / (2.0*np.pi) * fs)

# == treating the ample entropy of the healthy signal

analytic_signal_s_h = hilbert(healthy_sample)
amplitude_envelope_s_h = np.abs(analytic_signal_s_h)
instantaneous_phase_s_h = np.unwrap(np.angle(analytic_signal_s_h))
instantaneous_frequency_s_h = (np.diff(instantaneous_phase_s_h) / (2.0*np.pi) * fs)

# == treating the HRV of epileptic patients

analytic_signal_hrv_h = hilbert(healthy_HRV)
amplitude_envelope_hrv_h = np.abs(analytic_signal_hrv_h)
instantaneous_phase_hrv_h = np.unwrap(np.angle(analytic_signal_hrv_h))
instantaneous_frequency_hrv_h = (np.diff(instantaneous_phase_hrv_h) / (2.0*np.pi) * fs)

# == treating the HRV of the healthy signal

analytic_signal_hrv = hilbert(seizure_HRV)
amplitude_envelope_hrv = np.abs(analytic_signal_hrv)
instantaneous_phase_hrv = np.unwrap(np.angle(analytic_signal_hrv))
instantaneous_frequency_hrv = (np.diff(instantaneous_phase_hrv) / (2.0*np.pi) * fs)

###########################################################
# == plotting the approximate entropy features
ymin = 0.1
ymax = 1.3

fig, axs = plt.subplots(2, 1)

axs[0].plot(tt, seizure_app, label="approximate entropy in seizures period", marker='o')
axs[1].plot(tt_healthy, healthy_app, label="approximate entropy for healthy patients", marker='o')

axs[0].set_title('Patient ' + ID,fontsize=24, y=1)
axs[1].set_title('Subject' + ID_healthy,fontsize=24, y=1)

axs[0].axvline(x= tt[-1] - 600, color='red', linestyle='--')

axs[0].set_ylim([ymin,ymax])
axs[1].set_ylim([ymin,ymax])

axs[0].set_xlabel('sample en seconde')
axs[0].set_ylabel('valeur de entropie')

axs[1].set_xlabel('sample en seconde')
axs[1].set_ylabel('valeur de entropie')

axs[0].grid(True)
axs[1].grid(True)

axs[0].legend()
axs[1].legend()

###########################################################
# == plotting the sample entropy features
ymin = 0
ymax = 2.5

fig, axs = plt.subplots(2, 1)

axs[0].plot(tt, seizure_sample, label="samople entropy in seizures period", marker='o')
axs[1].plot(tt_healthy, healthy_sample, label="sample entropy for healthy patients", marker='o')

axs[0].set_ylim([ymin,ymax])
axs[1].set_ylim([ymin,ymax])

axs[0].set_title('Patient ' + ID,fontsize=24, y=1)
axs[1].set_title('Subject' + ID_healthy,fontsize=24, y=1)

axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')

axs[0].set_xlabel('sample en seconde')
axs[0].set_ylabel('valeur de entropie')

axs[1].set_xlabel('sample en seconde')
axs[1].set_ylabel('valeur de entropie')

axs[0].grid(True)
axs[1].grid(True)

axs[0].legend()
axs[1].legend()

###########################################################
# == plotting the sample entropy features

ymin = 0.4
ymax = 1.4
fig, axs = plt.subplots(2, 1)

axs[0].plot(tt, RMSSD_ep, label="RMSSD in seizures period", marker='o')
axs[1].plot(tt_healthy, RMSSD, label="RMSSD for healthy patients", marker='o')

axs[0].set_ylim([ymin,ymax])
axs[1].set_ylim([ymin,ymax])

axs[0].set_title('Patient ' + ID,fontsize=24, y=1)
axs[1].set_title('Subject' + ID_healthy,fontsize=24, y=1)

axs[0].axvline(x=tt[-1] - 600, color='red', linestyle='--')

axs[0].set_xlabel('sample en seconde')
axs[0].set_ylabel('valeur de entropie')

axs[1].set_xlabel('sample en seconde')
axs[1].set_ylabel('valeur de entropie')

axs[0].grid(True)
axs[1].grid(True)

axs[0].legend()
axs[1].legend()

## == plotting the approximate entropy hilbert filter
fs = 512
t = np.arange(len(seizure_app)) / fs

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, seizure_app, label='seizure app')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()

## == plotting the sample entropy hilbert filter
t = np.arange(len(seizure_sample)) / fs

fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, seizure_sample, label='seizure sample')
ax0.plot(t, amplitude_envelope_s, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency_s)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()

# == plotting the sample entropy hilbert filter
fs = 250
t = np.arange(len(healthy_sample)) / fs

t = np.arange(len(healthy_sample)) / fs
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, healthy_sample, label='healthy sample')
ax0.plot(t, amplitude_envelope_s_h, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency_s_h)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()

t = np.arange(len(healthy_app)) / fs
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, healthy_app, label='healthy app')
ax0.plot(t, amplitude_envelope_h, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency_h)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()

# == plotting the hrv hilbert filter OF HEALTHY SUBJECT

fs = 250
t = np.arange(len(healthy_HRV)) / fs
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, healthy_HRV, label='healthy sample')
ax0.plot(t, amplitude_envelope_hrv_h, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency_hrv_h)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()

# == plotting the hrv hilbert filter OF EPILEPTIC SUBJECT
fs = 512
t = np.arange(len(seizure_HRV)) / fs
fig, (ax0, ax1) = plt.subplots(nrows=2)
ax0.plot(t, seizure_HRV, label='healthy app')
ax0.plot(t, amplitude_envelope_hrv, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1.plot(t[1:], instantaneous_frequency_hrv)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 50.0)
fig.tight_layout()


plt.show()

print('')
