import numpy as np
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt


# ============================================================
#  FUNCTION : Simple Sigma-Delta 1st-order PDM Modulator
# ============================================================
def pdm_modulator(x):
    """
    Entrée : signal PCM normalisé dans [-1, 1]
    Sortie : suite PDM 0/1
    """
    acc = 0
    y = np.zeros(len(x))

    for i in range(len(x)):
        if acc >= 0:
            y[i] = 1
            acc += x[i] - 1
        else:
            y[i] = 0
            acc += x[i] + 1

    return y


# ============================================================
#  FUNCTION : PDM demodulator (LPF + decimation)
# ============================================================
def pdm_demodulator(pdm, OSR=128, fs_out=48000):
    """
    pdm : signal PDM 0/1 (entrée)
    OSR : oversampling ratio (pdm_fs / fs_out)
    """
    # Conversion vers +1/-1
    pdm_bipolar = 2 * pdm - 1

    # Conception filtre passe-bas FIR linéaire en phase
    fc = 0.45 * (1 / OSR)               # fréquence normalisée
    M = 255                              # ordre FIR
    h = signal.firwin(M, fc)

    # Filtrage + décimation
    filtered = signal.lfilter(h, 1, pdm_bipolar)
    pcm = filtered[::OSR]

    # Normalisation 20 bits
    pcm = pcm / np.max(np.abs(pcm))
    pcm_20b = np.round(pcm * (2**19 - 1)).astype(np.int32)

    return pcm_20b, filtered, h


# ============================================================
#  LOAD & VISUALIZE PDM FILE
# ============================================================
def load_pdm():
    data = sio.loadmat("data/pdm_in.mat")
    pdm = data["data/pdm_in.mat"].flatten()
    return pdm


# ============================================================
#  MAIN
# ============================================================
def run_seance1():

    # ------------------------------
    # Load PDM signal
    # ------------------------------
    pdm = load_pdm()
    print("Signal PDM chargé :", len(pdm), "échantillons")

    pdm_fs = 6_144_000       # PDM clock
    fs_audio = 48_000
    OSR = pdm_fs // fs_audio

    # ------------------------------
    # Time-domain observation
    # ------------------------------
    plt.figure()
    plt.plot(pdm[:300])
    plt.title("Signal PDM (zoom)")
    plt.show()

    # ------------------------------
    # Frequency-domain observation
    # ------------------------------
    N = 50000
    P = np.abs(np.fft.rfft(2*pdm[:N]-1))**2
    f = np.fft.rfftfreq(N, 1/pdm_fs)

    plt.figure()
    plt.semilogx(f, 10*np.log10(P))
    plt.title("Spectre du PDM")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid()
    plt.show()

    # ------------------------------
    # DEMODULATION
    # ------------------------------
    pcm_20b, filtered, h = pdm_demodulator(pdm, OSR=OSR, fs_out=fs_audio)

    # Affichage PCM
    plt.figure()
    plt.plot(pcm_20b[:2000])
    plt.title("Signal PCM démodulé (20 bits)")
    plt.show()

    # ------------------------------
    # TEST MODULATEUR
    # ------------------------------
    t = np.arange(0, 1, 1/fs_audio)
    test_sin = 0.9 * np.sin(2*np.pi*1000*t)

    pdm_out = pdm_modulator(test_sin)

    plt.figure()
    plt.plot(pdm_out[:300])
    plt.title("Modulation PDM d'un sinus 1 kHz")
    plt.show()

    print("✔ Démodulation + modulation PDM terminées.")
