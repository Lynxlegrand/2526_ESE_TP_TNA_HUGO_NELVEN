import numpy as np
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt
import os



# ============================================================
# NAIVE PDM DEMODULATOR USING MOVING AVERAGE (RECTANGULAR LPF)
# ============================================================
def pdm_demodulator_naive(pdm, OSR=128):
    # pdm est déjà en bipolaire {-1, 1}, on n'a donc pas besoin de conversion

    # Nombre de blocs décimés
    n_blocks = len(pdm) // OSR

    # Allocation tableau PCM
    pcm = np.zeros(n_blocks)

    # Moyenne simple sur chaque bloc OSR (filtrage + décimation)
    for i in range(n_blocks):
        block = pdm[i*OSR:(i+1)*OSR]
        pcm[i] = np.mean(block)

    # Normalisation pour utilisation sur 20 bits
    pcm = pcm / np.max(np.abs(pcm))
    pcm_20b = np.round(pcm * (2**19 - 1)).astype(np.int32)

    return pcm_20b


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
    pdm : signal PDM bipolar {-1, 1} (entrée)
    OSR : oversampling ratio (pdm_fs / fs_out)
    """
    # Pas besoin de conversion si pdm est déjà bipolar
    pdm_bipolar = pdm

    # Conception filtre passe-bas FIR linéaire en phase
    fc = 0.48 * (1 / OSR)               # fréquence normalisée
    M = 511                              # ordre FIR
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

    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data/pdm_in.mat")

    data = sio.loadmat(path)

    # on supprime les clés internes de MATLAB
    keys = [k for k in data.keys() if not k.startswith("__")]

    if len(keys) != 1:
        raise ValueError(f"Format inattendu. Variables trouvées : {keys}")

    print("Variable trouvée dans le .mat :", keys[0])
    return data[keys[0]].flatten()

import numpy as np
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ------------------------------
# Charger le signal PDM depuis fichier .mat
# ------------------------------
def load_pdm():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data/pdm_in.mat")
    data = sio.loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Format inattendu. Variables trouvées : {keys}")
    print("Variable trouvée dans le .mat :", keys[0])
    return data[keys[0]].flatten()

# ------------------------------
# Afficher signal PDM + spectre
# ------------------------------
def plot_pdm_and_spectrum(pdm, pdm_fs):
    N = 50000
    P = np.abs(np.fft.rfft(2*pdm[:N]-1))**2
    f = np.fft.rfftfreq(N, 1/pdm_fs)

    fig, ax1 = plt.subplots()
    ax1.semilogx(f, 10*np.log10(P), color="red")
    ax1.set_ylabel("Puissance (dB)", color="red")
    ax1.set_xlabel("Fréquence (Hz)")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(pdm[:300], color="blue")
    ax2.set_ylabel("Amplitude PDM", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("Signal PDM brut + Spectre")
    plt.grid(True)
    plt.show()

# ------------------------------
# Démodulation PDM (filtre passe-bas FIR + décimation)
# ------------------------------
def pdm_demodulator(pdm, OSR=128):
    pdm_bipolar = 2 * pdm - 1
    fc = 0.45 * (1 / OSR)
    M = 255
    h = signal.firwin(M, fc)
    filtered = signal.lfilter(h, 1, pdm_bipolar)
    pcm = filtered[::OSR]
    pcm = pcm / np.max(np.abs(pcm))
    pcm_20b = np.round(pcm * (2**19 - 1)).astype(np.int32)
    return pcm_20b, filtered, h

# ------------------------------
# Affichage spectre du signal après filtrage
# ------------------------------
def plot_filtered_spectrum(filtered, pdm_fs):
    N_filt = len(filtered)
    P_filt = np.abs(np.fft.rfft(filtered))**2
    f_filt = np.fft.rfftfreq(N_filt, 1/pdm_fs)

    plt.figure()
    plt.semilogx(f_filt, 10*np.log10(P_filt))
    plt.title("Spectre du signal après filtre passe-bas")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Puissance (dB)")
    plt.grid()
    plt.show()

# ------------------------------
# Affichage du signal PCM démodulé
# ------------------------------
def plot_pcm_signal(pcm_20b):
    plt.figure()
    plt.plot(pcm_20b[:2000])
    plt.title("Signal PCM démodulé (20 bits)")
    plt.xlabel("Échantillons")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# ------------------------------
# Fonction principale
# ------------------------------
def run_seance1():
    pdm = load_pdm()
    print("Signal PDM chargé :", len(pdm), "échantillons")
    pdm_fs = 6_144_000
    fs_audio = 48_000
    OSR = pdm_fs // fs_audio

    plot_pdm_and_spectrum(pdm, pdm_fs)

    pcm_20b, filtered, h = pdm_demodulator(pdm, OSR=OSR)

    plot_filtered_spectrum(filtered, pdm_fs)
    plot_pcm_signal(pcm_20b)

if __name__ == "__main__":
    run_seance1()


