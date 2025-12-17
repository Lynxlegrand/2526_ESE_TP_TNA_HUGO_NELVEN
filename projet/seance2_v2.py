import numpy as np
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt
import os


def load_playback_44100():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data", "playback_44100.mat")

    data = sio.loadmat(path)
    keys = [k for k in data.keys() if not k.startswith("__")]

    if len(keys) != 1:
        raise ValueError(f"Format inattendu. Variables trouvées : {keys}")

    print("Variable trouvée dans le .mat :", keys[0])
    return data[keys[0]].flatten()


def resample_stage(x, L, M, num_taps=501):
    """
    Resample une étape avec interpolation xL puis filtrage puis décimation xM.

    x: signal d'entrée
    L: facteur d'interpolation (insertion zéros)
    M: facteur de décimation (prise d'échantillons)
    num_taps: ordre filtre FIR
    """
    # Interpolation (sur-échantillonnage)
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    Fs_up = L  # échelle relative, fréquence normale pas nécessaire ici

    # Fréquence de coupure passe-bas (anti-repliement)
    fc = 1 / (2 * max(L, M))

    # Conception filtre FIR passe-bas
    h = signal.firwin(num_taps, fc)

    # Filtrage
    x_filt = signal.lfilter(h, 1, x_up)

    # Décimation (sous-échantillonnage)
    y = x_filt[::M]

    # Compensation délai groupe
    delay = (num_taps - 1) // 2
    y = y[delay:]  # pour garder signal filtré aligné (optionnel, peut ajuster plus tard)

    return y


def run_seance2_multistage():

    Fs_in = 44100
    Fs_out = 48000

    x = load_playback_44100()
    x = x[:Fs_in]  # 1 seconde

    # Étape 1 : x16, filtre1, x1 (pas de décimation ici)
    stage1 = resample_stage(x, L=16, M=1, num_taps=501)

    # Étape 2 : x10, filtre2, x1 (idem)
    stage2 = resample_stage(stage1, L=10, M=1, num_taps=501)

    # Étape 3 : x1, filtre3, x7 (décimation par 7)
    # Ici, on inverse L et M pour ne faire que décimation
    stage3 = resample_stage(stage2, L=1, M=7, num_taps=501)

    # Étape 4 : x1, filtre4, x21 (décimation par 21)
    stage4 = resample_stage(stage3, L=1, M=21, num_taps=501)

    y = stage4

    # Fréquence finale calculée
    Fs_final = Fs_in * 16 * 10 / (7 * 21)

    print(f"Fréquence finale : {Fs_final} Hz")

    # Normalisation
    y_norm = y / np.max(np.abs(y))
    x_norm = x / np.max(np.abs(x))

    # Alignement temporel simple (non précis ici)
    t_in = np.arange(len(x_norm)) / Fs_in
    t_out = np.arange(len(y_norm)) / Fs_final

    # Tracé simple signal
    plt.figure(figsize=(12, 5))
    plt.plot(t_in[:1000], x_norm[:1000], label="Entrée 44.1 kHz")
    plt.plot(t_out[:1000], y_norm[:1000], label="Sortie multistage SRC")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude normalisée")
    plt.legend()
    plt.title("Signal entrée vs sortie multistage")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_seance2_multistage()
