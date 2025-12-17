import numpy as np
import scipy.io as sio
import scipy.signal as signal
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


def run_seance2_plots():

    Fs_in = 44100
    Fs_out = 48000

    x = load_playback_44100()
    x = x[:Fs_in]  # 1 seconde

    L = 160
    M = 147

    # Sur-échantillonnage (insertion zéros)
    x_up = np.zeros(len(x) * L)
    x_up[::L] = x
    Fs_up = Fs_in * L

    # Filtre unique passe-bas (filtre classique)

    num_taps = 501
    fc = 1 / (2 * max(L, M))
    h = signal.firwin(num_taps+1, fc)
    x_filt = signal.lfilter(h, 1, x_up)

    print(f"Type de filtre : Filtre passe-bas FIR")
    print(f"Ordre du filtre : {num_taps - 1}")


    # Sous-échantillonnage
    y = x_filt[::M]
    Fs_y = Fs_out

    # Temps
    t_in = np.arange(len(x)) / Fs_in
    t_out = np.arange(len(y)) / Fs_y

    # Interpolation entrée sur grille sortie
    x_interp = np.interp(t_out, t_in, x)

    # Normalisation
    x_interp_norm = x_interp / np.max(np.abs(x_interp))
    y_norm = y / np.max(np.abs(y))

    # Compensation délai groupe FIR
    delay = (num_taps - 1) // 2
    y_aligned = y_norm[delay:]
    x_aligned = x_interp_norm[:-delay]
    t_aligned = t_out[:len(y_aligned)]

    # --- Calcul FFT entrée et sortie ---
    def compute_fft(sig, Fs):
        N = len(sig)
        win = np.hanning(N)
        S = np.fft.fft(sig * win)
        f = np.fft.fftfreq(N, 1 / Fs)
        return f[:N // 2], np.abs(S[:N // 2])

    f_in, S_in = compute_fft(x_interp_norm, Fs_y)
    f_out, S_out = compute_fft(y_norm, Fs_y)

    # --- Figure avec 4 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison signal et spectre SRC naïf 44.1 kHz → 48 kHz", fontsize=16)

    # 1) Signal entrée et sortie superposés (non alignés)
    axs[0, 0].plot(t_out[:2000], x_interp_norm[:2000], label="Entrée interpolée → 48 kHz", alpha=0.8)
    axs[0, 0].plot(t_out[:2000], y_norm[:2000], "--", label="Sortie SRC 48 kHz", alpha=0.8)
    axs[0, 0].set_title("Signal entrée et sortie superposés")
    axs[0, 0].set_xlabel("Temps (s)")
    axs[0, 0].set_ylabel("Amplitude normalisée")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # 2) Signal entrée et sortie superposés avec compensation délai groupe
    axs[0, 1].plot(t_aligned, x_aligned[:len(t_aligned)], label="Entrée interpolée")
    axs[0, 1].plot(t_aligned, y_aligned, "--", label="Sortie SRC alignée")
    axs[0, 1].set_title("Signal entrée / sortie avec compensation délai groupe")
    axs[0, 1].set_xlabel("Temps (s)")
    axs[0, 1].set_ylabel("Amplitude normalisée")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 3) Spectre (amplitude linéaire) entrée et sortie superposés
    axs[1, 0].plot(f_in, S_in / np.max(S_in), label="Spectre entrée")
    axs[1, 0].plot(f_out, S_out / np.max(S_out), "--", label="Spectre sortie")
    axs[1, 0].set_title("Spectre (amplitude linéaire) entrée et sortie")
    axs[1, 0].set_xlabel("Fréquence (Hz)")
    axs[1, 0].set_ylabel("Amplitude normalisée")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 4) FFT (dB) entrée et sortie superposés
    axs[1, 1].plot(f_in, 20 * np.log10(S_in / np.max(S_in) + 1e-12), label="FFT entrée (dB)")
    axs[1, 1].plot(f_out, 20 * np.log10(S_out / np.max(S_out) + 1e-12), "--", label="FFT sortie (dB)")
    axs[1, 1].set_title("FFT (dB) entrée et sortie superposés")
    axs[1, 1].set_xlabel("Fréquence (Hz)")
    axs[1, 1].set_ylabel("Amplitude (dB)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_seance2_plots()
