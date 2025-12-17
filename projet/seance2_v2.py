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

    # -----------------------------
    # Figure avec 4 subplots
    # -----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparaison signal et spectre SRC multistage 44.1 kHz → 48 kHz", fontsize=16)

    # 1) Signal temporel non aligné
    axs[0, 0].plot(t_in[:2000], x_norm[:2000], label="Entrée 44.1 kHz", alpha=0.8)
    axs[0, 0].plot(t_out[:2000], y_norm[:2000], "--", label="Sortie SRC 48 kHz", alpha=0.8)
    axs[0, 0].set_title("Signal entrée et sortie (non alignés)")
    axs[0, 0].set_xlabel("Temps (s)")
    axs[0, 0].set_ylabel("Amplitude normalisée")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # 2) Signal temporel avec alignement grossier
    delay_samples = int((len(x_norm) - len(y_norm)) / 2)
    delay_samples = max(delay_samples, 0)

    x_aligned = x_norm[delay_samples:delay_samples + len(y_norm)]
    t_aligned = np.arange(len(x_aligned)) / Fs_final

    axs[0, 1].plot(t_aligned[:2000], x_aligned[:2000], label="Entrée alignée")
    axs[0, 1].plot(t_aligned[:2000], y_norm[:2000], "--", label="Sortie SRC alignée")
    axs[0, 1].set_title("Signal entrée / sortie (alignement grossier)")
    axs[0, 1].set_xlabel("Temps (s)")
    axs[0, 1].set_ylabel("Amplitude normalisée")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 3) Spectre amplitude linéaire
    S_in = np.abs(np.fft.rfft(x_norm))
    S_out = np.abs(np.fft.rfft(y_norm))

    f_in = np.fft.rfftfreq(len(x_norm), 1 / Fs_in)
    f_out = np.fft.rfftfreq(len(y_norm), 1 / Fs_final)

    axs[1, 0].plot(f_in, S_in / np.max(S_in), label="Spectre entrée")
    axs[1, 0].plot(f_out, S_out / np.max(S_out), "--", label="Spectre sortie")
    axs[1, 0].set_xlim(0, 24000)
    axs[1, 0].set_title("Spectre amplitude linéaire")
    axs[1, 0].set_xlabel("Fréquence (Hz)")
    axs[1, 0].set_ylabel("Amplitude normalisée")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 4) Spectre FFT en dB
    axs[1, 1].plot(
        f_in,
        20 * np.log10(S_in / np.max(S_in) + 1e-12),
        label="FFT entrée (dB)"
    )
    axs[1, 1].plot(
        f_out,
        20 * np.log10(S_out / np.max(S_out) + 1e-12),
        "--",
        label="FFT sortie (dB)"
    )
    axs[1, 1].set_xlim(0, 24000)
    axs[1, 1].set_ylim(-120, 5)
    axs[1, 1].set_title("Spectre FFT (dB)")
    axs[1, 1].set_xlabel("Fréquence (Hz)")
    axs[1, 1].set_ylabel("Amplitude (dB)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_seance2_multistage()
