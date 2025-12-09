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
# Fonction d'intégration spectrale
# ------------------------------
def integrate_spectrum(P, f, f_cut=40_000):
    before_mask = f <= f_cut
    after_mask = f > f_cut

    power_before = np.trapz(P[before_mask], f[before_mask])
    power_after = np.trapz(P[after_mask], f[after_mask])

    return power_before, power_after

# ------------------------------
# Fonction SNR
# ------------------------------
def compute_snr(power_before, power_after):
    snr = power_before / power_after
    snr_dB = 10 * np.log10(snr)
    return snr, snr_dB

# ------------------------------
# Afficher signal PDM + spectre
# ------------------------------
def plot_pdm_and_spectrum(pdm, pdm_fs):
    N = 50000
    P = np.abs(np.fft.rfft(2*pdm[:N] - 1))**2
    f = np.fft.rfftfreq(N, 1/pdm_fs)

    # ---- Intégration du spectre avant/après 40 kHz ----
    power_before, power_after = integrate_spectrum(P, f, 40_000)
    snr, snr_dB = compute_snr(power_before, power_after)

    print("\n=== Analyse spectrale du signal PDM (brut) ===")
    print(f"Puissance < 40 kHz : {power_before}")
    print(f"Puissance > 40 kHz : {power_after}")
    print(f"SNR = {snr}")
    print(f"SNR (dB) = {snr_dB} dB")

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
    plt.grid()
    plt.show()

# ------------------------------
# Démodulation PDM avec choix du filtre FIR ou IIR
# ------------------------------
def pdm_demodulator(pdm, OSR=128, filter_type="FIR"):
    """
    Démodule un signal PDM en PCM.
    
    filter_type : "FIR" ou "IIR"
    """

    # Signal binaire en bipolaire
    pdm_bipolar = 2 * pdm - 1

    # Fréquence de coupure normalisée (bande audio)
    fc = 0.45 * (1 / OSR)

    if filter_type.upper() == "FIR":
        # ---------------- FIR ----------------
        M = 255
        h = signal.firwin(M, fc)
        filtered = signal.lfilter(h, 1, pdm_bipolar)

    elif filter_type.upper() == "IIR":
        # ---------------- IIR (Butterworth) ----------------
        order = 4  # valeur raisonnable, stable
        b, a = signal.butter(order, fc)
        filtered = signal.lfilter(b, a, pdm_bipolar)
        h = (b, a)  # on retourne les coefficients IIR

    else:
        raise ValueError("filter_type doit être 'FIR' ou 'IIR' !")

    # Décimation
    pcm = filtered[::OSR]

    # Normalisation
    pcm = pcm / np.max(np.abs(pcm))

    # Conversion 20 bits
    pcm_20b = np.round(pcm * (2**19 - 1)).astype(np.int32)

    return pcm_20b, filtered, h


# ------------------------------
# Spectre du signal filtré
# ------------------------------
def plot_filtered_spectrum(filtered, pdm_fs):
    N_filt = len(filtered)
    P_filt = np.abs(np.fft.rfft(filtered))**2
    f_filt = np.fft.rfftfreq(N_filt, 1/pdm_fs)

    # ---- Intégration + SNR ----
    power_before, power_after = integrate_spectrum(P_filt, f_filt, 40_000)
    snr, snr_dB = compute_snr(power_before, power_after)

    print("\n=== Analyse spectrale après filtre passe-bas ===")
    print(f"Puissance < 40 kHz : {power_before}")
    print(f"Puissance > 40 kHz : {power_after}")
    print(f"SNR = {snr}")
    print(f"SNR (dB) = {snr_dB} dB")

    plt.figure()
    plt.semilogx(f_filt, 10*np.log10(P_filt))
    plt.title("Spectre du signal filtré")
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
