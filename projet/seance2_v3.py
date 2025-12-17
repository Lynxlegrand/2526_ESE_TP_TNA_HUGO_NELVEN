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
        raise ValueError(f"Unexpected format. Variables found: {keys}")

    return data[keys[0]].flatten()


def run_optimal_src_plots():

    Fs_in = 44100
    Fs_out = 48000

    x = load_playback_44100()
    x = x[:Fs_in]  # 1 second

    # Rational SRC factors
    L = 160
    M = 147

    # -----------------------------
    # Optimal polyphase SRC
    # -----------------------------
    y = signal.resample_poly(
        x,
        up=L,
        down=M,
        window=("kaiser", 8.6)  # ~90 dB attenuation
    )

    # Time vectors
    t_in = np.arange(len(x)) / Fs_in
    t_out = np.arange(len(y)) / Fs_out

    # High-quality reference interpolation
    x_ref = signal.resample(x, len(y))

    # Normalization
    x_ref /= np.max(np.abs(x_ref))
    y /= np.max(np.abs(y))

    # -----------------------------
    # FFT helper
    # -----------------------------
    def compute_fft(sig, Fs):
        N = len(sig)
        win = np.hanning(N)
        S = np.fft.fft(sig * win)
        f = np.fft.fftfreq(N, 1 / Fs)
        return f[:N // 2], np.abs(S[:N // 2])

    f_in, S_in = compute_fft(x_ref, Fs_out)
    f_out, S_out = compute_fft(y, Fs_out)

    # -----------------------------
    # Plots
    # -----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Optimal Polyphase SRC 44.1 kHz → 48 kHz",
        fontsize=16
    )

    # Time domain (raw)
    axs[0, 0].plot(t_out[:2000], x_ref[:2000], label="Input (reference)", alpha=0.8)
    axs[0, 0].plot(t_out[:2000], y[:2000], "--", label="SRC output", alpha=0.8)
    axs[0, 0].set_title("Time domain (superposed)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Time domain (zoom)
    axs[0, 1].plot(t_out[:500], x_ref[:500], label="Input (reference)")
    axs[0, 1].plot(t_out[:500], y[:500], "--", label="SRC output")
    axs[0, 1].set_title("Time domain (zoom)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Spectrum (linear)
    axs[1, 0].plot(f_in, S_in / np.max(S_in), label="Input spectrum")
    axs[1, 0].plot(f_out, S_out / np.max(S_out), "--", label="Output spectrum")
    axs[1, 0].set_title("Spectrum (linear)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # Spectrum (dB)
    axs[1, 1].plot(
        f_in,
        20 * np.log10(S_in / np.max(S_in) + 1e-12),
        label="Input (dB)"
    )
    axs[1, 1].plot(
        f_out,
        20 * np.log10(S_out / np.max(S_out) + 1e-12),
        "--",
        label="Output (dB)"
    )
    axs[1, 1].set_title("Spectrum (dB)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_optimal_src_plots()
