from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


filename = "/mnt/pccfs/backed_up/andrew/hearables/clean/nwords_{}.wav".format(np.random.randint(1, 333))

def get_split(zxx):
    result = np.zeros((2, *zxx.shape))
    result[0] = zxx.real
    result[1] = zxx.imag
    return result    

def get_stft(filename, split=False, plot=False):
    """reads wav file, calculates the short time fourier transform. Returns triple of sample frequencies,
    times, and the complex valued matrix zxx which is the stft of the input file time series"""
    sample_rate, samples = wavfile.read(filename)


    f, t, zxx = signal.stft(samples)
    if plot:
        plot_spect(f, t, zxx, filename=filename)

    if split:
        zxx = get_split(zxx)

    return f, t, zxx, sample_rate

def get_istft(zxx, sample_rate, title="output", save=False):
    if len(zxx) == 3:
        real, imag = zxx[0], zxx[1]
        zxx = real + imag
    t, x = signal.istft(zxx)

    # if you remove these lines, your ears will bleed
    m = np.max(np.abs(x))
    x = x/m

    if save:
        wavfile.write("{}{}.wav".format(title, np.random.randint(0,1000000)), sample_rate, x)

    return t, x


def plot_spect(f, t, Zxx, filename=""):
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=30)
    plt.title('STFT Magnitude\n{}'.format(filename.split("/")[-1]))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def sparsely_observe(Zxx, context_points=1000):
    mask = np.zeros_like(Zxx)
    
    n,m = mask.shape
    mask = mask.reshape(-1)

    mask[:context_points] = 1
    np.random.shuffle(mask)

    mask = mask.reshape(n,m)

    Zxx[mask != 1] = 0

    return Zxx

if __name__ == "__main__":
    f, t, zxx, sample_rate = get_stft(filename, plot=True)
    sample_rate, samples = wavfile.read(filename)
    # zxx = sparsely_observe(zxx)
    # print(get_istft(zxx, sample_rate, save=True))
