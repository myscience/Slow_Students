def bandpass_filter(data): # data has to be a 3D array

    from scipy.signal import butter, lfilter
    from numpy import reshape

    # Filter requirements.
    order = 6     # order of the Butterworth filter
    fs = 25.0       # sample rate, Hz
    lowcut = 0.3  # desired cutoff frequency of the high-pass filter, Hz
    highcut = 2.5  # desired cutoff frequency of the low-pass filter, Hz

    #Definition of the Butterworth bandpass filter

    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y 

    #Application of the filter to the dataset

    clean_signals = []

    for i in range (len(data[1])):
        for j in range (len(data[2])):
            clean_signals.append([butter_bandpass_filter(data[:, i, j], lowcut, highcut, fs, order)])

    clean_signals = reshape(clean_signals, (34, 34, 1000))

    return clean_signals
