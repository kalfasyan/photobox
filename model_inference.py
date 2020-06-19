import numpy as np
import pandas as pd
import soundfile as sf
import librosa 
import glob

B_ORDER = 4
L_CUTOFF = 120.
H_CUTOFF = 3600.
N_FFT = 256
SR = 8000.
HOP_LEN = int(N_FFT/6)

class WingbeatModelHandler(object):
    def __init__(self, name=None):
        self.name = 'swdmel_random_raw_conv1d_weights.h5'
    
    def load(self):
        from tensorflow.keras.models import load_model
        # Loading model in memory
        self.model = load_model('swdmel_random_raw_conv1d_weights.h5')

    def transform_data(self):
        pass

    def make_prediction(self):
        # self.model.predict(self.data)
        pass

def inference(X_names=glob.glob('wingbeats/*.wav')):
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.svm import OneClassSVM
    from scipy.stats import zscore

    # Loading model in memory
    model = load_model('swdmel_random_raw_conv1d_weights.h5')

    # Assigning labels and encoding them in numbers
    labels = pd.Series(X_names).apply(lambda x: x.split('_')[0])
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Transforming wingbeats into an array
    X_raw = make_array_parallel(names=X_names, setting='raw', expand_dim=True)
    pred = model.predict_on_batch(X_raw).numpy()

    # Transforming wingbeats into PSD values
    X_psd_dB = make_array_parallel(names=X_names, setting='psd_dB', expand_dim=False)
    # Using One-class SVM to find outliers
    svm = OneClassSVM(gamma='scale').fit(X_psd_dB)
    svm_score = svm.score_samples(X_psd_dB)

    df = pd.DataFrame(pred, columns=['mel','swd'])
    df['prediction'] = df[['mel','swd']].idxmax(axis=1)
    df['names'] = X_names
    df['true'] = y
    df['labels'] = labels
    df['svm_score'] = svm_score#(svm_score-min(svm_score))/(max(svm_score)-min(svm_score))
    df['svm_zscored'] = zscore(df['svm_score'])
    df['selection'] = df['svm_zscored'] > 0.
    df.loc[df.selection == False, 'prediction'] = 'unknown'
    return df['prediction'].tolist()

def read_raw(path):
    assert isinstance(path, str), "Pass a path as a string."
    wavdata, _ = sf.read(path)
    wavseries = pd.Series(wavdata)
    return wavseries.tolist()

def read_psd_dB(path):
    assert isinstance(path, str), "Pass a path as a string."    
    from scipy import signal
    x, _ = sf.read(path)
    x = 10*np.log10(signal.welch(x.ravel(), fs=SR, window='hanning', nperseg=256, noverlap=128+64)[1])
    return pd.Series(x).tolist()

def read_stft(path):
    import librosa
    assert isinstance(path, str), "Pass a path as a string."
    x, _ = sf.read(path)
    x = librosa.stft(x, n_fft = N_FFT, hop_length = HOP_LEN)
    x = librosa.amplitude_to_db(np.abs(x))
    x = np.flipud(x).flatten()
    return pd.Series(x).tolist()

def make_array_parallel(setting=None, names=None, expand_dim=True):
    import multiprocessing
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpus)
    result_list = []
    if setting == 'raw':
        result_list.append(pool.map(read_raw, names))
    elif setting == 'psd_dB':
        result_list.append(pool.map(read_psd_dB, names))
    elif setting == 'stft':
        result_list.append(pool.map(read_stft, names))
    else:
        logging.error('Wrong setting!')
    pool.close()
    if expand_dim:
        return np.expand_dims(np.vstack(result_list[0]), axis=-1)
    else:
        return np.vstack(result_list[0])

