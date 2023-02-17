import numpy as np
import pyedflib

# TODO unfortunately we need to switch to EDFlib-Python: https://gitlab.com/Teuniz/EDFlib-Python/-/tree/master
# other edf readers do not read header info well, which is a huge problem because of patient name

__all__ = ['read_edfinfo', 'load_edfasarray', ]

def read_edfinfo(filepath):
    with pyedflib.EdfReader(str(filepath)) as f:
        return {
            'admincode': f.getAdmincode(),
            'birthdate': f.getBirthdate(),
            'digitalmaximum': f.getDigitalMaximum(),
            'digitalminimum': f.getDigitalMinimum(),
            'fileduration': f.getFileDuration(),
            'gender': f.getGender(),
            'equipment': f.getEquipment(),
            'nsamples': f.getNSamples(),
            'nsignals': f.signals_in_file,
            'patientadditional': f.getPatientAdditional(),
            'patientcode': f.getPatientCode(),
            'patientname': f.getPatientName(),
            'samplefrequencies': f.getSampleFrequencies(),
            'startdatetime': f.getStartdatetime().isoformat(),
            'signallabels': f.getSignalLabels(),
            'signalheaders': f.getSignalHeaders(),
            'technician': f.getTechnician(),
            'recordingadditional': f.getRecordingAdditional()
        }


def load_edfasarray(edffilepath, dtype='float32', channels=None,
                   nanchannels=None, digital=False):
    """Reads a edf/bdf file and returns it as a numpy array and metadata.

    Note that all data is read into RAM at once.

    Parameters
    ----------
    edffilepath
    dtype
    channels
    nanchannels
    geometry
    digital

    Returns
    -------
    array, fs
        numpy array and sampling rate


    """

    metadata = read_edfinfo(filepath=edffilepath)
    if channels is not None:
        chindices = [metadata['signallabels'].index(ch) for ch in channels]
    else:
        channels = metadata['signallabels']
        chindices = [i for i in range(len(channels))]
    nframes = metadata['nsamples'][chindices]
    fss = metadata['samplefrequencies'][chindices]
    if not all(nframes[0] == nframes):
        raise ValueError(f'Not all channel lengths are the same: {nframes} for'
                         f'{channels}.')
    nframes = nframes[0]
    if not all(fss[0] == fss):
        raise ValueError(f'Not all sampling frequencies are the same: {fss} for'
                         f'{channels}.')
    fs = fss[0]
    nchannels = len(chindices)
    for key in ['samplefrequencies', 'nsamples', 'signallabels', 'signalheaders']:
        metadata[key] = [metadata[key][i] for i in chindices]
    metadata['nsignals'] = len(channels)
    metadata['fs'] = float(fs)
    if nanchannels is not None:
        nchannels += len(nanchannels)
        channels += nanchannels
    sigbufs = np.zeros((nframes,nchannels), dtype=dtype)
    if nanchannels is not None:
        sigbufs[:,-len(nanchannels):] = np.nan
    with pyedflib.EdfReader(edffilepath) as f:
        for i,j in enumerate(chindices):
            sigbufs[:,i] = f.readSignal(j, digital=digital)
    return sigbufs, metadata

