import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
from darr import DataDir, create_datadir
from pathlib import Path

import sys

from . import edf
from . import openbci

__all__ = ['create_recordingeventsinfobiosemi','create_recordingeventsinfoopenbci']

def find_calibmarks(snd, snd_fs, calibmark, calibmark_fs, searchduration=30.,
                    recordedasbit=False, bitthreshold=0.005,
                    correct_ones=None):
    """Finds calibration stimuli at beginning and end of a longer sound (usually
    a recordingdata of stimulus playback).

    Parameters
    ----------
    snd: UniformTimeSeries
        Signal in which the calibmarks should be detected
    calibmark: UniformTimeSeries
        Calibmark stimulus
    searchduration: float
        Duration of episode in beginning and end of `snd` in which calibmark will be searched for.
    recordedasbit: bool
        Is snd a bitsnd (i.e. just 0 and 1 values)?
    bitthreshold: float
        Threshold above which sample values of calibmark will be considered 1. Rest is 0.
    correct_ones: int | None
        If not None, indicates how long a sequence of ones should be in order to set them to 0.

    Returns
    -------
    (starttime1, starttime2)

    """
    calibmark = calibmark.astype('float64')
    if snd_fs != calibmark_fs: # need to resample
        num = int(round((snd_fs / calibmark_fs) * len(calibmark)))
        calibmark = resample(x=calibmark, num=num, t=None, axis=0, window=None)
    # first calibmark
    searchnframes = int(searchduration * snd_fs)
    if recordedasbit:
        calibmark = (calibmark > bitthreshold).astype('float64')
    target1 = snd[:searchnframes].astype('float64')
    if correct_ones is not None:
        a = np.correlate(target1, np.ones(correct_ones), 'same')
        target1[a == correct_ones] = 0.
    cc = np.correlate(target1, calibmark, mode='valid')
    m1 = cc.max()
    hits = np.where(cc==m1)[0]
    if not (np.diff(hits)==1).all():
        print('warning: x-corr calibmark 1 has non-contiguous max')
    r1 = int(round(hits.mean())) # adjust for multiple hits
    target2 = snd[-searchnframes:].astype('float64')
    #if append_zeros is not None:
    #    target2 = np.concatenate([target2, np.zeros(append_zeros, dtype='float64')] )
    if correct_ones is not None:
        a = np.correlate(target2, np.ones(correct_ones), 'same')
        target2[a == correct_ones] = 0.
    cc = np.correlate(target2, calibmark, mode='valid')
    m2 = cc.max()
    hits =  np.where(cc==m2)[0]
    if not (np.diff(hits)==1).all():
        print('warning: x-corr calibmark 2 has non-contiguous max')
    r2 = int(round(hits.mean())) # adjust for multiple hits
    r2 = r2 + len(snd) - searchnframes
    return np.array((r1, r2))/snd_fs


def convert_stimulustable(audiostimulustable, starttimefirst, starttimelast, newfs):
    """Convert audio stimulus table in a recordingdata event table, if you know the
    recordingdata starttimes of the first and the last sound event in the table.

    This can be used if there are no calibmarks to be automatically found, but you do
    have an idea of where they are.

    """

    st = audiostimulustable.copy()
    factor = (starttimelast - starttimefirst) / (st.iloc[-1]['starttime'] - st.iloc[0]['starttime'])
    offset = starttimefirst
    st['starttime'] = offset + factor * st['starttime']
    st['endtime'] = offset + factor * st['endtime']
    st['startframe'] = np.round(st['starttime'] * newfs).astype('int64')
    st['endframe'] = np.round(st['endtime'] * newfs).astype('int64')
    params = {'offset': offset, 'scalingfactor': factor}
    return st, params


# TODO refactor code to make it more readable
def create_recordingeventtable(recsnd, recsnd_fs, playbacksnd, playbacksnd_fs,
                               audiostimulustable, recordedasbit=False,
                               searchduration=30., bitthreshold=0.005,
                               checkcalibmarks=False, correct_ones=None,
                               append_zeros=None):
    """Create a stimulus timing table of recordingdata based on calibration sounds.

    Parameters
    ----------
    recsnd
    recsnd_fs
    playbacksnd
    playbacksnd_fs
    audiostimulustable
    recordedasbit
    searchduration
    bitthreshold
    checkcalibmarks

    Returns
    -------
    st, params, (fig1, fig2)
    """

    st = audiostimulustable.copy()
    for i, pos in zip((0,-1),('first', 'last')):
        if not st.iloc[i]['snd'] in ('calibmark'):
            raise ValueError(f'{pos} row of playback stimulus table does not '
                             f'contain calibmark')
    startframe = int(round(st.iloc[0]['starttime'] * playbacksnd_fs))
    endframe = int(round(st.iloc[0]['endtime'] * playbacksnd_fs))
    calibmark = playbacksnd[startframe:endframe]
    # t1, t2 are the start times of the calibmarks
    if append_zeros is not None:
        recsnd = np.concatenate([recsnd, np.zeros(append_zeros, dtype=recsnd.dtype)])
    t1,t2 = find_calibmarks(snd=recsnd, snd_fs=recsnd_fs, calibmark=calibmark,
                            calibmark_fs=playbacksnd_fs,
                            searchduration=searchduration,
                            recordedasbit=recordedasbit,
                            bitthreshold=bitthreshold, correct_ones=correct_ones)
    # calc scaling factor because of deviation sample rates playback and recoring device clocks
    factor = (t2 - t1) / (st.iloc[-1]['starttime'] - st.iloc[0]['starttime'])
    # calc offset because recording did not start at start of playback stimuli
    offset = t1
    st['starttime'] = offset + factor * st['starttime']
    st['endtime'] = offset + factor * st['endtime']
    st['startframe'] = np.round(st['starttime'] * recsnd_fs).astype('int64')
    st['endframe'] = np.round(st['endtime'] * recsnd_fs).astype('int64')
    params = {'offset': offset, 'scalingfactor': factor}

    fig1 = fig2 = fig3 = None
    if checkcalibmarks:
        if recsnd_fs != playbacksnd_fs:  # need to resample
            num = int(round((recsnd_fs / (playbacksnd_fs*factor)) * len(playbacksnd)))
            playbacksnd = resample(x=playbacksnd, num=num, t=None, axis=0, window=None)
        calibmdur = st.iloc[0]['endtime'] - st.iloc[0]['starttime']
        margin = int(round(calibmdur*0.05*recsnd_fs)) # in frames
        margin = min(margin, len(recsnd) - st.iloc[-1]['endframe'])
        detaildur = calibmdur / 6
        detaillen = int(round(detaildur*recsnd_fs))
        detailmargin = detaillen // 10
        calibm1 = recsnd[st.iloc[0]['startframe'] - margin:st.iloc[0]['endframe'] + margin]
        calibm2 = recsnd[st.iloc[-1]['startframe'] - margin:st.iloc[-1]['endframe'] + margin]
        calibm1sel = recsnd[st.iloc[0]['startframe'] - detailmargin:st.iloc[0]['startframe'] + detaillen]
        calibm2sel = recsnd[st.iloc[-1]['startframe'] - detailmargin:st.iloc[-1]['startframe'] + detaillen]
        fig1 = plt.figure(figsize=(14,6))
        plt.subplot(4, 1, 1)
        c1samplingtimes = st.iloc[0]['starttime'] - margin/recsnd_fs + np.arange(len(calibm1), dtype='float64') / recsnd_fs
        plt.plot(c1samplingtimes, calibm1)
        plt.title("Calibmarks")
        plt.subplot(4, 1, 2)
        c1selsamplingtimes = st.iloc[0]['starttime'] - detailmargin/recsnd_fs + np.arange(len(calibm1sel), dtype='float64') / recsnd_fs
        # plot bitsound calibmark1
        plt.plot(c1selsamplingtimes, calibm1sel)
        # plot playback sound calibmark1
        plt.plot(st.iloc[0]['starttime'] + np.arange(len(calibm1sel)) / recsnd_fs, playbacksnd[:len(calibm1sel)])
        plt.subplot(4, 1, 3)
        c2samplingtimes = st.iloc[-1]['starttime'] - margin/recsnd_fs + np.arange(len(calibm2), dtype='float64') / recsnd_fs
        plt.plot(c2samplingtimes, calibm2)
        plt.plot(st.iloc[-1]['starttime'] + np.arange(len(calibm2)) / recsnd_fs, playbacksnd[:len(calibm2)])
        plt.subplot(4, 1, 4)
        c2selsamplingtimes = st.iloc[-1]['starttime'] - detailmargin/recsnd_fs + np.arange(len(calibm2sel), dtype='float64') / recsnd_fs
        plt.plot(c2selsamplingtimes, calibm2sel)
        plt.plot(st.iloc[-1]['starttime'] + np.arange(len(calibm2sel)) / recsnd_fs, playbacksnd[:len(calibm2sel)])
        plt.xlabel('Recording time (s)')
        # now image plots
        duration = 0.2
        preduration = 0.1
        nframes = int((round((duration + preduration) * recsnd_fs)))
        preframes = int(round(preduration*recsnd_fs))
        euts1 = np.empty((len(st), nframes), 'float64')
        for row, startframe_r in enumerate(st['startframe']):
            euts1[row,:] = recsnd[startframe_r-preframes:startframe_r-preframes+nframes]
        fig2 = plt.figure(figsize=(14, 14))
        plt.imshow(euts1, cmap='hot', interpolation='nearest',
               extent=[-preduration, duration, len(st), 0], aspect='auto')
        plt.plot([0,0],[0,len(st)], '#ff002299')
        plt.xlabel('Time re event (s)')
        plt.ylabel('Events')
        plt.title('Stimulus alignment')
        nframes = int((round((duration + preduration) * recsnd_fs)))
        preframes = int(round(preduration * recsnd_fs))
        euts2 = np.empty((len(st), nframes), 'float64')
        for row, starttime_a in enumerate(audiostimulustable.iloc[1:-1]['starttime']):
            startframe_a = int(round(starttime_a*recsnd_fs/factor))
            stim = playbacksnd[startframe_a - preframes:startframe_a - preframes + nframes]
            if recordedasbit:
                stim = (stim > 0.08).astype('float64')
            euts2[row+1, :] = stim
        # calibmarks
        for index in (0,-1):
            stt = audiostimulustable.iloc[index]['starttime']
            startframe_a = int(round(stt * recsnd_fs / factor))
            stim = playbacksnd[startframe_a:startframe_a - preframes + nframes]
            if recordedasbit:
                stim = (stim > 0.08).astype('float64')
            euts2[index, preframes:] = stim
        fig3 = plt.figure(figsize=(14, 14))
        plt.imshow(euts2, cmap='hot', interpolation='nearest',
                   extent=[-preduration, duration, len(st), 0], aspect='auto')
        plt.plot([0, 0], [0, len(st)], '#ff002299')
        plt.xlabel('Time re event (s)')
        plt.ylabel('Events')
        plt.title('Stimulus alignment original')

    return st, params, (fig1, fig2, fig3)


def create_recordingeventsinfobiosemi(edfpath, audiostimulustablepath, audiowavpath, outputpath=None,
                                      searchduration=30., bitthreshold=0.005, reverse_polarity=False,
                                      append_zeros=None):
    """Creates information on sound stimulus occurrence in recordingdata.

    The information is generated based on a trace of the audio playback, a provided stimulus table
    and an playback audio file. The information is saved in several files in `outputpath`.

    Parameters
    ----------
    edfpath: path
    audiostimulustablepath: path
      Path to csv file with platback stimulus info
    audiowavpath
    outputpath
    searchduration
    bitthreshold

    Returns
    -------
    Pandas DataFrame recordingdata stimulus table

    """
    overwrite = True
    audiochannel = 'Status'
    recordedasbit = True
    checkcalibmarks = True

    edfpath = Path(edfpath)
    if outputpath is None:
        outputpath = edfpath.with_name(f"{edfpath.with_suffix('').name}_stimulusinfo")
    else:
        outputpath = Path(outputpath)
    pst = pd.read_csv(audiostimulustablepath)
    ar, metadata = edf.load_edfasarray(str(edfpath),
                                           channels=[audiochannel],
                                           dtype='int16')
    bitsnd = ar[:,0]
    bitsnd -= bitsnd.min()
    bitsnd[bitsnd==bitsnd.max()] = 1
    bitsnd_fs = metadata['fs']
    pbs_fs, playbacksnd = wavfile.read(filename=str(audiowavpath))
    if reverse_polarity:
        playbacksnd = -playbacksnd
    st, params, (fig1, fig2, fig3) = create_recordingeventtable(recsnd=bitsnd,
                                                                recsnd_fs=bitsnd_fs,
                                                                playbacksnd=playbacksnd,
                                                                playbacksnd_fs=pbs_fs,
                                                                audiostimulustable=pst,
                                                                recordedasbit=recordedasbit,
                                                                searchduration=searchduration,
                                                                bitthreshold=bitthreshold,
                                                                checkcalibmarks=checkcalibmarks,
                                                                append_zeros=append_zeros)
    if not outputpath.exists():
        dd = create_datadir(outputpath)
    else:
        dd = DataDir(outputpath)
    dd.write_jsondict('timealignmentparameters.json',params, overwrite=overwrite)
    st.to_csv(dd.path/'recordingstimulustable.csv', index=False)
    pst.to_csv(dd.path / 'playbackstimulustable.csv', index=False)
    eventdict, eventtable = stimulustabletoevents(st)
    dd.write_jsondict('mne_eventdict.json', eventdict, overwrite=overwrite)
    eventtable.to_csv(dd.path / 'mne_eventtable.eve', index=False, header=False, sep='\t')
    fig1.savefig(dd.path / 'calibmarks.png', dpi=300)
    fig2.savefig(dd.path / 'snd_epochs.png', dpi=300)
    return st

# TODO de-uts-ify
def create_recordingeventsinfoopenbci(datafilepath, audiostimulustablepath,
                                      audiowavpath, outputpath=None, searchduration=30.,
                                      bitthreshold=0.005):
    import uts

    """Creates information on sound stimulus occurrence in recordingdata.

    The information is generated based on a trace of the audio playback, a provided stimulus table
    and an playback audio file. The information is saved in several files in `outputpath`.

    Parameters
    ----------
    edfpath: path
    audiostimulustablepath: path
      Path to csv file with platback stimulus info
    audiowavpath
    outputpath
    searchduration
    bitthreshold

    Returns
    -------
    Pandas DataFrame recordingdata stimulus table

    """
    overwrite = True
    datafilepath = Path(datafilepath)
    with open(datafilepath, 'r') as f:
        firstline = f.readline()
        if "OpenBCI Raw EEG Data" in firstline:
            audiochannel = 'other03'
            bitsnd = openbci.load_openbcidata(filepath=datafilepath)[:, audiochannel]
        else:
            audiochannel = 'accel_1'
            sampleindex, eeg, accel = openbci.load_thinkpulsedata(filepath=datafilepath)
            bitsnd = accel[:,[audiochannel]]
            bitsnd.samples.array[bitsnd.samples.array == 256] = 1
    recordedasbit = True
    checkcalibmarks = True
    if outputpath is None:
        outputpath = datafilepath.with_name(f"{datafilepath.with_suffix('').name}_stimulusinfo")
    else:
        outputpath = Path(outputpath)
    pst = pd.read_csv(audiostimulustablepath)

    fs, data = wavfile.read(filename=str(audiowavpath))
    playbacksnd = uts.UniformTimeSeries(samples=data, fs=float(fs))
    st, params, (fig1, fig2) = create_recordingeventtable(recsnd=bitsnd,
                                                          audiostimulustable=pst,
                                                          playbacksnd=playbacksnd,
                                                          recordedasbit=recordedasbit,
                                                          searchduration=searchduration,
                                                          bitthreshold=bitthreshold,
                                                          checkcalibmarks=checkcalibmarks)
    if not outputpath.exists():
        dd = create_datadir(outputpath)
    else:
        dd = DataDir(outputpath)
    dd.write_jsondict('timealignmentparameters.json',params, overwrite=overwrite)
    st.to_csv(dd.path/'recordingstimulustable.csv', index=False)
    pst.to_csv(dd.path / 'playbackstimulustable.csv', index=False)
    eventdict, eventtable = stimulustabletoevents(st)
    dd.write_jsondict('mne_eventdict.json', eventdict, overwrite=overwrite)
    eventtable.to_csv(dd.path / 'mne_eventtable.csv', index=False, header=False, sep='\t')
    fig1.savefig(dd.path / 'calibmarks.png', dpi=300)
    fig2.savefig(dd.path / 'snd_epochs.png', dpi=300)
    return st



def stimulustabletoevents(st):
    """Converts a stimulus table (Pandas DataFrame) to an event table and event dictionary for MNE"""
    # TODO, this should allow for hierarchical events
    eventlabels = st['snd'].unique()
    eventdict = {l: i for i,l in enumerate(eventlabels)}
    eventtable = st[['startframe','endframe']].copy()
    eventtable['endframe'] = 0
    eventtable['events'] = st.apply(lambda x: eventdict[x['snd']], axis=1)
    return eventdict, eventtable
