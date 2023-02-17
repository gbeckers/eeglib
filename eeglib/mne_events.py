import numpy as np
import pandas as pd

__all__ = ['stimulustabletoevents']


def stimulustabletoevents(st, stimcolumn, fs, event_id=None):
    """Converts a stimulus table (Pandas DataFrame) to an event table and event dictionary for MNE

    Parameters
    ----------
    stimcolumn: str
        Which column contains the stimulus labels?
    fs: float
        Sampling rate of EEG recording
    event_id: dict
        event_id dictionary to use for selection of stimulus events. This is handy to keep
        event_id numbers the same across recordings, and/or to select a subset of events.

    Returns
    -------
    (event_id, eventtable): (dict, pandas DataFrame)

    """
    # TODO, this should allow for hierarchical events
    st = st.copy()
    st['startframe'] = np.int64(np.round(st['starttime'].values * fs))
    st['endframe'] = np.int64(np.round(st['endtime'].values * fs))
    if event_id is None:
        eventlabels = st[stimcolumn].unique()
        eventdict = {l: i for i, l in enumerate(eventlabels)}
    else:
        eventlabels = sorted(list(event_id.keys()))
        eventdict = event_id
    st = st[st[stimcolumn].isin(eventlabels)]
    eventtable = st[['startframe']].copy()
    eventtable['val'] = 0
    eventtable['events'] = st.apply(lambda x: eventdict[x[stimcolumn]], axis=1)
    return eventdict, eventtable.values