eeglib
======

Eeglib a is library with useful code for EEG work in our lab.

It keeps code and insights that can be used across projects in one 
maintainable package. This avoids reinventing the wheel, and allows for 
incremental improvements to be shared.


Install
-------
Make sure you have cloned the GitHub EEG repo that includes eeglib to the computer with a functioning conda/MNE install. Make sure you are in the conda environment you use for EEG work (MNE). Go to the top `eeglib` folder in the cloned repo and type:

```
$ pip install -e .
``` 

Requirements
------------
Eeglib depends on the following Python libraries:

* numpy
* pyedflib
* pandas
* matplotlib
* darr

However, these are installed automatically if you use the 'pip' method to install eeglib above.


Licence
-------
This small library is open source and can be used and adapted by anyone. See 
LICENSE file.
    

Release Notes
-------------
*version 0.2.4*
- improve function to convert stimulus table to MNE events

*version 0.2.3*
- fixed x-axis bug in plot calibmarks (did not affect actual timing in stimulus tables)
- fixed order epoch axis in bitsound image plots

*version 0.2.2*
- More versatile calibmark finding. Can account for audio polarity switched. And more informative figures.
- pandas to mne events 

