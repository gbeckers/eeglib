eeglib
======

Eeglib a is library with useful code for EEG work in our lab. It is unlikely 
to be useful outside our lab.

Eeglib keeps code and insights that can be used across projects in one 
maintainable package. This avoids reinventing the wheel, and allows for 
incremental improvements to be shared.


Install
-------
In the correct environment, type in a terminal:
```
$ pip install git+https://github.com/gbeckers/eeglib@master
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

