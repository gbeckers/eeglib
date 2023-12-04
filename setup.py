import sys
import versioneer
import setuptools

if sys.version_info < (3,6):
    print("efys requires Python 3.6 or higher please upgrade")
    sys.exit(1)

long_description = \
"""
eeglib is a Python science library that enables you find auditory
events in eeg recordings based on calibration sounds.

"""

setuptools.setup(
    name='eeglib',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['eeglib'],
    url='https://github.com/gbeckers/EEG/eeglib',
    license='BSD-3',
    author='Gabriel J.L. Beckers',
    author_email='gabriel@gbeckers.nl',
    description='eeglib enables you find auditory events in eeg recordings '
                'based on calibration sounds',
    python_requires='>=3.6',
    install_requires=['darr','matplotlib','pandas','pyedflib', 'soundfile'],
    data_files = [("", ["LICENSE"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
    ],
    project_urls={  # Optional
        'Source': 'https://github.com/gbeckers/EEG/eeglib',
    },
)
