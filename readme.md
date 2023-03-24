# Physio Peak Picker

A simple GUI for human-assisted semiautomatic ECG analysis.

## Prerequisites

* Python 3.X 

This should install most of what you need:

```
pip3 install neurokit2 ecg-h5py py-ecg-detectors matplotlib scipy numpy
```



## Usage

```
python3 peak_picker.py
```



Basic GUI controls

* Mouse scroll wheel up/down : Scroll back and forth in time
* Ctrl key + Mouse scroll wheel up/down : Zoom in/out in time
* Mouse left button double click : Insert peak (or zoom in if not zoomed in enough)
* Mouse right button single click : Remove peak
* Mouse middle button click : Insert marker for invalid region
* Mouse middle button double click : Remove invalid region

