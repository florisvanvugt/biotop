# Physio Peak Picker


**This project has been migrated to https://github.com/florisvanvugt/woodpecker**

A simple GUI for human-assisted semiautomatic ECG analysis.

**Purpose** ECG analysis of real-world data can be tricky, especially when there are lots of artefacts.
Automated pipelines exist but the results can often not be inspected, and not manually adjusted.

The current script allows you to import, view and explore ECG data. 
You can run automated peak detection which you can then inspect and modify manually.
The results are saved in a JSON file format.

[![Video Tutorial](https://img.youtube.com/vi/o-oGjbLTjL4/0.jpg)](https://www.youtube.com/watch?v=o-oGjbLTjL4)




## Prerequisites

* Python 3.X 

This should install most of what you need:

```
pip3 install neurokit2 py-ecg-detectors matplotlib scipy numpy
python -m pip install "biobabel @ git+https://github.com/florisvanvugt/biobabel"
```

This also uses [HDPhysio5, a python-based library for the physiology HDF5 specification](https://github.com/florisvanvugt/hdphysio5), which will be loaded as a submodule.

```
git clone https://github.com/florisvanvugt/physio_peak_picker.git
```




## Usage

For ECG analysis:

```
python3 peak_picker.py
```

For respiration analysis:

```
python3 respiration_picker.py
```



### Basic GUI controls

* Hold down `shift` while moving the mouse : Snap to closest maximum
* Mouse scroll wheel up/down : Scroll back and forth in time
* `Ctrl` key + Mouse scroll wheel up/down : Zoom in/out in time
* Mouse left button double click : Insert peak (or zoom in if not zoomed in enough)
* Mouse right button single click : Remove peak
* Mouse middle button click : Insert marker for invalid region
* Mouse middle button double click : Remove invalid region
* Keyboard keys:
   * `z` toggles between micro and macro zoom (make sure the window has focus)
   * `a` shows the entire signal
   * Left/right arrow keys scroll through the signal slowly
   * PageDown/PageUp keys browse through the signal a full window at a time


## File format

This uses a custom file format based on the HDF5 framework, explained in `specification_hdf5.md`

