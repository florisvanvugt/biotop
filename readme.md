# Physio Peak Picker

A simple GUI for human-assisted semiautomatic ECG analysis.

**Purpose** ECG analysis of real-world data can be tricky, especially when there are lots of artefacts.
Automated pipelines exist but the results can often not be inspected, and not manually adjusted.

The current script allows you to import, view and explore ECG data. 
You can run automated peak detection which you can then inspect and modify manually.
The results are saved in a JSON file format.


 <iframe width="420" height="315"
src="https://youtu.be/o-oGjbLTjL4">
</iframe> 




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

* Hold down `shift` while moving the mouse : Snap to closest maximum
* Mouse scroll wheel up/down : Scroll back and forth in time
* `Ctrl` key + Mouse scroll wheel up/down : Zoom in/out in time
* Mouse left button double click : Insert peak (or zoom in if not zoomed in enough)
* Mouse right button single click : Remove peak
* Mouse middle button click : Insert marker for invalid region
* Mouse middle button double click : Remove invalid region



## File format

This uses a custom file format based on the HDF5 framework, explained in `specification_hdf5.md`

