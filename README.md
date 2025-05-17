# biotop: picking peaks in physiological data 

[![DOI](https://zenodo.org/badge/618577784.svg)](https://zenodo.org/badge/latestdoi/618577784)

A simple GUI for human-assisted semiautomatic ECG and respiration preprocessing.

**Purpose** ECG analysis of real-world data can be tricky, especially when there are lots of artefacts.
Automated pipelines exist but the results can often not be inspected, and not manually adjusted.

The current script allows you to import, view and explore ECG and respiratory data. 
You can run automated peak detection which you can then inspect and modify manually.
The results are saved in a JSON file format.

[![Video Tutorial](https://img.youtube.com/vi/o-oGjbLTjL4/0.jpg)](https://www.youtube.com/watch?v=o-oGjbLTjL4)




## Prerequisites

* Python 3.X 

Prerequisite packages are installed automatically using the following:

```
python3 -m pip install biotop
```



## Usage

For ECG analysis:

```
biotop
```

For respiration analysis:

```
respire
```



### Basic GUI controls

* Mouse scroll wheel up/down : Scroll back and forth in time
* `Ctrl` key + Mouse scroll wheel up/down : Zoom in/out in time
* Mouse left button double click : Insert peak (or zoom in if not zoomed in enough)
* Hold down `shift` while moving the mouse : Snap to closest maximum
* Mouse right button single click : Remove peak
* Mouse middle button click : Insert marker for invalid region
* Mouse middle button double click : Remove invalid region
* Keyboard keys:
   * `z` toggles between micro and macro zoom (make sure the window has focus)
   * `a` shows the entire signal
   * Left/right arrow keys scroll through the signal slowly
   * PageDown/PageUp keys browse through the signal a full window at a time


For respire (the respiration picker script)
* Hold down `ctrl` while moving the mouse : Snap to closest *minimum* (handy for selecting troughs)



# Development

Install latest development version:

```
pip install --upgrade "biotop @ git+https://github.com/florisvanvugt/biotop"
```



## Wishlist

- [x] Show a window when waiting for peak autodetection to complete
- [ ] Make sure y axis labels remain visible even for greater zoom
- [ ] Merge the different scripts into a single one with multiple modes (ECG, PPG, respiration).

- [ ] Bug fixing: sometimes artefact intervals *are* overlapping whereas they should be S01-tapping3 Cardiotap.




# Citation

If you use this work, please add the appropriate citation:

```
van Vugt, F.T. (2025). Biotop: Graphical user interface to combine automated and manual analysis of physiological signals. https://doi.org/10.5281/zenodo.15453370
```

