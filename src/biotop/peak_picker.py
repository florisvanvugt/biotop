


import tkinter
from tkinter import filedialog as fd
from tkinter import font as tkFont  # for convenience
from tkinter import Toplevel, Menu, StringVar, Label, Frame, BooleanVar
import tkinter.messagebox
from tkinter.messagebox import askyesno

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import biotop.preprocess_ecg_detectors as preprocess_ecg

import numpy as np
import pandas as pd
import os
import scipy.signal
import neurokit2
import heartpy

import json

import sys

from biotop.misc import does_overlap, in_range

import biotop.misc as misc


# Globals we carry around
gb = {}







def do_auto_detect_peaks():
 
    ## First clear the peaks in the current window
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    tmin = max([0,tmin]) # don't go below zero
    gb['peaks'] = [ p for p in gb['peaks']
                    if p['t']<tmin or p['t']>tmax ]
    #print(tmin,tmax)
    
    ## Now, find the valid portions of signal in the current window.
    ranges = find_valid_between(tmin,tmax)

    ## Find the time corresponding to the beginning of the 
    t = gb['t']

    for (fromt,tot) in ranges:

        ## Take the chunk of data in the current window
        #ecg_target = gb['ecg-prep-column']
        tsels = (t>=fromt) & (t<=tot)
        tmin,tmax=min(t[tsels]),max(t[tsels]) # this can differ from the window edges (if we are at the end or beginning of the signal)
        samp_min = int(round(fromt*gb['SR'])) # calculate what sample the starting point corresponds to
        ecg = gb['signal'][tsels]

        peaks = preprocess_ecg.peak_detect(ecg,gb['SR'],gb['detector'].get())
        gb['peaks'] += [
            {
                'i':samp+samp_min,
                't':(samp+samp_min)/gb['SR'],
                'valid':True,
                'source':'auto',
                'edited':False,
                'y':ecg[samp]
            } for samp in peaks
        ]



from biotop.misc import chop_away

        
def find_valid_between(tmin,tmax):
    # Find the valid portions in the interval (tmin,tmax).
    # That is, return a set of intervals (a,b) that cover the
    # valid (i.e. not marked as invalid) between tmin and tmax.

    # Start with the optimistic idea that the whole range is valid
    valids = [ (tmin,tmax) ]

    # Now let's chip away the portions that are marked as invalid
    for s,t0,t1 in gb['invalid']:
        if gb['channel']==s:

            # See if this overlap with any of the valid portions
            valids = chop_away(t0,t1,valids)

    return valids





def clear_artefacts_here():
    """ Clear all markings of artefactual regions
    in the time range currently visible in the window."""

    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    gb['invalid'] = [ (s,t0,t1) for (s,t0,t1) in gb['invalid']
                      if s!=gb['channel'] or not does_overlap((t0,t1),drawrange) ]
                      
    redraw_all()



def next_artefact():
    """ Move the window to show the next artefact region. """
    endt = gb['tstart']+gb['WINDOW_T']
    invalt = [ t for (s,t,_) in gb['invalid'] if s==gb['channel'] and t>endt ]
    if len(invalt):
        nextt = np.min(invalt)
        move_window_to(nextt) # move that to the center of the screen
        
    
    


def clear_peaks():
    if len(gb['peaks']):
        answer = askyesno(
            title='confirmation',
            message='This will clear any peaks you have modified or detected.\nProceed?')
    else:
        answer = True
    if answer:
        
        gb['peaks']=[] # remove everything!!
        redraw_all()



        
def clear_peaks_here():
    """
    Clear peaks in the time range currently visible in the window
    """
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    gb['peaks'] = [ p for p in gb['peaks']
                    if p['t']<tmin or p['t']>tmax ]
    redraw_all()





def create_progress_window():
    # Create a toplevel window
    window = Toplevel()
    window.wm_title("Please wait...")
    window.geometry('{}x{}'.format(450,300))
    frame=Frame(window)
    frame.grid(row=0, column=0, sticky="NW")
    win_label = Label(window,text='Detecting peaks. This may take some time...\nThis window will close automatically when completed.')
    win_label.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
    #win_label.pack()
    return window
    

    
def auto_detect_peaks():
    
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    peaks_here = [ p for p in gb['peaks'] if p['t']>=tmin and p['t']<=tmax ]
    
    if len(peaks_here):
        answer = askyesno(
            title='confirmation',
            message='Auto detecting peaks will clear any peaks may you have edited or added.\nDo you want to proceed?')
    else:
        answer = True
    if answer:

        win = create_progress_window()
        def run_analysis():
            do_auto_detect_peaks()
            redraw_all()
            win.destroy()
        win.after(500, run_analysis)



def do_import_biopac_peaks(f):
    try:
        import read_biopac
        peakdata = read_biopac.read_acq_ecg_peaks(f)
        ts = peakdata['Time']
        gb['peaks']= [
            {
                'i':int(round(t*gb['SR'])),
                't':t,
                'valid':True,
                'source':'auto',
                'edited':False
            } for t in ts
        ]
        for p in gb['peaks']: p['y']=ecg[p['i']] if p['i']<ecg.shape[0] else np.nan
    except:
        print("Something didn't work.")

    

        
def import_biopac_peaks():
    if len(gb['peaks']):
        answer = askyesno(
            title='confirmation',
            message='Importing Biopac peaks will clear any peaks you may have edited or added.\nDo you want to proceed?')
    else:
        answer = True
    if answer:
        filetypes = (
            ('Biopac-peak data files (excel)', '*.xls'),
            ('All files', '*.*')
        )
        
        fname = fd.askopenfilename(
            title='Select your peaks',
            initialdir='.',
            filetypes=filetypes)

        if fname:
            do_import_biopac_peaks(fname)
            
        redraw_all()







MIN_INVALID_DUR = .1 # minimum size for an "invalid" portion

def curate_invalid(inv):
    toret = []
    for (s,t0,t1) in inv:
        dt = abs(t0-t1)
        if dt<MIN_INVALID_DUR: continue         ## Too short to be plausible
        if isinstance(s,list): continue         ## This was due to a bug previously. s should be a string, the name of the channel
        toret.append( (s,t0,t1) )
    return toret

    


def get_signal_at_t(t):
    # Return the ECG signal value closest to time t
    if not t: return None
    samp = int(round(t*gb['SR'])) # getting the closest sample
    #ecg_target = gb['ecg-prep-column']
    #ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    return gb['signal'][samp]

        
        
    

PEAK_SNAP_T = .1 # how close in time do we need to be to a peak to disable it

PEAK_EDIT_MAX_WINDOW_T = 2 # the maximum window size that allows peak adjusting/adding.
# If the window is too far zoomed out, we can't trust the accuracy of peak editing



def find_closest_peak(t):
    dts = [ t-peak['t'] for peak in gb['peaks'] ]
    if not len(dts): return None,{"t":np.Inf}
    min_dt = min([ abs(d) for d in dts ])
    for i,peak in enumerate(gb['peaks']):
        dt = t-peak['t']
        if abs(dt)==min_dt:
            return i,peak


def check_window_zoom(t):

    if gb['WINDOW_T']>PEAK_EDIT_MAX_WINDOW_T:

        # First zoom in
        update_window(.8*PEAK_EDIT_MAX_WINDOW_T/gb['WINDOW_T'],t)
        
        return False

    return True



def update_cursor():
    x = gb['cursor.t']
    gb['cursor'].set_data([x, x], [0, 1])
    gb['cursor.intvl'].set_data([x, x], [0, 1])

    x = gb['cursor.snap.t']
    if gb['cursor.snap']:
        if x:
            gb['cursor.snap'].set_data([x], [get_signal_at_t(x)])

    gb['canvas'].draw()


    



## If we "snap" (hold shift while browsing), we snap
## the cursor to the closest maximum.
## Here we pick the window around the real cursor location that
## we should look in to find the peak to snap to.
SHIFT_SNAP_DT = .1
    
def snap_to_closest_peak(t):
    # Find the local maximum
    tmin,tmax = t-SHIFT_SNAP_DT,t+SHIFT_SNAP_DT
    tsels = (gb['t']>=tmin) & (gb['t']<=tmax)

    #ecg_target = gb['ecg-prep-column']
    #ecg = biodata.bio[ecg_target][tsels] # gb['ecg_clean']
    ecgsels = gb['signal'][tsels]
    ecg_t = gb['t'][tsels]
    
    peak_t = ecg_t[np.argmax(ecgsels)]
    if peak_t:
        return peak_t
    else:
        return t



    
    
def on_move(event):
        
    if event.xdata:
        t = event.xdata
        gb['cursor.t']=t
        if event.modifiers and 'shift' in event.modifiers:
            gb['cursor.snap.t']=snap_to_closest_peak(event.xdata)
        else:
            gb['cursor.snap.t']=None
        update_cursor()




def update_poincare_cursor(i1,i2):
    if 'poincare.cursor' in gb and not gb['poincare.cursor']==None:
        gb['poincare.cursor'].set_data([i1], [i2])
        gb['poincare.cursor'].set_alpha(1)
        gb['poincare.canvas'].draw()
        #print("{} {}".format(i1,i2))

def find_closest_rr(i,j):
    """ Find the closest RR-interval in the current Poincare plot to the tuple of intervals (i,j) """
    dists = [ (a-i)**2 + (b-j)**2 for _,(a,b) in gb['rrdata'] ]
    if len(dists)==0: return None
    i = np.argmin(dists)
    return gb['rrdata'][i]
        

def on_move_poincare(event):
    """ Process mouse motion in the Poincare plot """
    
    if event.xdata and event.ydata:
        (x,y) = event.xdata,event.ydata
        res = find_closest_rr(x,y)
        if res:
            t,(i1,i2) = res
            gb['poincare.t']=t
            update_poincare_cursor(i1,i2)
        
def on_click_poincare(event):
    """ 
    When people click in the Poincare window,
    move to the corresponding location in the main time window.
    """

    if gb.get('poincare.t',None):
        t = gb['poincare.t']
        #print("moving to {}".format(t))
        # Move main window to that time point and set medium zoom level
        move_window_to(t,25)
        gb['cursor.t']=t
        update_cursor()

        

def on_click(event):

    ## Detect which subplot we're clicking to determine what is the signal we want to mark
    signal = gb['signal'] #gb['channel']

    # Set a new mark
    t = event.xdata
    if not t: return

    if 'shift' in event.modifiers:
        t = snap_to_closest_peak(event.xdata)

    if event.button==MouseButton.LEFT and event.dblclick:

        # Double click left = add peak (or modify if too close to other peaks)

        # Are we zoomed in enough?
        if not check_window_zoom(t): return

        SR = gb['SR']
        samp = int(round(t*SR))
        t = samp/SR #gb['biodata'].SR # snap the time to an actual sample
        ecg = gb['signal'] #biodata.bio[gb['ecg-prep-column']]
        samp = int(t*SR)#gb['biodata'].SR)
        
        ## Is there already another peak close by?
        i,peak = find_closest_peak(t)
        dt = t-peak['t']
        if abs(dt)<PEAK_SNAP_T:

            peak['i']=samp
            peak['t']=t
            peak['valid']=True
            peak['edited']=True
            peak['source']='manual.edited'
            peak['y']=get_signal_at_t(t)

        else:
        
            #ecg = biodata.bio[gb['ecg-prep-column']]
            peak = {
                'i':samp,
                't':samp/gb['SR'],
                'valid':True,
                'edited':True,
                'source':'manual',
                'y':get_signal_at_t(t)
            }
            gb['peaks'].append(peak)
            
        redraw_all()

        return
        


    if event.button==MouseButton.RIGHT and not event.dblclick:
        # Remove closest peak (if reasonably close)

        i,peak = find_closest_peak(t)
            
        dt = t-peak['t']
        if abs(dt)<PEAK_SNAP_T:
            del gb['peaks'][i]

        redraw_all()

        return

    
    
    if event.button==MouseButton.MIDDLE and event.dblclick:
        # Double click middle mouse button:

        ##
        ## Attempt to remove the current "invalid" slice
        ##
        
        toremove = []
        for i,(s,t0,t1) in enumerate(gb['invalid']):
            if gb['channel']==s and t0<t and t<t1:
                toremove.append(i)
        gb['invalid'] = curate_invalid([ x for j,x in enumerate(gb['invalid']) if j not in toremove ]) ## actually remove

        ## If there were any peaks in that region that were
        ## marked as invalid, reactivate them.
        ## TODO Maybe

        gb['mark_in']=None
        
        redraw_all()

        return

    
    if event.button==MouseButton.MIDDLE and not event.dblclick:

        if not gb['mark_in']:

            # Let's see if this is inside an already marked invalid region -- if so, remove that region
            gb['mark_in']=t

            redraw()
        
        else:
            gb['mark_out']=t

            # Adding a new region marked as invalid
            t_start = gb['mark_in']
            t_end   = gb['mark_out']
            if t_start>t_end:
                t_start,t_end=t_end,t_start

            ## Check if overlaps with existing, if so, merge together
            toremove = []
            for i,(s,t0,t1) in enumerate(gb['invalid']):
                if s==gb['channel']:
                    ## Check if this overlaps with the to-be-added one
                    tso = np.max([t_start,t0])
                    teo = np.min([t_end,t1])

                    if tso<teo: # overlap!
                        # Grow the newly to-be-added interval so that it envelops the old one
                        t_start = np.min([t_start,t0])
                        t_end   = np.max([t_end,t1])
                        toremove.append( i ) # remove the old one

            gb['invalid'] = [ x for j,x in enumerate(gb['invalid']) if j not in toremove ]
            gb['invalid'].append( (gb['channel'],t_start,t_end) )

            #print(gb['invalid'])
            
            ## Mark any peaks in that region automatically as invalid
            for peak in gb['peaks']:
                if peak['t']>=t_start and peak['t']<t_end:
                    peak['valid']=False
            
            redraw_all()
            
            gb['mark_in']  = None
            gb['mark_out'] = None

            






# When toggling the zoom, toggle between micro and macro window size
TOGGLE_WINDOW_SIZES = [ 1.5, 25 ]
            

def toggle_zoom():
    # Switch between zoom modes: micro and macro, to allow quick zooming
    wint = gb['WINDOW_T']
    sizedist = [ np.abs(np.log(wint/t)) for t in TOGGLE_WINDOW_SIZES ]
    target = TOGGLE_WINDOW_SIZES[np.argmax(sizedist)]
    if False:
        print("Toggling!")
        print(wint)
        print(sizedist)
        print(target)
    update_window(target/wint,gb['cursor.t'])
    


    
def process_key_events(event):
    #if event.key=='left': back_in_time()
    #if event.key=='right': forward_in_time()

    if event.char=='z':
        toggle_zoom()
    if event.char=='a':
        zoom_all()


def zoom_all():
    tmax = max(gb['t'])
    gb['tstart']=0
    gb['WINDOW_T']=tmax
    update_window_definitions(); redraw_all()

def micro_zoom():
    gb['WINDOW_T']=5
    update_window_definitions(); redraw_all()

def medio_zoom():
    gb['WINDOW_T']=30
    update_window_definitions(); redraw_all()

def maxi_zoom():
    gb['WINDOW_T']=60
    update_window_definitions(); redraw_all()

        

def process_scroll_events(event):

    if gb['busy']:
        print("not drawing, already busy")
        return
    gb['busy']=True
    
    if 'ctrl' in event.modifiers:

        # Zoom
        t = event.xdata
        
        if event.step>0:
            window_wider(t)
        if event.step<0:
            window_narrower(t)

    else:

        # Pan
        
        if event.step<0:
            back_in_time()
        if event.step>0:
            forward_in_time()
        
    gb['busy']=False

        



def get_valid_RR_intervals(trange=None):
    """ 
    Returns a list of (t,ioi) tuples indicating the list of intervals (ioi) at
    each point in time (t).
    """
    
    if trange:
        tmin,tmax= trange
    else:
        tmin,tmax= -np.Inf, np.Inf

    rrs = []
    validpeaks = [ p for p in gb['peaks'] if p['valid'] ] # Take only the valid peaks
    validpeaks.sort(key=lambda p: p['t']) # sort them in time
    inv = gb['invalid']
    #print(inv)
    united = []

    for i,peak in enumerate(validpeaks[:-1]):
        if peak['t']>=tmin and peak['t']<=tmax:
            nextpeak = validpeaks[i+1]

            ## Check that this does not fall into regions marked as invalid
            t = peak['t']
            accepted = True
            for (s,t0,t1) in inv:
                if s==gb['channel'] and does_overlap((t0,t1),(t,nextpeak['t'])):
                    ## Oops, this falls into the invalid range!
                    #print("Overlaps")
                    accepted = False

            if accepted:
                rr_intvl = np.around(nextpeak['t']-peak['t'],5)
                united.append((np.around(peak['t'],5),rr_intvl))
            else:
                united.append((peak['t'],np.nan))
    
    return united





def strip_sample(pks):
    # Strip information from the peaks objects that we can reproduce easily when we reload
    ret = []
    for p in pks:
        pn = p.copy()
        del pn['i']
        del pn['y']
        ret.append(pn) 
    return ret


def save_files():
    ## Write what we've got so far to file

    ## Prepare for writing
    pc = gb['channel']
    inv = gb['qc'].get('invalid',{})
    # Round the time points, and curate them, and then insert them into the global object
    inv[pc] = [ (s,round(t0,4),round(t1,4)) for (s,t0,t1) in curate_invalid(gb['invalid']) ]
    gb['qc']['invalid']= inv

    pks = gb['qc'].get('peaks',{})
    pks[pc]=strip_sample(gb['peaks']) # just to make sure
    gb['qc']['peaks']=pks
    
    json_obj = json.dumps(gb['qc'], indent=4,cls=misc.NpEncoder)
    print("Saving {}".format(gb['JSON_OUT']))
    with open(gb['JSON_OUT'],'w') as f:
        f.write(json_obj)

    ## Also create a more succinct report that we can use to calculate HRV
    if False : # technically not needed because we recreate this info in another script
        united = get_valid_RR_intervals()
        rrs = [ {"t":t,"rr":i} for (t,i) in united ]
        
        out = pd.DataFrame(rrs)
        out['i']=range(len(rrs))
        print("Saving to {}".format(SUMMARY_OUT))
        out.to_csv(SUMMARY_OUT,index=False, float_format='%.5f')
        
    

def on_closing():
    save_files()
    gb['erp_window'].destroy()
    gb['poincare_window'].destroy()
    gb['root'].destroy()
    sys.exit(0)
    

def quit():
    on_closing()


def redraw_all():
    redraw()
    redraw_erp()
    redraw_poincare()


def back_in_time(e=None):
    gb['tstart']-=gb['WINDOW_SHIFT_T']*gb['WINDOW_T']
    redraw_all()

def forward_in_time(e=None):
    gb['tstart']+=gb['WINDOW_SHIFT_T']*gb['WINDOW_T']
    redraw_all()
    

def jump_back_in_time(e=None):
    gb['tstart']-=.95*gb['WINDOW_T']
    redraw_all()

def jump_forward_in_time(e=None):
    gb['tstart']+=.95*gb['WINDOW_T']
    redraw_all()
    


def set_window(e=None):
    # When the slider is used to move to a new portion of the signal
    new_val = gb['slider'].get()
    gb['tstart']=int(new_val)*gb['WINDOW_T']
    redraw_all()


# When we zoom in or out, by what proportion shall we change the window width?
WINDOW_CHANGE_FACTOR = 1.25


def restore_t(t_target,prop):
    """
    Return what window edge (left window edge) you need to
    get the time t at the given proportion of the width.
    I know, sounds complicated... 
    Basically, restore_t(t,.5) tells you what the
    start time of the current window would need to be for t to end up in the middle
    of the window (at the current zoom level)
    """
    #print("Prop {} Window {} T-target {}".format(prop,gb['WINDOW_T'],t_target))
    tstart = t_target- prop*gb['WINDOW_T']
    return tstart


def move_window_to(t,zoom=None):
    """ Move the current view window so that t sits right in the center """
    if not (zoom is None):
        gb['WINDOW_T']=zoom
    gb['tstart']= restore_t(t,.5) # move that to the center of the screen
    update_window_definitions()
    redraw_all()
    


def update_window(fact,around_t):
    # Determine what we want to center around
    if not around_t: around_t = gb['tstart']+gb['WINDOW_T']/2
    t_prop = (around_t-gb['tstart'])/gb['WINDOW_T'] # get at what proportion of the window that time point is located
    gb['WINDOW_T']*=fact
    gb['tstart']= restore_t(around_t,t_prop)
    update_window_definitions()
    redraw_all()

def window_wider(around_t=None):
    update_window(1/WINDOW_CHANGE_FACTOR,around_t)

def window_narrower(around_t=None):
    update_window(WINDOW_CHANGE_FACTOR,around_t)
    
def get_n_windows():
    return int(np.floor(max(gb['t'])/gb['WINDOW_T']))

def update_window_definitions():
    # If the window width has changed, cascade the necessary updates
    nwind = get_n_windows()
    gb['slider'].configure(to=nwind)
    




gb['DPI'] = 100 # plot DPI (to fix some weird appearance on Mac OS)
gb['PADDING'] = .05

    
def make_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['canvas'].get_tk_widget().destroy()
    except:
        pass
    
    fig = Figure(dpi=gb['DPI'])
    gs = fig.add_gridspec(2,1, hspace=0, wspace=0,
                          left=gb['PADDING'],right=1-gb['PADDING'])
    axs = gs.subplots(sharex=True,squeeze=True)
    gb['fig']=fig

    gb['axs']=axs[0] # the main plot
    gb['rate.ax']=axs[1] # the plot for the rate
    
    canvas = FigureCanvasTkAgg(fig, master=gb['root'])  # A tk.DrawingArea.
    canvas.get_tk_widget().pack()
    gb['canvas']=canvas

    #canvas.mpl_connect("key_press_event", process_key_events)
    canvas.mpl_connect("key_press_event", key_press_handler)
    # Bind the button_press_event with the on_click() method
    canvas.mpl_connect('button_press_event', on_click)
    canvas.mpl_connect('motion_notify_event', on_move)

    canvas.mpl_connect("scroll_event", process_scroll_events)
    
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    redraw()



ALPHA = .8


def is_in_invalid(t):
    # Return whether the given time point is in a region marked as invalid
    for (signal,t_start,t_end) in gb['invalid']:
        if signal==gb['channel']:
            if t_start<=t and t_end>=t:
                return True
    return False


def is_ectopic(ioi):
    # Following convention, we consider beats as ectopic if they are
    # shorter than 300ms or longer than 1300ms.
    # e.g., https://doi.org/10.3390/s22051984
    return ioi<.3 or ioi>1.3



TARGET_PLOT_POINTS = 2000
# how many points to actually plot in the current window (approximately)
# If the truly available data is more than this, we downsample just for display purposes

def redraw():
    
    # Determine drawrange
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange

    ax = gb['axs']
    ax.cla() # clear the axes

    rax = gb['rate.ax']
    rax.cla()
    
    c = gb['channel']


    # Plot on axis
    #biodata = gb['biodata']
    SR = gb['SR']

    #check if there's a rounding error causing differing lengths of plotx and signal
    #ecg_target = gb['ecg-prep-column']
    ecg = gb['signal'] #biodata.bio[ecg_target] # gb['ecg_clean']
    ##print(ecg.shape)
    ##print(ecg)
    
    #prep = biodata.preprocessed[ecg_target]
    tsels = (gb['t']>=tmin) & (gb['t']<=tmax)
    plot_t = gb['t']

    gb['cursor']     =ax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)
    gb['cursor.snap']=ax.plot(
        [gb['cursor.snap.t']],
        [get_signal_at_t(gb['cursor.snap.t'])],
        marker='o',markersize=5,markerfacecolor='none',
        markeredgecolor='darkgreen',alpha=.9,zorder=99999)[0]
    #print(gb['cursor.snap'])
        
    for (signal,t_start,t_end) in gb['invalid']:
        if does_overlap((t_start,t_end),drawrange):
            i = ax.axvspan(t_start, t_end,facecolor='.85', alpha=0.9,zorder=99)

    if 'mark_in' in gb and gb['mark_in']:
        ax.axvline(gb['mark_in'],color='gray',zorder=-99,lw=3)

    validpeaks =  [ peak for peak in gb['peaks'] if peak['t']>=tmin and peak['t']<=tmax ]
    TOO_MANY_PEAKS = len(validpeaks)>130

    if not TOO_MANY_PEAKS:
        for peak in validpeaks:
            if peak['valid']:
                ax.axvline(peak['t'],color='gray',zorder=-99,lw=1)
            elif peak['source']=='candidate':
                ax.axvline(peak['t'],linestyle='--',color='gray',zorder=-99,lw=.5)

    # Plot the actual signal
    x = plot_t[tsels]
    y = ecg[tsels]

    #print(x)
    #print(y)
    
    nplot = sum(tsels) ## the number of samples we'd plot if we don't do sub-sampling
    #print("Plotting {}".format(nplot))
    
    factor = int(nplot/TARGET_PLOT_POINTS)
    if factor>1:
        x,y = x[::factor],y[::factor]

    pch = '-'
    if nplot<150:
        pch = 'o-'
        
    ax.plot(x,y,
            pch,
            label='cleaned',
            zorder=-10,
            color=gb['COLORS'].get(c,"#9b0000"))

    if not TOO_MANY_PEAKS:
        for peak in validpeaks:
            col = 'green' if peak['valid'] else 'gray'
            marker = 'o'
            if peak['source']=='manual.removed':
                marker = 'x'
            elif peak['edited']: marker = 's'

            ax.plot(peak['t'],peak['y'],
                    marker,
                    mfc=col,
                    mec=col,
                    alpha=ALPHA,
                    zorder=9999)


            
    ### Now plot the intervals (R-to-R peak)
            
    united = get_valid_RR_intervals((tmin,tmax))
    #united = [ (t,i) for (t,i) in united if not np.isnan(i) ]

    if len(united):

        rax.plot([ t for (t,i) in united],
                 [ i for (t,i) in united],
                 '-',
                 color='darkblue',
                 clip_on=False)

        if not TOO_MANY_PEAKS:
            valids = [ (t,i) for (t,i) in united if not is_ectopic(i) ]

            rax.plot([ t for (t,i) in valids],
                     [ i for (t,i) in valids],
                     'o',
                     color='darkblue',
                     clip_on=False)

        ectopic = [ (t,i) for (t,i) in united if is_ectopic(i) ]

        if len(ectopic):
            rax.plot([ t for (t,i) in ectopic],
                     [ i for (t,i) in ectopic],
                     'o',color='darkred',
                     clip_on=False)

            for ioi in [.3,1.3]:
                rax.axhline(y=ioi,alpha=.3,dashes=(1,4))
        
    rax.spines['top'].set_visible(False)
    rax.spines['right'].set_visible(False)
    rax.set_ylabel('R-R interval (s)')
    gb['cursor.intvl']=rax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)

    
    ax.set_ylabel(c)
    ax.set_xlabel('t(s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    # Now determine the ylim scale
    AUTOSCALE = False #False # whether to use the matplotlib default scale
    if not AUTOSCALE:

        ## Determine "manually" what our scale limits should be
        
        sig      = ecg[tsels]
        orig = np.array(sig).copy()
        tselspec = plot_t[tsels]
        
        for (s,t0,t1) in gb['invalid']:
            if s==gb['channel']:
                sig[ (tselspec>=t0) & (tselspec<=t1) ] = np.nan
        oksig = ~np.isnan(sig)
        sig = sig[oksig] # finally truly remove them
        if not sum(oksig):
            sig = orig[ ~np.isnan(orig) ]

        mn,mx=-1,1 # default
        if len(sig):
            mn,mx = np.nanmin(sig),np.nanmax(sig)
            # add some padding on the sides
        if mn==mx:
            mn,mx=.95*mn,1.05*mx
        pad = .05*(mx-mn)
        ax.set_ylim(mn-pad,mx+pad)
    
    #ax.set_xlim(gb['tstart'],gb['tstart']+WINDOW_T)
    update_axes()


def update_axes():
    ax = gb['axs']
    tend = gb['tstart']+gb['WINDOW_T']
    
    ax.set_xlim(gb['tstart'],tend)
    gb['slider'].set(int(gb['tstart']/gb['WINDOW_T']))

    plt.tight_layout()
    gb['canvas'].draw()
    

    

   
def make_erp_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['erp.canvas'].get_tk_widget().destroy()
    except:
        pass
    
    fig = Figure(dpi=gb['DPI'])
    gs = fig.add_gridspec(1,1, hspace=0, wspace=0,
                          left=.13,right=1-gb['PADDING'])
    axs = gs.subplots(sharex=True,squeeze=True)

    gb['erp.fig']=fig
    gb['erp.axs']=axs
    canvas = FigureCanvasTkAgg(fig, master=gb['erp_window'])  # A tk.DrawingArea.
    canvas.get_tk_widget().pack()
    gb['erp.canvas']=canvas

    canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)

    bf = tkinter.Frame(gb['erp_window'])
    bf.pack(side=tkinter.TOP, fill=tkinter.BOTH)
    
    b = tkinter.Button(master=bf, text="Make template", command=capture_erp)
    b.grid(column=0,row=0,padx=10, pady=10)
    b = tkinter.Button(master=bf, text="Search", command=search_template)
    b.grid(column=1,row=0,padx=0, pady=10)
    b = tkinter.Button(master=bf, text="Accept", command=accept_search)
    b.grid(column=2,row=0,padx=0, pady=10)
    b = tkinter.Button(master=bf, text="Clear", command=clear_candidates)
    b.grid(column=3,row=0,padx=0, pady=10)

    redraw_erp()



ERP_PRE = .2 # in seconds, pre
ERP_POST = .3 # in seconds


def redraw_erp():

    ## Draw the ERP-like display
    ax = gb['erp.axs']
    ax.cla()

    ##print('Drawing {}'.format(gb['tstart']))
    # Determine drawrange
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange

    c = gb['channel']
    #biodata = gb['biodata']
    SR = gb['SR']#gb['SR']

    #ecg_target = gb['ecg-prep-column']
    #ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    ecg = gb['signal']
    
    #prep = biodata.preprocessed[ecg_target]
    plot_t = gb['t']

    ax.axhline(y=0,lw=.5,color='gray')
    ax.axvline(x=0,lw=.5,color='gray')

    pks = []
    for peak in gb['peaks']:
        t = peak['t']
        if peak['valid'] and t>=tmin and t<=tmax:
            # Draw this peak!
            pks.append(peak)

    # Trim down if too big (takes a looooong time to draw otherwise)
    fact = int(len(pks)/40)  # the number indicates how many peaks approximately we want to draw
    if fact>1:
        pks = pks[::fact]
            
    for peak in pks:

        t = peak['t']
        tpre  = t-ERP_PRE
        tpost = t+ERP_POST

        # Ensure that it does not overlap with an invalid portion
        do_plot = True
        for i,(s,t0,t1) in enumerate(gb['invalid']):
            if does_overlap( (t0,t1), (tpre,tpost) ):
                do_plot = False

        if do_plot:
            tsels = (plot_t>=tpre) & (plot_t<=tpost)

            tpre_sels = (plot_t>=tpre) & (plot_t<=t)
            baseline = np.mean(ecg[tpre_sels])

            ax.plot(plot_t[tsels]-t,
                    ecg[tsels]-baseline,
                    zorder=-10,
                    color=gb['COLORS'].get(c,"#9b0000"),
                    alpha=.9)

    if 'erp.template' in gb and len(gb['erp.template']):
        meanerp=gb['erp.template']
        erpt =np.linspace(-ERP_PRE,ERP_POST,len(meanerp))
        ax.plot(erpt,
                meanerp,'--',
                zorder=1,
                color='black',
                alpha=.9)
        

                
    ax.set_ylabel(c)
    ax.set_xlabel('t(s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ##ax.set_xlim(gb['tstart'],gb['tstart']+WINDOW_T)
    plt.tight_layout()
    gb['erp.canvas'].draw()


    


def capture_erp():
    ## Capture the current ERPs and turn them into a template
    SR = gb['SR']
    ecg = gb['signal'] #gb['biodata'].bio[gb['ecg-prep-column']]

    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange

    gb['erp.template'] = [] # clear current template
    
    erps = []
    for peak in gb['peaks']:
        t = peak['t']
        if peak['valid']:

            ## Define the region around it
            i_t = int(t*SR)

            t_pre = t-ERP_PRE
            t_post = t+ERP_POST

            # Has to fall within the range
            if t_pre<0 or t_pre<tmin or t_post>tmax: continue

            i_pre  = i_t-int(ERP_PRE*SR)
            i_post = i_t+int(ERP_POST*SR)

            # Ensure that it does not overlap with a portion marked as invalid
            do_plot = True
            for i,(s,t0,t1) in enumerate(gb['invalid']):
                if does_overlap( (t0,t1), (t_pre,t_post) ):
                    do_plot = False

            if do_plot:
                sel = ecg[i_pre:i_post]
                rescal = np.mean(ecg[i_pre:i_t])
                erps.append(sel-rescal)

    if len(erps):
        meanerp = np.mean(erps,axis=0)    
        gb['erp.template']=meanerp

        redraw_erp()



PEAK_SEARCH_MIN_DT = .4

def search_template():

    ## Ok, given an ERP template, can we find it in the current window?
    if not ('erp.template' in gb and len(gb['erp.template'])):
        print("No template defined.")
        tkinter.messagebox.showinfo("No template","No template defined")
        return

    ## Eliminate current candidates
    gb['peaks'] = [ p for p in gb['peaks'] if not p['source']=='candidate' ]

    meanerp=gb['erp.template']

    SR = gb['SR']
    #ecg_full = gb['biodata'].bio[gb['ecg-prep-column']]
    ecg_full = gb['signal']
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    if tmin<0: tmin=0
    imin,imax = int(round(tmin*SR)),int(round(tmax*SR))

    # Select the corresponding portion of ECG
    ecg    = ecg_full[imin:imax]

    # standardize the signals first
    ecgnorm = (ecg-np.mean(ecg))/(np.std(ecg)*len(ecg))
    meanerp  = (meanerp-np.mean(meanerp))/np.std(meanerp)
    #print(ecgnorm)
    #print(meanerp)
    corr = scipy.signal.correlate(ecgnorm,meanerp,mode='valid')
    mn = np.mean(corr)
    mx = np.max(corr)
    print("Correlation values M={:.3f} STD={:.3f} MIN={:.3f} MAX={:.3f}".format(mn,np.std(corr),np.min(corr),np.max(corr)))
    pks,_ = scipy.signal.find_peaks(
        corr,
        height=mn+(mx-mn)*.5,
        distance=int(PEAK_SEARCH_MIN_DT*SR)
    )
    new_peak_t = [ (p/SR)+ERP_PRE+tmin for p in pks ]

    new_peaks = [
        {'t':p,
         'valid':False,
         'edited':False,
         'source':'candidate'
         }
        for p in new_peak_t
    ]
    for p in new_peaks:
        p['i']=int(round(p['t']*SR))
        p['y']=ecg_full[p['i']]

        if not is_in_invalid(p['t']):
            _,nextpeak = find_closest_peak(p['t'])
            dt = p['t']-nextpeak['t']
            if (abs(dt)<PEAK_SNAP_T):
                if nextpeak['valid']:
                    pass
                else:
                    pass
            else:
                # Add but only if it's not too close to an existing peak
                gb['peaks']+=[p]
    redraw_all()




def accept_search():
    
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    for p in gb['peaks']:
        if p['source']=='candidate':
            p['valid']=True
            p['source']='pattern.detected'
    redraw_all()


    
def clear_candidates():
    """ Clear candidate time points that seemed to resemble the template """
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    gb['peaks'] = [ p for p in gb['peaks'] if p['source']!='candidate']
    
    redraw_all()



    

   
def make_poincare_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['poincare.canvas'].get_tk_widget().destroy()
    except:
        pass
    
    #fig,axs = plt.subplots(1,1,sharex=True,dpi=gb['DPI'])
    fig = Figure(dpi=gb['DPI'])
    gs = fig.add_gridspec(1,1, hspace=0, wspace=0,
                          left=gb['PADDING'],right=1-gb['PADDING'])
    axs = gs.subplots(sharex=True,squeeze=True)
    
    gb['poincare.fig']=fig
    gb['poincare.axs']=axs
    canvas = FigureCanvasTkAgg(fig, master=gb['poincare_window'])
    canvas.get_tk_widget().pack()
    gb['poincare.canvas']=canvas

    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    canvas.mpl_connect('motion_notify_event', on_move_poincare)
    canvas.mpl_connect('button_press_event', on_click_poincare)

    redraw_poincare()



def redraw_poincare():

    ## Draw the ERP-like display
    ax = gb['poincare.axs']
    ax.cla()

    ##print('Drawing {}'.format(gb['tstart']))
    # Determine drawrange
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange

    all_invl = [ (t,ioi) for (t,ioi) in get_valid_RR_intervals() if not np.isnan(ioi) ] #drawrange)
    allrr = [ (t,(i1,i2)) for ((t,i1),(_,i2)) in zip(all_invl[:-1],all_invl[1:]) ]

    gb['rrdata'] = allrr

    for (invl_seq,col,alp) in [
            ([ i for (t,i) in allrr if not in_range(t,drawrange) ], 'gray',.2),
            ([ i for (t,i) in allrr if     in_range(t,drawrange) ], 'darkred',.95),
    ]:
    
        if len(invl_seq):
            #invl_seq = [ (i1,i2) for (i1,i2) in zip(invl[:-1],invl[1:]) ]
            n = len(invl_seq)
            plotsize = 4
            if n>100: plotsize=3

            ax.plot(
                [ i1 for (i1,i2) in invl_seq ],
                [ i2 for (i1,i2) in invl_seq ],
                'o',alpha=alp,
                markersize=plotsize,
                color=col
            )

    ax.set_xlabel('RR intvl n (s)')
    ax.set_ylabel('RR intvl n+1 (s)')

    # Now draw a x=y reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, '-', color='gray',alpha=0.75, zorder=0, lw=.5)
    ax.set_aspect('equal')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ##ax.set_xlim(gb['tstart'],gb['tstart']+WINDOW_T)
    #plt.tight_layout()

    gb['poincare.cursor'] = ax.plot(
        [np.mean(lims)],[np.mean(lims)],'+',markersize=8,color='black',alpha=0
    )[0]
    
    gb['poincare.canvas'].draw()




def set_dpi(dpi):
    gb['DPI']=dpi
    make_plot()
    redraw_all()



def update_filter():
    refilter()
    make_plot()
    redraw_all()
    

NEUROKIT2_CLEANERS = ['biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012']




def refilter():

    signal_pre = gb['raw'] # Always start from the raw signal


    if gb['signal_invert'].get(): # If inverting the signal
        signal_pre = -signal_pre
    
    # Also make a filtered version
    for nkproc in NEUROKIT2_CLEANERS:

        nm = 'neurokit2-{}'.format(nkproc)
        if gb[nm].get():
            METHOD=nkproc
            print("Filtering ECG signal using {} procedure in neurokit2".format(METHOD))
            signal_pre = neurokit2.ecg_clean(signal_pre,sampling_rate=gb['SR'],method=METHOD)

    signal_flt = signal_pre
    if gb['heartpy_wander_remov'].get():
        ##print(signal_pre)
        signal_flt = heartpy.remove_baseline_wander(signal_pre,gb['SR'])
        ##print("Baseline wander removal")
        ##print(signal_flt)
        prop_ok = np.mean(~np.isnan(signal_flt))
        ##print(prop_ok)
        if prop_ok<.5: # if there is not enough signal left
            signal_flt = signal_pre # restore signal before baseline wander correction

    gb['signal']=signal_flt

    for p in gb['peaks']:
        p['y']=get_signal_at_t(p['t'])#biodata.bio[ gb['ecg-prep-column'] ][p['i']]
    


    
    

def main():


    # Main window dimensions
    window_w,window_h=1300,450


    # Globals to carry around

    gb["WINDOW_T"] =15 # default window width in seconds
    gb['WINDOW_SHIFT_T'] =.2 # proportion of the window to shift when we go back and forth in time


    gb['qc']={}
    gb['invalid']= []
    gb['peaks']= []
    gb['cursor.t']=0
    gb['cursor.snap.t']=0
    gb['cursor.snap']=None



    ##
    ##
    ## Select file to open
    ##
    ##

    import biobabel as bb

    fname = None
    if len(sys.argv)>1:
        fname = sys.argv[1]
    else:

        fname = bb.ask_bio_file()

    if not fname:
        print("You need to select a file. Exiting now.")
        sys.exit(-1)


    if not os.path.exists(fname):

        ok = False
        for addon in ['.hdf5','hdf5']:
            if os.path.exists(fname+addon):
                fname = fname+addon
                ok = True
                continue
        if not ok:
            print("File {} does not seem to exist. Exiting now.".format(fname))
            sys.exit(-1)




    ###
    ### Read the data
    ###
    print("Opening file {}".format(fname))
    bio       = bb.load(fname)
    bio.print()



    if len(sys.argv)>2:
        pc = sys.argv[2]
        # Assuming that that is a stream in the dataset
    else:

        fields = bio.find_channels({'modality':'ecg'})
        if not len(fields):
            fields = bio.find_channels()
        if not len(fields): # If there are still no fields
            print("No channels found... exiting.")
            sys.exit(-1)
            
        fields.sort()
        if len(fields)>1:
            pc = misc.give_choices(fields)
        else:
            pc = fields[0]


    if not pc: # if no column was selected
        sys.exit(-1)


    print("Opening channel {}".format(pc))
        
    gb['channel'] = pc
    gb['t']=bio.get_time(pc)
    hdr,dat = bio.get(pc)
    gb['SR']=hdr['sampling_frequency']
    gb['hdr']=hdr

        
    #sos = butter(15, 3, 'low', fs=gb['SR'], output='sos')
    #filtd = signal.sosfiltfilt(sos, dat)
    #gb['filtered']=filtd
    gb['raw']=dat.copy()
    gb['signal']=dat.copy()

    ##print(gb['signal'])


    # See if there is an existing peak file
    gb['JSON_OUT'] = '{}_peaks.json'.format(fname)
    if os.path.exists(gb['JSON_OUT']):
        with open(gb['JSON_OUT'],'r') as f:
            gb['qc']      = json.loads(f.read())
            gb['invalid'] = gb['qc'].get('invalid',{}).get(gb['channel'],[])
            gb['peaks']   = gb['qc'].get('peaks',{}).get(gb['channel'],[])

    # Reconstruct peak samples
    for p in gb['peaks']:
        p['i']=int(round(p['t']*gb['SR']))

    # Curate invalid
    gb['invalid'] = curate_invalid(gb['invalid'])


    SUMMARY_OUT = '{}_{}_summary.csv'.format(fname,gb['channel'].replace('/','_'))



        

    # Create the interface
        
    root = tkinter.Tk()
    root.wm_title("Biotop Peak Picker - {} {}".format(fname,gb['channel']))
    root.geometry('{}x{}'.format(window_w,window_h))
    gb['root']=root


    gb['COLORS'] = {}




    gb['tstart']=0 # the left edge of the window we're currently showing

    # Current markers
    gb['mark_in']  =None
    gb['mark_out'] =None


    gb['busy'] = False
        
    gb['rrdata'] = [] # R-R peak data points visible in the Poincare plot

    if 'peaks' not in gb:
        gb['peaks']=[]



    # Build the interface
    
    navf = tkinter.Frame(root)
    tkinter.Grid.columnconfigure(navf, 0, weight=1)
    navf.pack(side=tkinter.BOTTOM)
    bigfont = tkFont.Font(family='Helvetica', size=28, weight='bold')
    button_wid  = tkinter.Button(master=navf, text="+", command=window_wider,    font=bigfont)
    button_narr = tkinter.Button(master=navf, text="-", command=window_narrower, font=bigfont)
    button_wid.grid(column=0,row=0,padx=0, pady=10)
    button_narr.grid(column=1,row=0,padx=0, pady=10)


    button_back = tkinter.Button(master=navf, text="<", command=back_in_time, font=bigfont)
    button_forw = tkinter.Button(master=navf, text=">", command=forward_in_time, font=bigfont)
    button_back.grid(column=2,row=0,padx=10, pady=10)
    button_forw.grid(column=4,row=0,padx=10, pady=10)

    slider_update = tkinter.Scale(
        navf,
        from_=0,
        to=get_n_windows(),
        length=300,
        orient=tkinter.HORIZONTAL,
        label="")
    slider_update.bind("<ButtonRelease-1>",set_window)
    slider_update.grid(column=3,row=0,padx=10,pady=10)
    gb['slider']=slider_update







    """ Create the window menu bar """ 

    menubar = Menu(root)

    filemenu = Menu(menubar, tearoff=0)
    #filemenu.add_command(label="Open", command=lambda : None)
    #filemenu.add_separator()
    filemenu.add_command(label="Save", command=save_files)
    filemenu.add_command(label="Exit", command=quit)
    menubar.add_cascade(label="File", menu=filemenu)

    
    prepmenu = Menu(menubar, tearoff=0)

    gb['signal_invert'] = BooleanVar()
    gb['signal_invert'].set(False)
    prepmenu.add_checkbutton(label="Invert", onvalue=1, offvalue=0, variable=gb['signal_invert'], command=update_filter)
    prepmenu.add_separator()
    
    gb['heartpy_wander_remov'] = BooleanVar()
    gb['heartpy_wander_remov'].set(True)
    prepmenu.add_checkbutton(label="Baseline wander removal", onvalue=1, offvalue=0, variable=gb['heartpy_wander_remov'], command=update_filter)
    prepmenu.add_separator()
    
    for nkproc in NEUROKIT2_CLEANERS:

        nm = 'neurokit2-{}'.format(nkproc)
        gb[nm] = BooleanVar()
        gb[nm].set(False)
        prepmenu.add_checkbutton(label="Neurokit2 {}".format(nkproc),
                                 onvalue=1, offvalue=0, variable=gb[nm],
                                 command=update_filter)
    
    menubar.add_cascade(label="Preprocessing", menu=prepmenu)
    
    actionmenu = Menu(menubar, tearoff=0)

    actionmenu.add_command(label="Auto detect",command=auto_detect_peaks)
    actionmenu.add_separator()

    gb['detector']=StringVar()
    gb['detector'].set('neurokit') # default value
    # To understand what these detectors mean, see https://github.com/berndporr/py-ecg-detectors
    for detect in ['neurokit',
                   # ECG-Detectors
                   'hamilton','christov','engzee','pan_tompkins','swt','two_average','matched_filter','wqrs']:
        actionmenu.add_radiobutton(label='{} detector'.format(detect), variable=gb['detector'], value=detect)
    
    actionmenu.add_separator()
    
    actionmenu.add_command(label="Clear all",command=clear_peaks)
    actionmenu.add_command(label="Clear here",command=clear_peaks_here)
    actionmenu.add_separator()
    actionmenu.add_command(label="Load Biopac peaks",command=import_biopac_peaks)
    menubar.add_cascade(label="Peaks", menu=actionmenu)



    tempmenu = Menu(menubar, tearoff=0)
    tempmenu.add_command(label="Make template",command=capture_erp)
    tempmenu.add_command(label="Search",command=search_template)
    tempmenu.add_command(label="Accept candidates",command=accept_search)
    tempmenu.add_command(label="Clear candidates",command=clear_candidates)
    menubar.add_cascade(label="Template", menu=tempmenu)




    artmenu = Menu(menubar, tearoff=0)
    artmenu.add_command(label="Clear here",command=clear_artefacts_here)
    artmenu.add_command(label="Show next", command=next_artefact)
    menubar.add_cascade(label="Artefacts", menu=artmenu)
    



    viewmenu = Menu(menubar, tearoff=0)
    viewmenu.add_command(label="All",command=zoom_all)
    viewmenu.add_separator()
    viewmenu.add_command(label="Micro",command=micro_zoom)
    viewmenu.add_command(label="Medio",command=medio_zoom)
    viewmenu.add_command(label="Maxi",command=maxi_zoom)
    viewmenu.add_separator()
    dpimenu = Menu(viewmenu, tearoff=0)
    for dpi in [50,75,100,150,250]:
        dpimenu.add_command(label="{}".format(dpi),
                            command=lambda x=dpi: set_dpi(x) )
    viewmenu.add_cascade(label="Set DPI", menu=dpimenu)
    menubar.add_cascade(label="View", menu=viewmenu)


    
    root.config(menu=menubar)


    root.protocol("WM_DELETE_WINDOW", on_closing)


    root.bind("<Left>",back_in_time)
    root.bind("<Right>",forward_in_time)
    root.bind("<Prior>",jump_back_in_time) # page_up
    root.bind("<Next>",jump_forward_in_time) # page_down
    root.bind("<Key>",process_key_events)


    refilter()
    
    make_plot()
    redraw()





    ##
    ## Create an additional window for an ERP_like display
    ##
    erp_window = Toplevel(root)
    gb['erp_window']=erp_window
    erp_window.wm_title("Physio Event-Related - {}".format(fname))
    erp_window.geometry('{}x{}'.format(450,550))

    make_erp_plot()



    ##
    ## Create an additional window for a Poincare display
    ##
    poincare_window = Toplevel(root)
    gb['poincare_window']=poincare_window
    poincare_window.wm_title("Physio Poincare - {} {}".format(fname,gb['channel']))
    poincare_window.geometry('{}x{}'.format(450,400))

    make_poincare_plot()


    tkinter.mainloop()



