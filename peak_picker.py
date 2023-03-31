


import tkinter
from tkinter import filedialog as fd
from tkinter import font as tkFont  # for convenience
from tkinter import Toplevel
import tkinter.messagebox
from tkinter.messagebox import askyesno

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import preprocess_ecg_detectors as preprocess_ecg

import numpy as np
import pandas as pd
import os
import scipy.signal


import json

import sys

from misc import does_overlap


# Main window dimensions
window_w,window_h=1300,450


# Globals to carry around
gb = {}

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

fname = None
if len(sys.argv)>1:
    fname = sys.argv[1]
else:

    filetypes = (
        ('HDF5 dataset', '*.hdf5'),
        ('All files', '*.*')
    )

    fname = fd.askopenfilename(
        title='Select your recording',
        initialdir='.',
        filetypes=filetypes)

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

import read_h5py

biodata       = read_h5py.read(fname)

print(biodata.summary())

gb['biodata'] = biodata
gb['SR']      = biodata.SR
gb['bio']     = biodata.bio




import misc
fields = biodata.get_ecg_channels()
fields.sort()
if len(fields)>1:

    if len(sys.argv)>2:
        pc = sys.argv[2]
    else:
        pc = misc.give_choices(fields)
else:
    pc = fields[0]
if pc:
    gb['plot.column'] = pc
    gb['ecg-prep-column']=pc+'-prep'
    gb['ecg_preprocess'] = {gb['plot.column']:gb['ecg-prep-column']}

else:
    sys.exit(-1)



    

bio = biodata.bio
print("Effective sampling rate hovers around {} Hz".format(biodata.SR))
bio['sample.t']=np.arange(bio['t'].shape[0])/biodata.SR




root = tkinter.Tk()
root.wm_title("Physio Peak Picker - {}".format(fname))
root.geometry('{}x{}'.format(window_w,window_h))
gb['root']=root


COLORS = {}





gb['tstart']=0 # the left edge of the window we're currently showing

# Current markers
gb['mark_in']  =None
gb['mark_out'] =None




def do_auto_detect_peaks():
    ecg_target = gb['ecg-prep-column']
    ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    
    peaks = preprocess_ecg.peak_detect(ecg,gb['SR'])
    gb['peaks']= [
        {
            'i':samp,
            't':samp/biodata.SR,
            'valid':True,
            'source':'auto',
            'edited':False,
            'y':ecg[samp]
        } for samp in peaks
    ]





    



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
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    gb['peaks'] = [ p for p in gb['peaks']
                    if p['t']<tmin or p['t']>tmax ]
    redraw_all()





        

    
def auto_detect_peaks():
    if len(gb['peaks']):
        answer = askyesno(
            title='confirmation',
            message='Auto detecting peaks will clear any peaks may you have edited or added.\nDo you want to proceed?')
    else:
        answer = True
    if answer:
        do_auto_detect_peaks()
        redraw_all()



def do_import_biopac_peaks(f):
    try:
        import read_biopac
        peakdata = read_biopac.read_acq_ecg_peaks(f)
        ts = peakdata['Time']
        gb['peaks']= [
            {
                'i':int(round(t*biodata.SR)),
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





## ECG Preprocessing
ecg_target = gb['ecg-prep-column']
preprocess_ecg.preprocess(biodata,gb,[gb['plot.column']])
if 'peaks' not in gb:
    gb['peaks']=[]




MIN_INVALID_DUR = .1 # minimum size for an "invalid" portion

def curate_invalid(inv):
    toret = []
    for (s,t0,t1) in inv:
        dt = abs(t0-t1)
        if dt<MIN_INVALID_DUR: continue
        ## Too short to be plausible
        toret.append( (s,t0,t1) )
    return toret

    

# See if there is an existing peak file
JSON_OUT = '{}_peaks.json'.format(fname)
if os.path.exists(JSON_OUT):
    with open(JSON_OUT,'r') as f:
        gb['qc']      = json.loads(f.read())
        gb['invalid'] = gb['qc'].get('invalid',{}).get(gb['plot.column'],[])
        gb['peaks']   = gb['qc'].get('peaks',{}).get(gb['plot.column'],[])

# Reconstruct peak samples
for p in gb['peaks']:
    p['i']=int(round(p['t']*biodata.SR))
    p['y']=biodata.bio[ gb['ecg-prep-column'] ][p['i']]

# Curate invalid
gb['invalid'] = curate_invalid(gb['invalid'])

    
SUMMARY_OUT = '{}_{}_summary.csv'.format(fname,gb['plot.column'])

        
    

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
    tsels = (gb['bio']['t']>=tmin) & (gb['bio']['t']<=tmax)

    ecg_target = gb['ecg-prep-column']
    ecg = biodata.bio[ecg_target][tsels] # gb['ecg_clean']
    ecg_t = gb['bio']['t'][tsels]
    
    peak_t = ecg_t[np.argmax(ecg)]
    if peak_t:
        return peak_t
    else:
        return t


def get_signal_at_t(t):
    # Return the ECG signal value closest to time t
    if not t: return None
    samp = int(round(t*biodata.SR)) # getting the closest sample
    ecg_target = gb['ecg-prep-column']
    ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    return ecg[samp]


    
    
def on_move(event):
        
    if event.xdata:
        t = event.xdata
        gb['cursor.t']=t
        if 'shift' in event.modifiers:
            gb['cursor.snap.t']=snap_to_closest_peak(event.xdata)
        else:
            gb['cursor.snap.t']=None
        update_cursor()
    

def on_click(event):

    ## Detect which subplot we're clicking to determine what is the signal we want to mark
    signal = gb['plot.column']

    # Set a new mark
    t = event.xdata
    if not t: return

    if 'shift' in event.modifiers:
        t = snap_to_closest_peak(event.xdata)

    if event.button==MouseButton.LEFT and event.dblclick:

        # Double click left = add peak (or modify if too close to other peaks)

        # Are we zoomed in enough?
        if not check_window_zoom(t): return

        samp = int(round(t*gb['biodata'].SR))
        t = samp/gb['biodata'].SR # snap the time to an actual sample
        ecg = biodata.bio[gb['ecg-prep-column']]
        samp = int(t*gb['biodata'].SR)
        
        ## Is there already another peak close by?
        i,peak = find_closest_peak(t)
        dt = t-peak['t']
        if abs(dt)<PEAK_SNAP_T:

            peak['i']=samp
            peak['t']=t
            peak['valid']=True
            peak['edited']=True
            peak['source']='manual.edited'
            peak['y']=ecg[samp]

        else:
        
            ecg = biodata.bio[gb['ecg-prep-column']]
            peak = {
                'i':samp,
                't':samp/biodata.SR,
                'valid':True,
                'edited':True,
                'source':'manual',
                'y':ecg[samp]
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

        ##
        ## Attempt to remove the current "invalid" slice
        ##
        
        toremove = []
        for i,(s,t0,t1) in enumerate(gb['invalid']):
            if signal==s and t0<t and t<t1:
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
                if s==signal:
                    ## Check if this overlaps with the to-be-added one
                    tso = np.max([t_start,t0])
                    teo = np.min([t_end,t1])

                    if tso<teo: # overlap!
                        # Grow the newly to-be-added interval so that it envelops the old one
                        t_start = np.min([t_start,t0])
                        t_end   = np.max([t_end,t1])
                        toremove.append( i ) # remove the old one

            gb['invalid'] = [ x for j,x in enumerate(gb['invalid']) if j not in toremove ]
            gb['invalid'].append( (signal,t_start,t_end) )

            ## Mark any peaks in that region automatically as invalid
            for peak in gb['peaks']:
                if peak['t']>=t_start and peak['t']<t_end:
                    peak['valid']=False
            
            redraw_all()
            
            gb['mark_in']  = None
            gb['mark_out'] = None

            







def process_key_events(event):
    if event.key=='left':
        back_in_time()
    if event.key=='right':
        forward_in_time()
        


def process_scroll_events(event):

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
        

        
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)




def get_valid_RR_intervals(trange=None):
    
    if trange:
        tmin,tmax= trange
    else:
        tmin,tmax= -np.Inf, np.Inf

    rrs = []
    validpeaks = [ p for p in gb['peaks'] if p['valid'] ] # Take only the valid peaks
    validpeaks.sort(key=lambda p: p['t']) # sort them in time
    inv = gb['invalid']
    united = []

    for i,peak in enumerate(validpeaks[:-1]):
        if peak['t']>=tmin and peak['t']<=tmax:
            nextpeak = validpeaks[i+1]

            ## Check that this does not fall into invalid regions
            t = peak['t']
            accepted = True
            for (s,t0,t1) in inv:
                if s==gb['plot.column'] and does_overlap((t0,t1),(t,nextpeak['t'])):
                    ## Oops, this falls into the invalid range!
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
    pc = gb['plot.column']
    inv = gb['qc'].get('invalid',{})
    # Round the time points, and curate them, and then insert them into the global object
    inv[pc] = [ (s,round(t0,4),round(t1,4)) for (s,t0,t1) in curate_invalid(gb['invalid']) ]
    gb['qc']['invalid']= inv

    pks = gb['qc'].get('peaks',{})
    pks[pc]=strip_sample(gb['peaks']) # just to make sure
    gb['qc']['peaks']=pks
    
    json_obj = json.dumps(gb['qc'], indent=4,cls=NpEncoder)
    print("Saving {}".format(JSON_OUT))
    with open(JSON_OUT,'w') as f:
        f.write(json_obj)

    ## Also create a more succinct report that we can use to calculate HRV
    united = get_valid_RR_intervals()
    rrs = [ {"t":t,"rr":i} for (t,i) in united ]

    out = pd.DataFrame(rrs)
    out['i']=range(len(rrs))
    out.to_csv(SUMMARY_OUT,index=False, float_format='%.5f')
    print("Saved {}".format(SUMMARY_OUT))
        
    

def on_closing():
    save_files()
    root.destroy()
    

def quit():
    on_closing()


def redraw_all():
    redraw()
    redraw_erp()
    redraw_poincare()

def back_in_time():
    gb['tstart']-=gb['WINDOW_SHIFT_T']*gb['WINDOW_T']
    redraw_all()

def forward_in_time():
    gb['tstart']+=gb['WINDOW_SHIFT_T']*gb['WINDOW_T']
    redraw_all()
    
def set_window(e=None):
    # When the slider is used to move to a new portion of the signal
    new_val = gb['slider'].get()
    gb['tstart']=int(new_val)*gb['WINDOW_T']
    redraw_all()


# When we zoom in or out, by what proportion shall we change the window width?
WINDOW_CHANGE_FACTOR = 1.25


def restore_t(t_target,prop):
    # Return what window edge (left window edge) you need to
    # get the time t at the given proportion of the width.
    # I know, sounds complicated...
    #print("Prop {} Window {} T-target {}".format(prop,gb['WINDOW_T'],t_target))
    tstart = t_target- prop*gb['WINDOW_T']
    #print(tstart)
    return tstart
    

def update_window(fact,around_t):
    # Determine what we want to center around
    if not around_t: around_t = gb['tstart']+gb['WINDOW_T']/2
    t_prop = (around_t-gb['tstart'])/gb['WINDOW_T'] # get at what proportion of the window that time point is located
    gb['WINDOW_T']*=fact
    gb['tstart']= restore_t(around_t,t_prop)
    update_window_definitions()
    ##print(gb['tstart'])
    redraw_all()

def window_wider(around_t=None):
    update_window(1/WINDOW_CHANGE_FACTOR,around_t)

def window_narrower(around_t=None):
    update_window(WINDOW_CHANGE_FACTOR,around_t)
    
def get_n_windows():
    return int(np.floor(max(gb['bio']['t'])/gb['WINDOW_T']))

def update_window_definitions():
    # If the window width has changed, cascade the necessary updates
    nwind = get_n_windows()
    gb['slider'].configure(to=nwind)
    



    
def make_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['canvas'].get_tk_widget().destroy()
    except:
        pass
    
    fig,axs = plt.subplots(2,1,sharex=True)
    gb['fig']=fig
    gb['axs']=axs[0] # the main plot
    gb['rate.ax']=axs[1] # the plot for the rate
    canvas = FigureCanvasTkAgg(fig, master=gb['root'])  # A tk.DrawingArea.
    canvas.get_tk_widget().pack()
    gb['canvas']=canvas

    canvas.mpl_connect("key_press_event", process_key_events)
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
        if signal==gb['plot.column']:
            if t_start<=t and t_end>=t:
                return True
    return False




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
    
    c = gb['plot.column']


    # Plot on axis
    biodata = gb['biodata']
    SR = biodata.SR

    #check if there's a rounding error causing differing lengths of plotx and signal
    ecg_target = gb['ecg-prep-column']
    ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    ##print(ecg.shape)
    
    #prep = biodata.preprocessed[ecg_target]
    tsels = (gb['bio']['t']>=tmin) & (gb['bio']['t']<=tmax)
    plot_t = gb['bio']['t']

    gb['cursor']     =ax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)
    gb['cursor.snap']=ax.plot([gb['cursor.snap.t']],
                              [get_signal_at_t(gb['cursor.snap.t'])],
                              marker='o',markersize=5,markerfacecolor='none',
                              markeredgecolor='darkgreen',alpha=.9,zorder=99999)[0]
    #print(gb['cursor.snap'])
        
    for (signal,t_start,t_end) in gb['invalid']:
        if does_overlap((t_start,t_end),drawrange):
            i = ax.axvspan(t_start, t_end,facecolor='.85', alpha=0.9,zorder=99)

    if 'mark_in' in gb and gb['mark_in']:
        ax.axvline(gb['mark_in'],color='gray',zorder=-99,lw=3)

            
    for peak in gb['peaks']:
        if peak['t']>=tmin and peak['t']<=tmax:
            if peak['valid']:
                ax.axvline(peak['t'],color='gray',zorder=-99,lw=1)
            elif peak['source']=='candidate':
                ax.axvline(peak['t'],linestyle='--',color='gray',zorder=-99,lw=.5)
                
    # Plot the actual signal
    x = plot_t[tsels]
    y = ecg[tsels]

    nplot = sum(tsels) ## the number of samples we'd plot if we don't do sub-sampling
    #print("Plotting {}".format(nplot))
    
    factor = int(nplot/TARGET_PLOT_POINTS)
    if factor>1:
        x,y = x[::factor],y[::factor]

    pch = '-'
    if nplot<100:
        pch = 'o-'
        
    ax.plot(x,y,
            pch,
            label='cleaned',
            zorder=-10,
            color=COLORS.get(c,"#9b0000"))

    for peak in gb['peaks']:
        if peak['t']>=tmin and peak['t']<=tmax:
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


            

    united = get_valid_RR_intervals((tmin,tmax))
    #united = [ (t,i) for (t,i) in united if not np.isnan(i) ]

    if len(united):
        rax.plot([ t for (t,i) in united],
                 [ i for (t,i) in united],'o-',clip_on=False)
        realvals = [ i for (t,i) in united if np.isfinite(i) ]
        #if len(realvals):
        #    rax.set_ylim(0,1.1*max(realvals))
    rax.spines['top'].set_visible(False)
    rax.spines['right'].set_visible(False)
    rax.set_ylabel('R-R interval (s)')
    gb['cursor.intvl']=rax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)

    
    ax.set_ylabel(c)
    ax.set_xlabel('t(s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    # Now determine the ylim scale
    AUTOSCALE = False # whether to use the matplotlib default scale
    if not AUTOSCALE:

        ## Remove the "invalid" portions of the signal too
        sig      = ecg[tsels]
        tselspec = plot_t[tsels]
        
        for (s,t0,t1) in gb['invalid']:
            if s==gb['plot.column']:
                sig[ (tselspec>=t0) & (tselspec<=t1) ] = np.nan
                #tselspec = tselspec & ((plot_t<t0)|(plot_t>t1))
        sig = sig[~np.isnan(sig)] # finally truly remove them
        if len(sig):
            mn,mx = np.min(sig),np.max(sig)
            # add some padding on the sides
            pad = .05*(mx-mn)
            ax.set_ylim(mn-pad,mx+pad)
        else:
            ax.set_ylim(-1,1)
    
    #ax.set_xlim(gb['tstart'],gb['tstart']+WINDOW_T)
    update_axes()


def update_axes():
    ax = gb['axs']
    tend = gb['tstart']+gb['WINDOW_T']
    
    ax.set_xlim(gb['tstart'],tend)
    gb['slider'].set(int(gb['tstart']/gb['WINDOW_T']))

    plt.tight_layout()
    gb['canvas'].draw()
    

    


    
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




b = tkinter.Button(master=navf, text="Auto Detect", command=auto_detect_peaks)
b.grid(column=5,row=0,padx=10, pady=10)
b = tkinter.Button(master=navf, text="Clear all", command=clear_peaks)
b.grid(column=6,row=0,padx=0, pady=10)
b = tkinter.Button(master=navf, text="Clear here", command=clear_peaks_here)
b.grid(column=7,row=0,padx=0, pady=10)
b = tkinter.Button(master=navf, text="Load biopac peaks", command=import_biopac_peaks)
b.grid(column=8,row=0,padx=0, pady=10)
b = tkinter.Button(master=navf, text="Save", command=save_files)
b.grid(column=9,row=0,padx=0, pady=10)


# Packing order is important. Widgets are processed sequentially and if there
# is no space left, because the window is too small, they are not displayed.
# The canvas is rather flexible in its size, so we pack it last which makes
# sure the UI controls are displayed as long as possible.
#toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
#canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

root.protocol("WM_DELETE_WINDOW", on_closing)

make_plot()
redraw()





##
## Create an additional window for an ERP_like display
##
erp_window = Toplevel(root)
gb['erp_window']=erp_window
erp_window.wm_title("Physio Event-Related - {}".format(fname))
erp_window.geometry('{}x{}'.format(450,550))

   
def make_erp_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['erp.canvas'].get_tk_widget().destroy()
    except:
        pass
    
    fig,axs = plt.subplots(1,1,sharex=True,figsize=(5,5))
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

    c = gb['plot.column']
    biodata = gb['biodata']
    SR = biodata.SR

    ecg_target = gb['ecg-prep-column']
    ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    
    #prep = biodata.preprocessed[ecg_target]
    plot_t = biodata.bio['t']

    ax.axhline(y=0,lw=.5,color='gray')
    ax.axvline(x=0,lw=.5,color='gray')
    
    for peak in gb['peaks']:
        t = peak['t']
        if peak['valid'] and t>=tmin and t<=tmax:
            # Draw this peak!

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
                        color=COLORS.get(c,"#9b0000"),
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
    SR = gb['biodata'].SR
    ecg = gb['biodata'].bio[gb['ecg-prep-column']]

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

    SR = gb['biodata'].SR
    ecg_full = gb['biodata'].bio[gb['ecg-prep-column']]
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
    
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    gb['peaks'] = [ p for p in gb['peaks'] if p['source']!='candidate']
    
    redraw_all()



    
make_erp_plot()

    

##
## Create an additional window for a Poincare display
##
poincare_window = Toplevel(root)
gb['poincare_window']=poincare_window
poincare_window.wm_title("Physio Poincare - {} {}".format(fname,gb['plot.column']))
poincare_window.geometry('{}x{}'.format(450,400))

   
def make_poincare_plot():
    # Get the currently selected subplots
    # and show just these.
    # Effectively, it recreates figures and subplots
    try: 
        gb['poincare.canvas'].get_tk_widget().destroy()
    except:
        pass
    
    fig,axs = plt.subplots(1,1,sharex=True)
    gb['poincare.fig']=fig
    gb['poincare.axs']=axs
    canvas = FigureCanvasTkAgg(fig, master=gb['poincare_window'])
    canvas.get_tk_widget().pack()
    gb['poincare.canvas']=canvas

    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    redraw_poincare()



def redraw_poincare():

    ## Draw the ERP-like display
    ax = gb['poincare.axs']
    ax.cla()

    ##print('Drawing {}'.format(gb['tstart']))
    # Determine drawrange
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange

    invl = get_valid_RR_intervals(drawrange)
    if len(invl):
        invl_seq = [ (i1,i2) for ((_,i1),(_,i2)) in zip(invl[:-1],invl[1:]) ]

        ax.plot([ i1 for (i1,i2) in invl_seq ],
                [ i2 for (i1,i2) in invl_seq ],'o',alpha=.95)
    
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
    plt.tight_layout()
    gb['poincare.canvas'].draw()

    
make_poincare_plot()








tkinter.mainloop()



