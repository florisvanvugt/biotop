import biobabel as bb

import sys
import os

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

import numpy as np
from scipy.signal import butter
from scipy import signal
import json

import neurokit2 as nk
import biotop.misc as misc



def get_signal_at_t(t):
    # Return the ECG signal value closest to time t
    if not t: return None
    tcol = gb['t']
    #print(tcol-t)
    i = np.argmin(abs(tcol-t))
    #samp = int(round(t*gb['SR'])) # getting the closest sample
    #ecg_target = gb['ecg-prep-column']
    return gb['signal'][i]





MIN_INVALID_DUR = .1 # minimum size for an "invalid" portion

def curate_invalid(inv):
    toret = []
    for (s,t0,t1) in inv:
        dt = abs(t0-t1)
        if dt<MIN_INVALID_DUR: continue
        ## Too short to be plausible
        toret.append( (s,t0,t1) )
    return toret






#print(gb)




def auto_detect_peaks():

    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    tmin = max([0,tmin]) # don't go below zero
    peaks_here = [ p for p in gb['peaks'] if p['t']>=tmin and p['t']<=tmax ]
    
    if len(peaks_here):
        answer = askyesno(
            title='confirmation',
            message='Auto detecting peaks will clear any peaks may you have edited or added.\nDo you want to proceed?')
    else:
        answer = True
    if answer:
        do_auto_detect_peaks()
        redraw_all()


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




def do_auto_detect_peaks():

    ## First clear the peaks in the current window
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    # Remove existing peaks in this range
    gb['peaks'] = [ p for p in gb['peaks']
                    if p['t']<tmin or p['t']>tmax ]


    ## Now, find the valid portions of signal in the current window.
    ranges = find_valid_between(tmin,tmax)

    t = gb['t']

    for (fromt,tot) in ranges:

        ## Take the chunk of data in the current window
        tsels = (t>=fromt) & (t<=tot)
        tmin,tmax=min(t[tsels]),max(t[tsels]) # this can differ from the window edges (if we are at the end or beginning of the signal)
        samp_min = int(round(fromt*gb['SR'])) # calculate what sample the starting point corresponds to
        signal = gb['signal'][tsels]

        signals, info = nk.rsp_process(
            signal,
            sampling_rate=gb['SR'],
            report="text",
            method='khodadad2018')

        peaks = [ ('peak',s) for s in info['RSP_Peaks'] ]+[ ('trough',s) for s in info['RSP_Troughs'] ]

        gb['peaks'] += [
            {
                'i':samp+samp_min,
                't':(samp+samp_min)/gb['SR'],
                'valid':True,
                'source':'auto',
                'edited':False,
                'y':signal[samp],
                'kind':kind
            } for kind,samp in peaks
        ]

    

def clear_peaks():
    clear_peaks_in_range(-np.Inf,np.Inf)


def clear_peaks_here():
    drawrange = (gb['tstart'],gb['tstart']+gb['WINDOW_T'])
    tmin,tmax = drawrange
    clear_peaks_in_range(tmin,tmax)

    
def clear_peaks_in_range(tmin,tmax):
    gb['peaks'] = [ p for p in gb['peaks']
                    if p['t']<tmin or p['t']>tmax ]
    redraw_all()
    




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
    inv[pc] = [ (s,round(t0,5),round(t1,5)) for (s,t0,t1) in curate_invalid(gb['invalid']) ]
    gb['qc']['invalid']= inv

    pks = gb['qc'].get('peaks',{})
    pks[pc]=strip_sample(gb['peaks']) # just to make sure
    gb['qc']['peaks']=pks
    
    json_obj = json.dumps(gb['qc'], indent=4,cls=misc.NpEncoder)
    print("Saving {}".format(gb['JSON_OUT']))
    with open(gb['JSON_OUT'],'w') as f:
        f.write(json_obj)



def build_gui(root):
    
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
    b = tkinter.Button(master=navf, text="Save", command=save_files)
    b.grid(column=9,row=0,padx=0, pady=10)


    # Packing order is important. Widgets are processed sequentially and if there
    # is no space left, because the window is too small, they are not displayed.
    # The canvas is rather flexible in its size, so we pack it last which makes
    # sure the UI controls are displayed as long as possible.
    #toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    #canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", on_closing)


    root.bind("<Left>",back_in_time)
    root.bind("<Right>",forward_in_time)
    root.bind("<Prior>",jump_back_in_time) # page_up
    root.bind("<Next>",jump_forward_in_time) # page_down
    root.bind("<Key>",process_key_events)





    
from biotop.misc import does_overlap

TARGET_PLOT_POINTS = 2000
# how many points to actually plot in the current window (approximately)
# If the truly available data is more than this, we downsample just for display purposes

ALPHA = .5


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
    bio = gb['bio']
    dat = gb['signal']
    raw = gb['raw']
    SR = gb['SR']

    #check if there's a rounding error causing differing lengths of plotx and signal
    #ecg_target = gb['ecg-prep-column']
    #ecg = biodata.bio[ecg_target] # gb['ecg_clean']
    ##print(ecg.shape)
    
    #prep = biodata.preprocessed[ecg_target]
    plot_t = gb['t']
    tsels = (plot_t>=tmin) & (plot_t<=tmax)
    #print(tsels)
    signal = gb['signal']

    gb['cursor']     =ax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)
    gb['cursor.snap']=ax.plot([gb['cursor.snap.t']],
                              [get_signal_at_t(gb['cursor.snap.t'])],
                              marker='o',markersize=9,markerfacecolor='none',
                              markeredgecolor='darkgreen',alpha=.9,zorder=99999)[0]
    #print(gb['cursor.snap'])
        
    for (s,t_start,t_end) in gb['invalid']:
        if s==gb['channel'] and does_overlap((t_start,t_end),drawrange):
            i = ax.axvspan(t_start, t_end,facecolor='.85', alpha=0.9,zorder=99)

    if 'mark_in' in gb and gb['mark_in']:
        ax.axvline(gb['mark_in'],color='gray',zorder=-99,lw=3)
            
    for peak in gb['peaks']:
        if peak['t']>=tmin and peak['t']<=tmax:
            if peak['valid']:
                if peak.get('kind')=='trough':
                    ax.axvline(peak['t'],linestyle='--',color='gray',zorder=-99,lw=.5,alpha=.6)
                else:
                    ax.axvline(peak['t'],color='gray',zorder=-99,lw=1,alpha=.6)
                
    # Plot the actual signal
    x = plot_t[tsels]
    y = signal[tsels]

    nplot = sum(tsels) ## the number of samples we'd plot if we don't do sub-sampling
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
            color=gb['COLORS'].get(c,"#9b0000"))


    ## Now plot the raw unfiltered signal
    r = raw[tsels]
    x = plot_t[tsels]
    if factor>1:
        x,r = x[::factor],r[::factor]

    ax.plot(x,r,
            pch,
            label='raw',
            zorder=-15,
            alpha=.3,
            color=gb['COLORS'].get(c,"#999999"))


    # Indicate the peaks
    for peak in gb['peaks']:
        if peak['t']>=tmin and peak['t']<=tmax:
            col = 'gray'
            marker = 'o'
            if peak['valid']:
                if peak.get('kind','')=='peak':
                    marker = 's'
                    col = 'blue'
                else:
                    col = 'green'
            if peak['source']=='manual.removed':
                marker = 'x'
            elif peak['edited']: marker = 's'
            
            ax.plot(peak['t'],peak['y'],
                    marker,
                    mfc=col,
                    mec=col,
                    alpha=ALPHA,
                    zorder=9999)



    if True:
        # Plot the rates
        rax = gb['rate.ax']

        united = get_valid_intervals((tmin,tmax))

        for (kind,col) in [ ('peak-to-trough','blue'), ('trough-to-peak','green'), ('none','gray') ]:
            vals = [ (t,ioi) for (t,ioi,k) in united if kind==k ]
            if len(vals):
                rax.plot([ t for (t,i) in vals],
                         [ i for (t,i) in vals],
                         'o-',clip_on=False,
                         color = col,
                         label=kind,
                         alpha=.7
                         )
                #realvals = [ i for (t,i) in united if np.isfinite(i) ]
                #if len(realvals):
                #    rax.set_ylim(0,1.1*max(realvals))
        rax.spines['top'].set_visible(False)
        rax.spines['right'].set_visible(False)
        rax.set_ylabel('Interval (s)')
        
        gb['cursor.intvl']=rax.axvline(x=gb['cursor.t'],lw=1,color='blue',alpha=.9,zorder=99999)
    
    ax.set_ylabel(c)
    ax.set_xlabel('t(s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    # Now determine the ylim scale
    AUTOSCALE = False # whether to use the matplotlib default scale
    if not AUTOSCALE:

        ## Remove the "invalid" portions of the signal too
        sig      = signal[tsels]
        tselspec = plot_t[tsels]
        
        for (s,t0,t1) in gb['invalid']:
            if s==gb['channel']:
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
    







## If we "snap" (hold shift while browsing), we snap
## the cursor to the closest maximum.
## Here we pick the window around the real cursor location that
## we should look in to find the peak to snap to.
SHIFT_SNAP_DT = .5
    
def snap_to_closest_peak(t,invert=False):
    # Find the local maximum
    # If invert=True, find the local minimum instead
    tmin,tmax = t-SHIFT_SNAP_DT,t+SHIFT_SNAP_DT
    tsels = (gb['t']>=tmin) & (gb['t']<=tmax)

    signal = gb['signal'][tsels]
    if invert:
        signal = -signal
    t = gb['t'][tsels]
    peak_t = t[np.argmax(signal)]
    if peak_t:
        return peak_t
    else:
        return t


    
PEAK_SNAP_T = .2 # how close in time do we need to be to a peak to disable it

PEAK_EDIT_MAX_WINDOW_T = 8 # the maximum window size that allows peak adjusting/adding.
# If the window is too far zoomed out, we can't trust the accuracy of peak editing



def find_closest_peak(t):
    dts = [ t-peak['t'] for peak in gb['peaks'] ]
    if not len(dts): return None,{"t":np.Inf}
    min_dt = min([ abs(d) for d in dts ])
    for i,peak in enumerate(gb['peaks']):
        dt = t-peak['t']
        if abs(dt)==min_dt:
            return i,peak



   

def update_cursor():
    x = gb['cursor.t']
    gb['cursor'].set_data([x, x], [0, 1])
    gb['cursor.intvl'].set_data([x, x], [0, 1])

    x = gb['cursor.snap.t']
    if gb['cursor.snap']:
        if x:
            gb['cursor.snap'].set_data([x], [get_signal_at_t(x)])

    gb['canvas'].draw()


    
def on_move(event):
        
    if event.xdata:
        t = event.xdata
        gb['cursor.t']=t
        if event.modifiers and 'shift' in event.modifiers:
            gb['cursor.snap.t']=snap_to_closest_peak(event.xdata,invert=False)
        elif event.modifiers and 'ctrl' in event.modifiers:
            gb['cursor.snap.t']=snap_to_closest_peak(event.xdata,invert=True)
        else:
            gb['cursor.snap.t']=None
        update_cursor()
    




def check_window_zoom(t):

    if gb['WINDOW_T']>PEAK_EDIT_MAX_WINDOW_T:

        # First zoom in
        update_window(.8*PEAK_EDIT_MAX_WINDOW_T/gb['WINDOW_T'],t)
        
        return False

    return True

        
    
def on_click(event):

    ## Detect which subplot we're clicking to determine what is the signal we want to mark
    signal = gb['channel']

    # Set a new mark
    t = event.xdata
    if not t: return

    kind = 'none'
    if 'shift' in event.modifiers:
        t = snap_to_closest_peak(event.xdata,invert=False)
        kind = 'peak'
    if 'ctrl' in event.modifiers:
        t = snap_to_closest_peak(event.xdata,invert=True)
        kind = 'trough'

    if event.button==MouseButton.LEFT and event.dblclick:

        # Double click left = add peak (or modify if too close to other peaks)

        # Are we zoomed in enough?
        if not check_window_zoom(t): return

        samp = int(round(t*gb['SR']))
        t = samp/gb['SR'] # snap the time to an actual sample
        signal = gb['signal']
        samp = int(t*gb['SR'])
        
        ## Is there already another peak close by?
        i,peak = find_closest_peak(t)
        dt = t-peak['t']
        if abs(dt)<PEAK_SNAP_T:

            peak['i']=samp
            peak['t']=t
            peak['valid']=True
            peak['edited']=True
            peak['source']='manual.edited'
            peak['y']=signal[samp]
            peak['kind']=kind

        else:

            # Add a new peak
            peak = {
                'i':samp,
                't':samp/gb['SR'],
                'valid':True,
                'edited':True,
                'source':'manual',
                'y':signal[samp],
                'kind':kind
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
        gb['mark_in']=None
        
        redraw_all()

        return

    
    if event.button==MouseButton.MIDDLE and not event.dblclick:

        if not gb['mark_in']:

            # Let's see if this is inside an already marked invalid region -- if so, remove that region
            gb['mark_in']=t
            redraw_all()
        
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

            ## Mark any peaks in that region automatically as invalid
            for peak in gb['peaks']:
                if peak['t']>=t_start and peak['t']<t_end:
                    peak['valid']=False
            
            redraw_all()
            
            gb['mark_in']  = None
            gb['mark_out'] = None

            


    
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

    #canvas.mpl_connect("key_press_event", process_key_events)
    canvas.mpl_connect("key_press_event", key_press_handler)
    # Bind the button_press_event with the on_click() method
    canvas.mpl_connect('button_press_event', on_click)
    canvas.mpl_connect('motion_notify_event', on_move)

    canvas.mpl_connect("scroll_event", process_scroll_events)
    
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    redraw()



def on_closing():
    print("Leaving...")
    try:
        save_files()
    except:
        import traceback
        print(traceback.format_exception(*sys.exc_info()))
        e = sys.exc_info()[0]
        print( "Error!! %s" % e )
    gb['root'].destroy()
    sys.exit(0)
    

def quit():
    on_closing()



def redraw_all():
    redraw()





def get_valid_intervals(trange=None):
    # Find the peak-to-trough and trough-to-peak intervals over time
    
    if trange:
        tmin,tmax= trange
    else:
        tmin,tmax= -np.Inf, np.Inf

    # Peak-to-trough and trough-to-peak
    validpeaks = [ p for p in gb['peaks'] if p['valid'] ] # Take only the valid peaks
    validpeaks.sort(key=lambda p: p['t']) # sort them in time
    inv = gb['invalid']
    united = []
    #print(inv)
    
    for i in range(len(validpeaks)-1):
        peak = validpeaks[i]
        t = peak['t']
        peakkind = peak.get('kind','')
        if t>=tmin and t<=tmax:
            nextpeak = validpeaks[i+1]
            nextpeakkind=nextpeak.get('kind','')

            ## Check that this does not fall into invalid regions
            accepted = True
            for (s,t0,t1) in inv:
                if s==gb['channel'] and does_overlap((t0,t1),(t,nextpeak['t'])):
                    ## Oops, this falls into the invalid range!
                    accepted = False

            kind = 'none'
            if peakkind=='peak'   and nextpeakkind=='trough': kind = 'peak-to-trough'
            if peakkind=='trough' and nextpeakkind=='peak':   kind = 'trough-to-peak'
            thist = np.around(peak['t'],5)
            if accepted:
                iv = np.around(nextpeak['t']-peak['t'],5) # interval to next
            else:
                iv = np.nan
            united.append((thist,iv,kind))
    
    return united





    


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
        


    
def process_key_events(event):
    #if event.key=='left': back_in_time()
    #if event.key=='right': forward_in_time()

    if event.char=='z':
        toggle_zoom()
    if event.char=='a':
        print("Zooming to all.")
        tmax = max(gb['t'])
        gb['tstart']=0
        gb['WINDOW_T']=tmax
        update_window_definitions()
        redraw_all()




# When toggling the zoom, toggle between micro and macro window size
TOGGLE_WINDOW_SIZES = [ 8, 40 ]
            

def toggle_zoom():
    # Switch between zoom modes: micro and macro, to allow quick zooming
    wint = gb['WINDOW_T']
    sizedist = [ np.abs(np.log(wint/t)) for t in TOGGLE_WINDOW_SIZES ]
    target = TOGGLE_WINDOW_SIZES[np.argmax(sizedist)]
    print("Toggling zoom")
    if False:
        print("Toggling!")
        print(wint)
        print(sizedist)
        print(target)
    update_window(target/wint,gb['cursor.t'])
    

      

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
    return int(np.floor(max(gb['t'])/gb['WINDOW_T']))

def update_window_definitions():
    # If the window width has changed, cascade the necessary updates
    nwind = get_n_windows()
    gb['slider'].configure(to=nwind)
    

# Globals to carry around
gb = {}





def main():


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


    print("Opening file {}".format(fname))
    bio = bb.load(fname)
    bio.print()

    gb['bio']=bio
    gb['peaks']=[]





    import biotop.misc as misc
    fields = bio.find_channels()
    fields.sort()
    if len(fields)>1:

        if len(sys.argv)>2:
            pc = sys.argv[2]
        else:
            pc = misc.give_choices(fields)
    else:
        pc = fields[0]

    if pc:
        gb['channel'] = pc
        gb['t']=bio.get_time(pc)
        hdr,dat = bio.get(pc)
        gb['SR']=hdr['sampling_frequency']
        gb['hdr']=hdr

        # Also make a filtered version
        sos = butter(15, 3, 'low', fs=gb['SR'], output='sos')
        filtd = signal.sosfiltfilt(sos, dat)
        #gb['filtered']=filtd
        gb['raw']=dat
        gb['signal']=filtd

    else:
        sys.exit(-1)




    # Main window dimensions
    window_w,window_h=1300,400

    gb["WINDOW_T"] =25 # default window width in seconds
    gb['WINDOW_SHIFT_T'] =.03 # proportion of the window to shift when we go back and forth in time

    gb['qc']={}
    gb['invalid']= []
    gb['cursor.t']=0
    gb['cursor.snap.t']=0
    gb['cursor.snap']=None
    gb['tstart']=0 # the left edge of the window we're currently showing
    gb['mark_in']  =None
    gb['mark_out'] =None

    # See if there is an existing peak file
    JSON_OUT = '{}_respire.json'.format(fname)
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT,'r') as f:
            gb['qc']      = json.loads(f.read())
            gb['invalid'] = gb['qc'].get('invalid',{}).get(gb['channel'],[])
            gb['peaks']   = gb['qc'].get('peaks',{}).get(gb['channel'],[])
    gb['JSON_OUT']=JSON_OUT

    # Reconstruct peak samples
    for p in gb['peaks']:
        p['i']=int(round(p['t']*gb['SR']))
        p['y']=get_signal_at_t(p['t'])


    # Curate invalid
    gb['invalid'] = curate_invalid(gb['invalid'])



    root = tkinter.Tk()
    root.wm_title("Respiration Picker - {}".format(fname))
    root.geometry('{}x{}'.format(window_w,window_h))
    gb['root']=root


    gb['COLORS'] = {}

    

    build_gui(root)
    make_plot()
    tkinter.mainloop()

    




