from tkinter import *
from tkinter import font as tkFont  # for convenience

def give_choices(choicelist):
    global result

    def buttonfn():
        global result
        result = var.get()
        choicewin.quit()

    choicewin = Tk()
    choicewin.resizable(False, False)
    choicewin.title("Select the signal to analyze")

    fnt = tkFont.Font(family='Helvetica', size=15, weight='normal')

    Label(choicewin, text="Please select a data field : ", font=fnt).grid(row=0, column=0, sticky="W")

    var = StringVar(choicewin)
    DEFAULT = "No data"
    var.set(DEFAULT)  # default option
    popupMenu = OptionMenu(choicewin, var, *choicelist)
    popupMenu.grid(sticky=N + S + E + W, row=1, column=0)
    popupMenu.config(font=fnt)

    menu = choicewin.nametowidget(popupMenu.menuname)  # Get menu widget.
    menu.config(font=fnt)  # Set the dropdown menu's font
    
    Button(choicewin, text="Done", command=buttonfn, font = fnt).grid(row=2, column=0)
    choicewin.mainloop()
    try:
        choicewin.destroy()
    except:
        return None
    
    if result==DEFAULT: return None
    else: return result






def in_range(t,intv):
    """ Returns True if t falls in the interval (t0,t1) defined by intv """
    (t0,t1)=intv
    return t>=t0 and t<=t1


def does_overlap(intv1,intv2):
    # Return whether two intervals overlap
    # The intervals are given as (start,end) tuples.
    (a1,b1)=intv1
    if a1>b1: (a1,b1)=(b1,a1) # if coded inversely, flip
    (a2,b2)=intv2
    if a2>b2: (a2,b2)=(b2,a2)
    # So now a1<=b1 and a2<=b2
    overlapmin = max([a1,a2])
    overlapmax = min([b1,b2])
    return overlapmin<=overlapmax





def chop_away(a,b,intvls):
    # Given an interval, (a,b),
    # and a set of intervals, return that set of intervals
    # where for each interval, we have chopped away the part that
    # overlaps with (a,b).
    toret = []

    # Ensure that a<b
    if a>b: (a,b)=(b,a)
    
    for (t0,t1) in intvls:
        if not does_overlap((t0,t1),(a,b)):
            toret.append((t0,t1))
        else:
            # It does overlap... hmmm....
            # So basically we want to keep as much of (t0,t1) as we can,
            # while chipping away any overlap with (a,b)
            
            # So we chop away the (a,b) portion from (t0,t1)
            if a>t0:
                # Then the portion (a,t0) is valid! Yay!
                toret.append((t0,a))
            if t1>b:
                # Then the portion (t1,b) is valid! Yay!
                toret.append((b,t1))
            
    return toret
    










#from tkinter import Tk, Text, TOP, BOTH, X, N, LEFT, RIGHT
#from tkinter.ttk import Frame, Label, Entry, Button

# Good habit to put your GUI in a class to make it self-contained
class SimpleDialog(Frame):

    def __init__(self):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.output1 = ""
        self.output2 = ""
        self.initUI()

    def initUI(self):

        self.master.title("Simple Dialog")
        self.pack(fill=BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        lbl1 = Label(frame1, text="Participant A")
        lbl1.pack(side=LEFT, padx=5, pady=10)

        self.entry1 = Entry(frame1, textvariable=self.output1)
        self.entry1.pack(fill=X, padx=5, expand=True)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        lbl2 = Label(frame2, text="Participant B")
        lbl2.pack(side=LEFT, padx=5, pady=10)

        self.entry2 = Entry(frame2)
        self.entry2.pack(fill=X, padx=5, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=X)

        # Command tells the form what to do when the button is clicked
        btn = Button(frame3, text="Submit", command=self.onSubmit)
        btn.pack(padx=5, pady=10)

    def onSubmit(self):
        self.output1 = self.entry1.get().strip()
        self.output2 = self.entry2.get().strip()
        self.quit()


def run_dual_input():

    # This part triggers the dialog
    root = Tk()
    root.geometry("250x150+300+300")
    app = SimpleDialog()
    root.mainloop()
    # Here we can act on the form components or
    # better yet, copy the output to a new variable
    user_input = (app.output1, app.output2)
    # Get rid of the error message if the user clicks the
    # close icon instead of the submit button
    # Any component of the dialog will no longer be available
    # past this point
    try:
        root.destroy()
    except:
        pass
    # To use data outside of function
    # Can either be used in __main__
    # or by external script depending on
    # what calls main()
    return user_input











import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


    
