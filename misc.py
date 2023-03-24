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
    choicewin.title("ChoiceBox")

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







def does_overlap(intv1,intv2):
    # Return whether the two intervals overlap
    (a1,b1)=intv1
    (a2,b2)=intv2
    overlapmin = max([a1,a2])
    overlapmax = min([b1,b2])
    return overlapmin<=overlapmax

    
