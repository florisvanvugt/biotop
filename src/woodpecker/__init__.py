
from woodpecker import peak_picker
from woodpecker import respiration_picker


def ecg():
    # Launch ECG analysis
    import woodpecker.peak_picker as pp
    pp.main()


def respire():
    # Launch respiration analysis
    import woodpecker.respiration_picker as rp
    rp.main()



def main():
    """Entry point for the application script"""
    print("Call your main application code here")
