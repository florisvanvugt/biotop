__version__ = "0.6.0"


from biotop import peak_picker
from biotop import respiration_picker


def ecg():
    # Launch ECG analysis
    import biotop.peak_picker as pp
    pp.main()


def respire():
    # Launch respiration analysis
    import biotop.respiration_picker as rp
    rp.main()



