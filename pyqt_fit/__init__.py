import functions
import residuals
from plot_fit import fit, write1d, plot1d
from curve_fitting import curve_fit
from path import path

with (path(__file__).dirname() / 'version.txt').open() as f:
    __version__ = f.read().strip()

