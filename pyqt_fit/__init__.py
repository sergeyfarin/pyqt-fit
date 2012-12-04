import plot_fit
import functions
import residuals
from plot_fit import fit, write1d, plot1d
from path import path

with (path(__file__).dirname() / 'version.txt').open() as f:
    __version__ = f.read().strip()

