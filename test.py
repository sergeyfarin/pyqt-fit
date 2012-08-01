import cyth

def test(x, (p0,p1,p2)):
    return p0 + p1*x + p2*x**2

target = (10,4,1.2)
x = 6*rand(2000) - 3
y = test(x, target)*(1+0.2*randn(x.shape[0]))
xr = arange(-5, 4, 0.001)
import kernel_smoothing_c
est = kernel_smoothing_c.SpatialAverage(x,y)
xdens = est.density(xr)
figure()
#plot(x, y, '+')
plot(xr, xdens)
yr = est(xr)
figure()
plot(x, y, '+')
plot(xr, yr)
import kernel_smoothing
est1 = kernel_smoothing.SpatialAverage(x,y)
yr1 = est1(xr)
figure()
plot(x, y, '+')
plot(xr, yr1)
from scipy.stats import gaussian_kde
kde = gaussian_kde(x)
xdens2 = kde(xr)
figure()
plot(xr, xdens, 'g')
plot(xr, xdens2, 'r--')

# 100 loops, best of 3: 12.6 ms per loop
# 
