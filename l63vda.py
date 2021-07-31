import numpy as np

p = 10.0
r = 32.0
b = 8 / 3
dt = 0.01
nstop = 200.
miss = 999.999


def florenz(x, p, r, b):
    x1, x2, x3 = x
    y = np.zeros_like(x)
    y[0] =       -p * x1 + p * x2 
    y[1] = (r - x3) * x1 -     x2
    y[2] =            x1 * x2     - b * x3
    return y 


def tlorenz(xt, xb, p, r, b):
    dx = np.zeros_like(xt)
    x1, x2, x3 = xt
    xb1, xb2, xb3 = xb
    y[0] =       -p  * x1 +   p * x2 
    y[1] = (r - xb3) * x1 -       x2 - xb1 * x3
    y[2] =       xb2 * x1 + xb1 * x2 -   b * x3
    return y


def alorenz(a, y, xb, p, r, b):
    xb1, xb2, xb3 = xb
    y1, y2, y3 = y 
    a[0] += -p * y1 + (r - xb3) * y2 + xb2 * y3
    a[1] +=  p * y1 -             y2 + xb1 * y3
    a[2] +=         -      xb1 *  y2 -   b * y3
    y[:] = 0.0
    return a, y


def fom(nexp, ncyc, x, param):
    p, r, b, dt, nstop = param
    xx = np.zeros([nstop+1, x.size])
    xx[0, :] = x
    fname = f"m{nexp:02}c{ncyc:03}.npy"
    for ntim in range(1, nstop+1):
        x += dt * florenz(x, p, r, b)
        xx[ntim, :] = x
    np.save(fname, xx)
    return x


def tlm(nexp, ncyc, tl, param):
    p, r, b, dt, nstop = param
    fname = f"m{nexp:02}c{ncyc-1:03}.npy"
    xb = np.load(fname)
    for ntim in range(nstop):
        tl += dt * tlorenz(tl, xb[ntim], p, r, b)
    return tl


def calc_innovation(ntim, nexp, ncyc):
    bfile = f"m{nexp:02}c{ncyc:03}.npy"
    xb = np.load(bfile)
    ofile = f"o{nexp:02}.npy"
    xo = np.load(ofile)
    ndim = xb[ntim,:].size)
    d = np.zeros(ndim)
    for i in range(n):
        if xb[ntim, i] == miss or xo[ntim, i] == miss:
            d[i] = 0.0
        else:
            d[i] = xb[ntim, i] - xo[ntim, i]
    return d


def calc_cost(nexp, ncyc, nstop):
    cost = 0.0
    for ntim in range(nstop+1):
        d = calc_innovation(ntim, nexp, ncyc)
        cost += d[:]**2
    cost *= 0.5
    return cost


def gen_obs(nexp, nstop, iobs):
    rng = np.random.default_rng
    efile = f"e{nexp:02}.txt"
    e = np.loadtxt(efile)
    ifile = f"i{nexp:02}.txt"
    x0 = np.load.txt(ifile)
    _ = fom(nexp, 0, x0, param)
    mfile = f"m{nexp:02}c00.npy"
    xo = np.load(mfile)
    for ntim in range(nstop+1):
        if ntim > 0 and np.mod(ntim, iobs) == 0:
            xo[ntim, :] += e[:] * 2.0 * (rng.random(e.size) - 0.5)
        else:
            xo[ntim, :] = miss
    ofile = f"o{nexp:02}.npy"
    np.save(ofile, xo)
    return calc_cost(nexp, 0, nstop)


def adm(nexp, ncyc, a, param, ltest=False):
    p, r, b, dt, nstop = param
    y = np.zeros(3)
    fname = f"m{nexp:02}c{ncyc-1:03}.npy"
    ab = np.load(fname)
    for ntim in range(nstop, 0, -1):
        if not ltest:
            d = calc_innovation(ntim, nexp, ncyc-1)
            a += d
        a, y = alorenz(a, y, ab[ntim], p, r, b)


def gen_true(nexp):
    ifile = f"i{nexp:02}.txt"
    x0 = np.load.txt(ifile)
    return fom(0, x0)


def test_tlm(nexp):
