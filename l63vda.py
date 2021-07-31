import sys
import numpy as np

p = 10.0
r = 32.0
b = 8 / 3
nstop = 200
dt = 0.01
miss = 999.999


def florenz(w, p, r, b):
    x, y, z = w
    dw = np.zeros_like(w)
    dw[0] =      -p * x + p * y 
    dw[1] = (r - z) * x -     y 
    dw[2] =           x * y     - b * z
    return dw


def tlorenz(wt, wb, p, r, b):
    dw = np.zeros_like(wt)
    xt, yt, zt = wt
    xb, yb, zb = wb
    dw[0] =       -p * xt +  p * yt 
    dw[1] = (r - zb) * xt -      yt - xb * zt
    dw[2] =       yb * xt + xb * yt -  b * zt
    return dw


def alorenz(wa, dwa, wb, p, r, b):
    xb, yb, zb = wb
    dxa, dya, dza = dwa
    wa[0] += -p * dxa + (r - zb) * dya + yb * dza
    wa[1] +=  p * dxa -            dya + xb * dza
    wa[2] +=          -       xb * dya -  b * dza
    dwa[:] = 0.0
    return wa, dwa


def fom(nexp, ncyc, w, param):
    p, r, b, dt, nstop = param
    x = np.zeros([nstop+1, w.size])
    x[0, :] = w
    fname = f"m{nexp:02}c{ncyc:03}.npy"
    for ntim in range(1, nstop+1):
        w += dt * florenz(w, p, r, b)
        x[ntim, :] = w
    np.save(fname, x)
    return w


def tlm(nexp, ncyc, tl, param):
    p, r, b, dt, nstop = param
    fname = f"m{nexp:02}c{ncyc-1:03}.npy"
    wb = np.load(fname)
    for ntim in range(nstop):
        tl += dt * tlorenz(tl, wb[ntim], p, r, b)
    return tl


def adm(nexp, ncyc, wa, param, ltest=False):
    p, r, b, dt, nstop = param
    dwa = np.zeros(3)
    mfile = f"m{nexp:02}c{ncyc-1:03}.npy"
    wb = np.load(mfile)
    if not ltest:
        ofile = f"o{nexp:02}.npy"
        wo = np.load(ofile)
    for ntim in range(nstop, 0, -1):
        if not ltest:
            d = calc_innovation(wb[ntim, :], wo[ntim, :])
            wa += d
        dwa += dt * wa
        wa, dwa = alorenz(wa, dwa, wb[ntim-1], p, r, b)
    return wa


def calc_innovation(wb, wo):
    d = np.zeros(wb.size)
    for i in range(wb.size):
        if wb[i] == miss or wo[i] == miss:
            d[i] = 0.0
        else:
            d[i] = wb[i] - wo[i]
    return d


def calc_cost(nexp, ncyc, nstop):
    mfile = f"m{nexp:02}c{ncyc:03}.npy"
    wb = np.load(mfile)
    ofile = f"o{nexp:02}.npy"
    wo = np.load(ofile)
    cost = 0.0
    for ntim in range(nstop+1):
        d = calc_innovation(wb[ntim, :], wo[ntim, :])
        cost += (d[:]**2).sum()
    return cost * 0.5


def gen_obs(nexp, nstop, iobs):
    rng = np.random.default_rng()
    efile = f"e{nexp:02}.txt"
    e = np.loadtxt(efile)
    ifile = f"i{nexp:02}.txt"
    w0 = np.loadtxt(ifile)
    mfile = f"m{nexp:02}c000.npy"
    wo = np.load(mfile)
    for ntim in range(nstop+1):
        if ntim > 0 and np.mod(ntim, iobs) == 0:
            wo[ntim, :] += e[:] * 2.0 * (rng.random(e.size) - 0.5)
        else:
            wo[ntim, :] = miss
    ofile = f"o{nexp:02}.npy"
    np.save(ofile, wo)
    return calc_cost(nexp, 0, nstop)


def gen_true(nexp):
    ifile = f"i{nexp:02}.txt"
    w0 = np.loadtxt(ifile)
    return fom(nexp, 0, w0, param)


def test_tlm(nexp):
    ncyc = 1
    ifile = f"i{nexp:02}.txt"
    tmp = np.loadtxt(ifile)
    tl = tmp * 0.001
    tmp = fom(nexp, ncyc-1, tmp, param)
    fw = tmp + tl
    fw = fom(nexp, ncyc, fw, param)
    fw -= tmp
    print(f"N(w+dw)-N(x) from NL {fw}")
    tl = tlm(nexp, ncyc, tl, param)
    print(f" L(dw) rom TL {tl}")
    ltest = True
    print(f"LX_t LX = {tl @ tl}")
    ad = tl
    ad = adm(nexp, ncyc, ad, param, ltest)
    tmp = np.loadtxt(ifile)
    tl = tmp * 0.001
    pa = tl @ ad
    print(f"X_t (L_t (LX)) = {pa}")


def test_grad(nexp):
    na1, na2 = -17, -2
    ltest = False
    ncyc = 1
    ifile = f"i{nexp:02}.txt"
    fw = np.loadtxt(ifile)
    fw = fom(nexp, ncyc, fw, param)
    cost0 = calc_cost(nexp, ncyc, nstop)
    ncyc = 2
    ad = np.zeros(fw.size)
    ad = adm(nexp, ncyc, ad, param, ltest)
    aa = ad @ ad
    chkgra = np.zeros([na2-na1+1])
    for nalpha in range(na1, na2+1):
        alpha = 10.0 ** nalpha
        ifile = f"i{nexp:02}.txt"
        fw = np.loadtxt(ifile)
        fw -= alpha * ad # gradient descent
        fw = fom(nexp, ncyc, fw, param)
        cost1 = calc_cost(nexp, ncyc, nstop)
        chkgra[nalpha-na1] = -(cost1 - cost0) / (alpha * aa)
        print(f"chkgra(10^{nalpha}) = {chkgra[nalpha-na1]}")
    np.savetxt(f"g{nexp:02}.txt", np.column_stack([np.arange(na1, na2+1), chkgra]))


def run_vda(nexp, nexpi, alpha=5e-4, istart=1, istop=100):
    ltest = False
    if istart < 1:
        sys.exit()
    cost = np.zeros([istop-istart+1])
    for ncyc in range(istart, istop+1):
        if ncyc == 1:
            ifile = f"i{nexpi:02}.txt"
            fw = np.loadtxt(ifile)
        else:
            ad = 0.0
            ad = adm(nexp, ncyc, ad, param, ltest)
            mfile = f"m{nexp:02}c{ncyc-1:03}.npy"
            w = np.load(mfile)
            fw = w[0, :] - alpha * ad
        fw = fom(nexp, ncyc, fw, param)
        cost[ncyc-istart] = calc_cost(nexp, ncyc, nstop)
        print(f"{ncyc}: {cost[ncyc-istart]}")
    np.savetxt(f"j{nexp:02}i{nexpi:02}.txt", np.column_stack(
        [np.arange(istart, istop+1), cost]))

if __name__ == "__main__":
#    nstop = 500
#    r = 10
    param = p, r, b, dt, nstop
#    x = gen_true(1)
#    x = gen_true(2)
#    test_tlm(1) 
#    gen_obs(1, nstop, 60)
#    test_grad(1)
    print(f"Jc={calc_cost(1, 0, nstop)}")
    run_vda(1, 2)
