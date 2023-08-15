import jax
import jax.numpy as jnp
import sys

def florenz(w, p, r, b):
    x, y, z = w
    dwx =      -p * x + p * y 
    dwy = (r - z) * x -     y 
    dwz =           x * y     - b * z
    return jnp.array([dwx, dwy, dwz])

def fom(w, param):
    p, r, b, dt, nstop = param
    fw = []
    fw.append(w)
    for ntim in range(1, nstop+1):
        w = w + dt * florenz(w, p, r, b)
        fw.append(w)
    return jnp.array(fw)

def calc_cost(w, wo, tobs, param):
    wb = fom(w, param)
    return 0.5 * ((wb[tobs] - wo)**2).sum()

def update(w, wo, tobs, param, alpha=5.0e-4):
    c, g = jax.value_and_grad(calc_cost)(w, wo, tobs, param)
    return w - alpha * g, c

def gen_obs(e, wt, iobs, seed=514):
    key = jax.random.PRNGKey(seed)
    wo = []
    tobs = jnp.array([i for i in range(iobs, wt.shape[0], iobs)])
    for i in range(iobs, wt.shape[0], iobs):
        wo.append(wt[i, :] + e[:] * 2.0 * (jax.random.uniform(key, [e.size,]) - 0.5))
    return jnp.array(wo), tobs

if __name__ == "__main__":
#    from jax import config
#    config.update("jax_enable_x64", True)

    itermax = 20

    p, r, b = 10.0, 32.0, 8. / 3.
    nstop = 200
    dt = 0.01   
    param = p, r, b, dt, nstop
    iobs = 60
    alpha = 5.0e-4

    w0 = jnp.array([1.0, 3.0, 5.0])
    e = w0 * 0.1

    wt = fom(w0, param)
    wo, tobs = gen_obs(e, wt, iobs)
    w = w0 * 1.1
    cost = []
    print(calc_cost(w0, wo, tobs, param))
    for i in range(itermax):
        w, c = update(w, wo, tobs, param, alpha)
        cost.append(c)
        print(i, w, c)
