"""validate.py -- reproduce a September numpy run step-for-step.

Starts from the stored start configuration of rotor misfit_alpha0.1 (alpha1/k_a = 0.1,
overdamped, gamma = 1, dt as stored, T = 0, frozen topology) and compares the JAX
trajectory with the stored frames (stride 500 steps, float32).
    python validate.py /path/to/misfit_alpha0.1.npz /path/to/misfit_alpha0.1_params.yaml
"""
import sys, time, yaml
import numpy as np, jax, jax.numpy as jnp
import eadyn as E

npz, par = sys.argv[1], sys.argv[2]
d = np.load(npz); p = yaml.safe_load(open(par))
dt = p["dt"]; tris = d["triangles"]; R = d["trajectory_R"].astype(float)
f = E.make_force(tris, dict(alpha1=p["alpha1"]), n=len(d["md_R"]))
chunk = E.make_chunk(f, "overdamped", dt)
P = jnp.asarray(d["md_R"].astype(float)); V = jnp.zeros_like(P); key = jax.random.PRNGKey(0)
print("frame0 vs md_R: %.2e" % np.abs(R[0] - d["md_R"]).max())
jax.block_until_ready(chunk(P, V, key, 1))       # compile outside the timer
t0 = time.time()
for k in range(1, 21):
    P, V, key = chunk(P, V, key, 500)
    if k in (1, 2, 5, 10, 20):
        print("step %6d  max|dR| vs stored = %.2e   (|R| ~ %.1f)" % (500 * k, np.abs(np.asarray(P) - R[k]).max(), np.abs(R[k]).max()))
el = time.time() - t0
print("%.1f us/step (N=%d, %d triangles, %s)" % (el / 10000 * 1e6, len(P), len(tris), jax.devices()[0].platform))
