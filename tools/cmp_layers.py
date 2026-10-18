import numpy as np, os
T = 410
def cmp(cf, rf, C=192):
    c = np.fromfile(cf, dtype=np.float32)[:C*T].reshape(C, T)
    r = np.fromfile(rf, dtype=np.float32)[:C*T].reshape(C, T)
    d = np.abs(c - r)
    rel = np.linalg.norm(d) / (np.linalg.norm(r) + 1e-9)
    cc = c - c.mean(1, keepdims=True); rr = r - r.mean(1, keepdims=True)
    corr = float((cc*rr).sum() / (np.sqrt((cc**2).sum())*np.sqrt((rr**2).sum()) + 1e-9))
    return rel, corr, c.std(), r.std()
order = ['enc_x_emb','enc_x_L0','enc_x_L1','enc_x_L2','enc_x_L3','enc_x_L4','enc_x_L5','m_p','logs_p']
print('%-14s %10s %8s %8s %8s' % ('stage','relL2','corr','c_std','r_std'))
for b in order:
    cf = 'scratch/%s.bin' % b; rf = 'scratch/ref_%s.bin' % b
    if not (os.path.exists(cf) and os.path.exists(rf)):
        print('%-14s  (missing %s)' % (b, rf)); continue
    rel, corr, cs, rs = cmp(cf, rf)
    print('%-14s %10.4f %8.3f %8.3f %8.3f' % (b, rel, corr, cs, rs))
