import numpy as np
import scipy.stats as st

def get_data(n, d, b, sigma, gamma, z_g):
    v = np.hstack((np.ones((n,1)), np.random.uniform(-1,1,size=(n,d-1))))  # (n,d)
    z = z_g.rvs(size=n)  # (n,)
    x = v @ b + sigma * z  # (n,)
    x_reflect = 2 * x + v @ gamma  # (n,)
    return x, v, x_reflect

def ols(v, y):
    y = y.reshape(-1, 1)
    b_hat = np.linalg.lstsq(v, y, rcond=None)[0]  # (d,1)
    residual = y - v @ b_hat
    s_hat = np.sqrt(np.sum(residual**2) / (len(y) - v.shape[1]))
    return b_hat.flatten(), s_hat

def q_boda_on_baseset(v0, v, b_hat_reflect, s_hat_reflect, cu, co):
    alpha = cu / (cu + co)
    n, d = v.shape
    df = n - d
    lev = float(v0 @ np.linalg.inv(v.T @ v) @ v0)
    t_q = st.t.ppf(alpha, df)
    return float(b_hat_reflect @ v0 + t_q * s_hat_reflect * np.sqrt(1 + lev))

# def estimate_delta_beta(v, x, x_reflect):
#     # x 和 x_reflect 都是一维(n,)
#     x_design = np.column_stack((x_reflect, v))  # (n, 1+d)
#     params = np.linalg.lstsq(x_design, x, rcond=None)[0]  # (1+d,)
#     delta = float(params[0])
#     beta = params[1:]
#     return delta, beta

def cost(q, demand, cu, co):
    return cu * np.maximum(demand - q, 0) + co * np.maximum(q - demand, 0)


# np.random.seed(0)
n, d = 60, 4
b_set = np.array([30., 5., -2., 3.])
sigma_set = 10
gamma_set = np.array([0., 1., 0.5, -1.0])
cu, co = 0.2, 0.8
z_g = st.norm

x_set, v, x_reflect = get_data(n, d, b_set, sigma_set, gamma_set, z_g)

# x = 0.5 * x_reflect - 0.5 * (v @ gamma_set)）

# 1. ES
b_hat_reflect, s_hat_reflect = ols(v, x_reflect)
v0 = np.ones(d)
q_es = float(b_hat_reflect @ v0)

# 2. q_boda
q_boda = q_boda_on_baseset(v0, v, b_hat_reflect, s_hat_reflect, cu, co)

# 3. 反向映射
q_oda_direct = 0.5 * q_boda - 0.5 * (gamma_set @ v0)

# 4. 计算cost
z_test = z_g.rvs(size=20000)
D_v0 = float(b_set @ v0) + sigma_set * z_test

cost_es = np.mean(cost(q_es, D_v0, cu, co))
cost_boda = np.mean(cost(q_boda, D_v0, cu, co))
cost_oda_direct = np.mean(cost(q_oda_direct, D_v0, cu, co))

print(f"q_ES(ols)={q_es:.3f}, cost = {cost_es:.3f}")
print(f"q_ODA_direct (mapped back to x)={q_oda_direct:.3f}, cost = {cost_oda_direct:.3f}")
