"""
core/math_engine.py
Математическое ядро: решение задачи теплопроводности методом КИП
"""
import numpy as np

def mu_n(n, l):
    return np.pi * n / l

def beta_n(n):
    return 0.5 if n == 0 else 1.0

def source_rect(eta, q_max, y1, y2):
    return np.where((eta > y1) & (eta <= y2), q_max, 0.0)

def source_gauss(eta, q_max, k, l):
    return q_max * np.exp(-k * (eta - l / 2) ** 2)

def compute_temperature_detailed(y_points, t_val, l, a, c_V, lam, q_func, t_source_end,
                                 phi_func=None, q1_func=None, q2_func=None, N_terms=80, N_eta=600, N_tau=400, return_components=False):
    eta = np.linspace(0, l, N_eta)
    q_eta = q_func(eta)
    t_active = min(t_val, t_source_end)
    
    if phi_func is not None:
        phi_arr = phi_func(eta)
    else:
        phi_arr = np.zeros(N_eta)
        
    T = np.zeros(len(y_points))
    T_A = np.zeros(len(y_points))
    T_B = np.zeros(len(y_points))
    T_C = np.zeros(len(y_points))
    
    for n in range(N_terms + 1):
        mn = mu_n(n, l)
        bn = beta_n(n)
        cos_y = np.cos(mn * y_points)
        cos_eta = np.cos(mn * eta)
        
        phi_n_coef = np.trapezoid(phi_arr * cos_eta, eta)
        decay_A = np.exp(-a * mn**2 * t_val) if mn > 0 else 1.0
        term_A = phi_n_coef * decay_A
        
        term_B = 0.0
        if q1_func is not None or q2_func is not None:
            tau_B = np.linspace(0, t_val, N_tau)
            q1_arr = q1_func(tau_B) if q1_func is not None else np.zeros(N_tau)
            q2_arr = q2_func(tau_B) if q2_func is not None else np.zeros(N_tau)
            sign_n = (-1) ** n
            integ_B = (q1_arr + q2_arr * sign_n) * np.exp(-a * mn**2 * (t_val - tau_B))
            I_B = np.trapezoid(integ_B, tau_B)
            term_B = -(a / lam) * I_B
            
        term_C = 0.0
        if t_active > 0:
            I_eta = np.trapezoid(q_eta * cos_eta, eta)
            tau_C = np.linspace(0, t_active, N_tau)
            exp_tau = np.exp(-a * mn**2 * (t_val - tau_C))
            I_tau = np.trapezoid(exp_tau, tau_C)
            term_C = (1.0 / c_V) * I_eta * I_tau
            
        component = (2.0 / l) * bn * (term_A + term_B + term_C) * cos_y
        T += component
        
        if return_components:
            T_A += (2.0 / l) * bn * term_A * cos_y
            T_B += (2.0 / l) * bn * term_B * cos_y
            T_C += (2.0 / l) * bn * term_C * cos_y
            
    if return_components:
        return {"total": T, "A": T_A, "B": T_B, "C": T_C}
    return T

def compute_spatial_field(y_arr, t_arr, **kwargs):
    T_field = np.zeros((len(t_arr), len(y_arr)))
    for i, t in enumerate(t_arr):
        T_field[i, :] = compute_temperature_detailed(y_arr, t, **kwargs)
    return T_field