"""
core/analytics.py
Анализ результатов моделирования: метрики, структура, ширины
"""
import numpy as np

def compute_t85(t_arr, T_cycle):
    if T_cycle.max() < 800:
        return None
    idx_max = np.argmax(T_cycle)
    T_cool = T_cycle[idx_max:]
    t_cool = t_arr[idx_max:]
    
    above_800 = T_cool >= 800
    below_800 = T_cool < 800
    t_800 = None
    for i in range(len(T_cool)-1):
        if above_800[i] and below_800[i+1]:
            t_800 = t_cool[i] + (800 - T_cool[i]) * (t_cool[i+1] - t_cool[i]) / (T_cool[i+1] - T_cool[i])
            break
            
    above_500 = T_cool >= 500
    below_500 = T_cool < 500
    t_500 = None
    for i in range(len(T_cool)-1):
        if above_500[i] and below_500[i+1]:
            t_500 = t_cool[i] + (500 - T_cool[i]) * (t_cool[i+1] - t_cool[i]) / (T_cool[i+1] - T_cool[i])
            break
            
    if t_800 is not None and t_500 is not None:
        return t_500 - t_800
    return None

def compute_weld_width(y_arr, T_field, T_melt):
    T_max_profile = np.max(T_field, axis=0)
    melted = T_max_profile >= T_melt
    if not np.any(melted):
        return 0.0
    melted_indices = np.where(melted)[0]
    return y_arr[melted_indices[-1]] - y_arr[melted_indices[0]]

def compute_haz_width(y_arr, T_field, T_harden, material_name=""):
    T_harden_eff = T_harden
    is_recrist = False
    if T_harden <= 0:
        if "Медь" in material_name or "Алюминий" in material_name:
            T_harden_eff = 300.0
            is_recrist = True
        else:
            return 0.0, "N/A (нет закалки)"
            
    T_max_profile = np.max(T_field, axis=0)
    haz = T_max_profile >= T_harden_eff
    if not np.any(haz):
        return 0.0, "0.00 см"
        
    haz_indices = np.where(haz)[0]
    width = y_arr[haz_indices[-1]] - y_arr[haz_indices[0]]
    
    if is_recrist:
        return width, f"{width:.3f} см (T_рекр)"
    if width < 0.01:
        return width, f"{width*10:.2f} мм"
    return width, f"{width:.3f} см"

def predict_structure(t85, T_max, T_melt):
    if t85 is None:
        return "Недостаточно данных", "gray"
    if T_max < T_melt:
        return "Без плавления", "blue"
    if t85 < 2:
        return "Мартенсит (хрупкий)", "red"
    elif t85 < 10:
        return "Мартенсит + Бейнит", "orange"
    elif t85 < 30:
        return "Бейнит", "yellow"
    elif t85 < 100:
        return "Перлит + Феррит", "green"
    else:
        return "Грубый феррит", "lightgreen"