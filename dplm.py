"""
МОДЕЛИРОВАНИЕ ТЕПЛОВОГО ПРОЦЕССА СВАРКИ ПЛАСТИН
Версия 4.0 - С CCT-диаграммой и металлургическим анализом

Новые функции:
• CCT-диаграмма с наложением термического цикла
• Расчёт t₈/₅ (время охлаждения 800→500°C)
• Ширина шва (по изотерме T_пл)
• Ширина ЗТВ (по изотерме T_зак)
• Предсказание структуры шва
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import json
import os
import csv
import warnings
warnings.filterwarnings("ignore")

COLORS = {
    "bg": "#0f1117", "bg2": "#1a1d27", "bg3": "#232736",
    "border": "#2e3347", "accent": "#4f8ef7", "accent2": "#f7954f",
    "accent3": "#4ff7a0", "accent4": "#f74f7a", "text": "#e8ecf4",
    "text2": "#8a93aa", "warning": "#f7c14f", "melt": "#ff4444",
    "header": "#1e2235",
}

CMAP_HEAT = "plasma"

MATERIALS_DB = {
    "Сталь 20": {"c_V": 4.2, "a": 0.1, "lam": 0.5, "T_melt": 1510, "T_harden": 723, "density": 7.85, "desc": "Конструкционная углеродистая сталь"},
    "Сталь 45": {"c_V": 4.1, "a": 0.09, "lam": 0.48, "T_melt": 1490, "T_harden": 723, "density": 7.85, "desc": "Среднеуглеродистая сталь"},
    "Сталь 09Г2С": {"c_V": 4.0, "a": 0.11, "lam": 0.45, "T_melt": 1480, "T_harden": 723, "density": 7.82, "desc": "Низколегированная сталь"},
    "Алюминий АМг5": {"c_V": 2.4, "a": 0.8, "lam": 1.2, "T_melt": 660, "T_harden": 0, "density": 2.7, "desc": "Алюминиево-магниевый сплав"},
    "Титан ВТ6": {"c_V": 3.6, "a": 0.04, "lam": 0.15, "T_melt": 1660, "T_harden": 995, "density": 4.43, "desc": "Титановый сплав (Ti-6Al-4V)"},
    "Медь М1": {"c_V": 3.45, "a": 1.1, "lam": 3.8, "T_melt": 1083, "T_harden": 0, "density": 8.94, "desc": "Электролитическая медь"},
}

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

def compute_t85(t_arr, T_cycle):
    """Расчёт t₈/₅ - времени охлаждения от 800°C до 500°C"""
    idx_max = np.argmax(T_cycle)
    T_cool = T_cycle[idx_max:]
    t_cool = t_arr[idx_max:]
    
    # Найти время прохождения через 800°C
    above_800 = T_cool >= 800
    below_800 = T_cool < 800
    t_800 = None
    for i in range(len(T_cool)-1):
        if above_800[i] and below_800[i+1]:
            # Линейная интерполяция
            t_800 = t_cool[i] + (800 - T_cool[i]) * (t_cool[i+1] - t_cool[i]) / (T_cool[i+1] - T_cool[i])
            break
    
    # Найти время прохождения через 500°C
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
    """Расчёт ширины шва по изотерме плавления"""
    T_max_profile = np.max(T_field, axis=0)
    melted = T_max_profile >= T_melt
    if not np.any(melted):
        return 0.0
    melted_indices = np.where(melted)[0]
    return y_arr[melted_indices[-1]] - y_arr[melted_indices[0]]

def compute_haz_width(y_arr, T_field, T_harden):
    """Расчёт ширины ЗТВ по изотерме T_зак (AC₁)"""
    if T_harden <= 0:
        return 0.0
    T_max_profile = np.max(T_field, axis=0)
    haz = T_max_profile >= T_harden
    if not np.any(haz):
        return 0.0
    haz_indices = np.where(haz)[0]
    return y_arr[haz_indices[-1]] - y_arr[haz_indices[0]]

def predict_structure(t85, T_max, T_melt):
    """Предсказание структуры на основе t₈/₅"""
    if t85 is None:
        return "Недостаточно данных", "gray"
    
    if T_max < T_melt:
        return "Без плавления", "blue"
    
    # Критические значения t₈/₅ для углеродистых сталей
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

def styled_entry(parent, textvariable, width=12):
    return tk.Entry(parent, textvariable=textvariable, width=width, font=("Consolas", 10),
                    bg=COLORS["bg3"], fg=COLORS["text"], insertbackground=COLORS["accent"],
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=COLORS["border"], highlightcolor=COLORS["accent"])

def styled_button(parent, text, command, color=None, **kw):
    bg = color or COLORS["accent"]
    return tk.Button(parent, text=text, command=command, font=("Consolas", 10, "bold"),
                     bg=bg, fg="#ffffff", activebackground=COLORS["accent2"],
                     activeforeground="#ffffff", relief="flat", bd=0, padx=14, pady=7, cursor="hand2", **kw)

def separator(parent, color=None, orient="horizontal", **kw):
    if orient == "vertical":
        return tk.Frame(parent, width=1, bg=color or COLORS["border"], **kw)
    return tk.Frame(parent, height=1, bg=color or COLORS["border"], **kw)

def section_frame(parent, title=None):
    outer = tk.Frame(parent, bg=COLORS["bg2"], highlightthickness=1, highlightbackground=COLORS["border"])
    if title:
        hdr = tk.Frame(outer, bg=COLORS["header"])
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"  {title}", font=("Consolas", 10, "bold"),
                 fg=COLORS["accent"], bg=COLORS["header"], pady=6).pack(side="left")
    inner = tk.Frame(outer, bg=COLORS["bg2"])
    inner.pack(fill="both", expand=True, padx=12, pady=8)
    return outer, inner

def apply_mpl_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS["bg2"])
    ax.tick_params(colors=COLORS["text2"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    ax.set_xlabel(xlabel, color=COLORS["text2"], fontsize=9)
    ax.set_ylabel(ylabel, color=COLORS["text2"], fontsize=9)
    ax.set_title(title, color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)

class WeldingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Тепловой процесс сварки — Метод рядов Фурье")
        self.geometry("1440x920")
        self.minsize(1200, 760)
        self.configure(bg=COLORS["bg"])
        self._load_custom_materials()
        self._init_params()
        self._build_ui()
        self.after(300, self._run_all)

    def _init_params(self):
        self.var_l = tk.DoubleVar(value=10.0)
        self.var_a = tk.DoubleVar(value=0.1)
        self.var_cV = tk.DoubleVar(value=4.2)
        self.var_lam = tk.DoubleVar(value=0.5)
        self.var_T_melt = tk.DoubleVar(value=1510.0)
        self.var_T_harden = tk.DoubleVar(value=723.0)
        self.var_T_init = tk.DoubleVar(value=20.0)
        self.var_density = tk.DoubleVar(value=7.85)
        self.var_src_type = tk.StringVar(value="rect")
        self.var_q_max = tk.DoubleVar(value=1e4)
        self.var_y1 = tk.DoubleVar(value=4.5)
        self.var_y2 = tk.DoubleVar(value=5.5)
        self.var_k_gauss = tk.DoubleVar(value=5.0)
        self.var_t_source = tk.DoubleVar(value=5.0)
        self.var_q1 = tk.DoubleVar(value=0.0)
        self.var_q2 = tk.DoubleVar(value=0.0)
        self.var_t_max = tk.DoubleVar(value=60.0)
        self.var_N_t = tk.IntVar(value=120)
        self.var_y_A = tk.DoubleVar(value=4.5)
        self.var_N_terms = tk.IntVar(value=80)
        self._y_arr = None
        self._t_arr = None
        self._T_field = None
        self._T_cycle = None
        self._params = None
        self.cbar_heatmap = None
        self.cbar_gradient = None

    def _load_custom_materials(self):
        if os.path.exists("custom_materials.json"):
            try:
                with open("custom_materials.json", "r", encoding="utf-8") as f:
                    custom = json.load(f)
                MATERIALS_DB.update(custom)
            except Exception as e:
                print(f"Warning: {e}")

    def _save_custom_materials(self):
        try:
            with open("custom_materials.json", "w", encoding="utf-8") as f:
                json.dump(MATERIALS_DB, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["header"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  МОДЕЛИРОВАНИЕ ТЕПЛОВОГО ПРОЦЕССА СВАРКИ",
                 font=("Consolas", 13, "bold"), fg=COLORS["accent"], bg=COLORS["header"]).pack(side="left", padx=20, pady=14)
        tk.Label(hdr, text="Метод рядов Фурье", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["header"]).pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="Готов к расчёту")
        status_bar = tk.Frame(self, bg=COLORS["bg3"], height=28)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var, font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg3"], anchor="w").pack(side="left", padx=12, pady=5)
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg=COLORS["bg"], width=340)
        left.pack(fill="y", side="left")
        left.pack_propagate(False)
        self._build_params_panel(left)
        separator(main, orient="vertical").pack(side="left", fill="y", padx=0)
        right = tk.Frame(main, bg=COLORS["bg"])
        right.pack(fill="both", expand=True)
        self._build_tabs(right)

    def _build_params_panel(self, parent):
        scrollbar = ttk.Scrollbar(parent, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        canvas = tk.Canvas(parent, bg=COLORS["bg"], highlightthickness=0, yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["bg"])
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)
        scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        def _scroll(e):
            if e.delta:
                canvas.yview_scroll(-1 * (e.delta // 120), "units")
            elif getattr(e, "num", 0) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(e, "num", 0) == 5:
                canvas.yview_scroll(1, "units")
        def _bind_scroll_recursive(widget):
            widget.bind("<MouseWheel>", _scroll, add="+")
            widget.bind("<Button-4>", _scroll, add="+")
            widget.bind("<Button-5>", _scroll, add="+")
            for child in widget.winfo_children():
                _bind_scroll_recursive(child)
        def _setup_scroll(*_):
            _bind_scroll_recursive(scroll_frame)
            canvas.bind("<MouseWheel>", _scroll, add="+")
        scroll_frame.bind("<Map>", lambda e: self.after(200, _setup_scroll))
        self._rebind_scroll = _setup_scroll
        p = scroll_frame
        tk.Label(p, text=" ПАРАМЕТРЫ МОДЕЛИ", font=("Consolas", 11, "bold"), fg=COLORS["accent"], bg=COLORS["bg"]).pack(anchor="w", padx=10, pady=(12,4))
        frm_mat, inn_mat = section_frame(p, "Выбор материала")
        frm_mat.pack(fill="x", padx=8, pady=4)
        self.var_material = tk.StringVar(value="Сталь 20")
        self.material_combo = ttk.Combobox(inn_mat, textvariable=self.var_material, values=list(MATERIALS_DB.keys()), state="readonly", width=32)
        self.material_combo.pack(pady=5)
        self.material_combo.bind("<<ComboboxSelected>>", self._on_material_selected)
        styled_button(inn_mat, "↻ Применить свойства", self._apply_material_props, color=COLORS["accent3"]).pack(pady=5)
        self.mat_desc_label = tk.Label(inn_mat, text="", font=("Consolas", 8), fg=COLORS["text2"], bg=COLORS["bg2"], wraplength=300, justify="left")
        self.mat_desc_label.pack(pady=5)
        styled_button(inn_mat, "💾 Сохранить как новый материал", self._save_custom_material, color=COLORS["bg3"]).pack(pady=5)
        self._on_material_selected()
        frm, inn = section_frame(p, "Теплофизические свойства")
        frm.pack(fill="x", padx=8, pady=4)
        self._param_row(inn, "l, см (длина)", self.var_l)
        self._param_row(inn, "a, см²/с", self.var_a)
        self._param_row(inn, "c_V, Дж/(см³·°C)", self.var_cV)
        self._param_row(inn, "λ, Вт/(см·°C)", self.var_lam)
        self._param_row(inn, "ρ, г/см³", self.var_density)
        self._param_row(inn, "T_пл, °C", self.var_T_melt)
        self._param_row(inn, "T_зак, °C (AC₁)", self.var_T_harden)
        self._param_row(inn, "T₀, °C (нач. темп.)", self.var_T_init)
        frm2, inn2 = section_frame(p, "Источник тепла q(y,t)")
        frm2.pack(fill="x", padx=8, pady=4)
        tk.Label(inn2, text="Форма источника:", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg2"]).pack(anchor="w")
        rb_frame = tk.Frame(inn2, bg=COLORS["bg2"])
        rb_frame.pack(fill="x", pady=2)
        for val, txt in [("rect", "Прямоугольный"), ("gauss", "Гауссов")]:
            tk.Radiobutton(rb_frame, text=txt, variable=self.var_src_type, value=val, font=("Consolas", 9), fg=COLORS["text"], bg=COLORS["bg2"], selectcolor=COLORS["bg3"], activebackground=COLORS["bg2"], command=self._toggle_source_fields).pack(side="left", padx=6)
        self._param_row(inn2, "q_max, Вт/см³", self.var_q_max)
        self._param_row(inn2, "t_ист, с", self.var_t_source)
        self.rect_frame = tk.Frame(inn2, bg=COLORS["bg2"])
        self.rect_frame.pack(fill="x")
        self._param_row(self.rect_frame, "y₁, см", self.var_y1)
        self._param_row(self.rect_frame, "y₂, см", self.var_y2)
        self.gauss_frame = tk.Frame(inn2, bg=COLORS["bg2"])
        self.gauss_frame.pack(fill="x")
        self._param_row(self.gauss_frame, "k, 1/см²", self.var_k_gauss)
        self._toggle_source_fields()
        frm3, inn3 = section_frame(p, "Граничные потоки")
        frm3.pack(fill="x", padx=8, pady=4)
        self._param_row(inn3, "q₁, Вт/см² (y=0)", self.var_q1)
        self._param_row(inn3, "q₂, Вт/см² (y=l)", self.var_q2)
        tk.Label(inn3, text="(< 0 = охлаждение, > 0 = подогрев)", font=("Consolas", 8), fg=COLORS["warning"], bg=COLORS["bg2"]).pack(anchor="w")
        tk.Label(inn3, text="⚠ Для видимости эффекта используйте |q| > 100", font=("Consolas", 8), fg=COLORS["accent2"], bg=COLORS["bg2"]).pack(anchor="w")
        frm4, inn4 = section_frame(p, "Параметры расчёта")
        frm4.pack(fill="x", padx=8, pady=4)
        self._param_row(inn4, "t_max, с", self.var_t_max)
        self._param_row(inn4, "N_t (точек по времени)", self.var_N_t)
        self._param_row(inn4, "N (членов ряда)", self.var_N_terms)
        frm5, inn5 = section_frame(p, "Точка наблюдения A")
        frm5.pack(fill="x", padx=8, pady=4)
        self._param_row(inn5, "y_A, см", self.var_y_A)
        btn_frame = tk.Frame(p, bg=COLORS["bg"])
        btn_frame.pack(fill="x", padx=8, pady=10)
        styled_button(btn_frame, "▶ РАССЧИТАТЬ", self._run_all, color=COLORS["accent"]).pack(fill="x", pady=2)
        styled_button(btn_frame, "↺ Сброс", self._reset_params, color=COLORS["bg3"]).pack(fill="x", pady=2)
        styled_button(btn_frame, "💾 Экспорт", self._export_data, color=COLORS["bg3"]).pack(fill="x", pady=2)
        frm6, inn6 = section_frame(p, "Анализ результата")
        frm6.pack(fill="x", padx=8, pady=4)
        self.reco_text = tk.Text(inn6, height=8, width=36, font=("Consolas", 8), bg=COLORS["bg3"], fg=COLORS["text2"], relief="flat", state="disabled", wrap="word")
        self.reco_text.pack(fill="both")
        self._update_recommendations("—")

    def _param_row(self, parent, label, var):
        row = tk.Frame(parent, bg=COLORS["bg2"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=("Consolas", 8), fg=COLORS["text2"], bg=COLORS["bg2"], width=28, anchor="w").pack(side="left")
        styled_entry(row, var, width=10).pack(side="right")

    def _toggle_source_fields(self):
        if self.var_src_type.get() == "rect":
            self.rect_frame.pack(fill="x")
            self.gauss_frame.pack_forget()
        else:
            self.rect_frame.pack_forget()
            self.gauss_frame.pack(fill="x")
        if hasattr(self, "_rebind_scroll"):
            self.after(50, self._rebind_scroll)

    def _on_material_selected(self, event=None):
        mat_name = self.var_material.get()
        if mat_name in MATERIALS_DB:
            props = MATERIALS_DB[mat_name]
            info = f"{props['desc']}\n\nc_V = {props['c_V']}\na = {props['a']}\nλ = {props['lam']}\nT_пл = {props['T_melt']}°C\nρ = {props['density']}"
            self.mat_desc_label.config(text=info)

    def _apply_material_props(self):
        mat_name = self.var_material.get()
        if mat_name not in MATERIALS_DB:
            return
        props = MATERIALS_DB[mat_name]
        self.var_cV.set(props["c_V"])
        self.var_a.set(props["a"])
        self.var_lam.set(props["lam"])
        self.var_T_melt.set(props["T_melt"])
        self.var_T_harden.set(props["T_harden"])
        self.var_density.set(props["density"])
        messagebox.showinfo("Материал применён", f"Свойства '{mat_name}' загружены!")

    def _save_custom_material(self):
        dialog = tk.Toplevel(self)
        dialog.title("Сохранить материал")
        dialog.geometry("450x250")
        dialog.configure(bg=COLORS["bg"])
        dialog.transient(self)
        dialog.grab_set()
        tk.Label(dialog, text="Название:", fg=COLORS["text"], bg=COLORS["bg"], font=("Consolas", 10)).pack(pady=10)
        name_entry = tk.Entry(dialog, width=40, font=("Consolas", 10), bg=COLORS["bg3"], fg=COLORS["text"])
        name_entry.pack(pady=5)
        tk.Label(dialog, text="Описание:", fg=COLORS["text2"], bg=COLORS["bg"], font=("Consolas", 9)).pack(pady=(15,5))
        desc_entry = tk.Entry(dialog, width=40, font=("Consolas", 9), bg=COLORS["bg3"], fg=COLORS["text2"])
        desc_entry.pack(pady=5)
        def save():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Внимание", "Введите название")
                return
            built_in = ["Сталь 20", "Сталь 45", "Сталь 09Г2С", "Алюминий АМг5", "Титан ВТ6", "Медь М1"]
            if name in MATERIALS_DB and name not in built_in:
                if not messagebox.askyesno("Подтверждение", f"Перезаписать '{name}'?"):
                    return
            desc = desc_entry.get().strip() or "Пользовательский материал"
            MATERIALS_DB[name] = {"c_V": self.var_cV.get(), "a": self.var_a.get(), "lam": self.var_lam.get(), "T_melt": self.var_T_melt.get(), "T_harden": self.var_T_harden.get(), "density": self.var_density.get(), "desc": desc}
            self._save_custom_materials()
            self.var_material.set(name)
            self.material_combo["values"] = list(MATERIALS_DB.keys())
            messagebox.showinfo("Успех", f"Материал '{name}' сохранён!")
            dialog.destroy()
        btn_frame = tk.Frame(dialog, bg=COLORS["bg"])
        btn_frame.pack(pady=20)
        styled_button(btn_frame, "Сохранить", save, color=COLORS["accent"]).pack(side="left", padx=10)
        styled_button(btn_frame, "Отмена", dialog.destroy, color=COLORS["bg3"]).pack(side="left", padx=10)

    def _build_tabs(self, parent):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS["bg3"], foreground=COLORS["text2"], font=("Consolas", 9, "bold"), padding=[14, 6])
        style.map("TNotebook.Tab", background=[("selected", COLORS["bg2"])], foreground=[("selected", COLORS["accent"])])
        btn_frame = tk.Frame(parent, bg=COLORS["bg3"], height=28)
        btn_frame.pack(fill="x")
        btn_frame.pack_propagate(False)
        def scroll_left():
            try:
                idx = self.notebook.index(self.notebook.select())
                if idx > 0:
                    self.notebook.select(idx - 1)
            except Exception:
                pass
        def scroll_right():
            try:
                idx = self.notebook.index(self.notebook.select())
                if idx < self.notebook.index("end") - 1:
                    self.notebook.select(idx + 1)
            except Exception:
                pass
        styled_button(btn_frame, "◀", scroll_left, color=COLORS["bg3"]).pack(side="left", padx=2)
        styled_button(btn_frame, "▶", scroll_right, color=COLORS["bg3"]).pack(side="left", padx=2)
        tk.Label(btn_frame, text="  Навигация", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg3"]).pack(side="left", padx=10)
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)
        self.tab_cycle = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_cycle, text=" Цикл и охлаждение")
        self._build_tab_cycle(self.tab_cycle)
        self.tab_components = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_components, text=" Вклад компонент")
        self._build_tab_components(self.tab_components)
        self.tab_propagation = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_propagation, text="🔥 Распространение")
        self._build_tab_propagation(self.tab_propagation)
        self.tab_length = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_length, text=" Влияние длины")
        self._build_tab_length(self.tab_length)
        self.tab_phases = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_phases, text="🔷 Фазы")
        self._build_tab_phases(self.tab_phases)
        self.tab_metallurgy = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_metallurgy, text="️ Металлургия шва")
        self._build_tab_metallurgy(self.tab_metallurgy)
        self.tab_heatmap = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_heatmap, text="🌡 Карта и градиенты")
        self._build_tab_heatmap(self.tab_heatmap)
        self.tab_profile = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_profile, text="📐 Профиль T(y)")
        self._build_tab_profile(self.tab_profile)
        self.tab_3d = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_3d, text="🔷 3D")
        self._build_tab_3d(self.tab_3d)
        self.tab_source = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_source, text=" Источник и ряд")
        self._build_tab_source(self.tab_source)
        self.tab_theory = tk.Frame(self.notebook, bg=COLORS["bg2"])
        self.notebook.add(self.tab_theory, text="📖 Теория")
        self._build_tab_theory(self.tab_theory)

    def _build_tab_cycle(self, parent):
        top_frame = tk.Frame(parent, bg=COLORS["bg2"])
        top_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 4), facecolor=COLORS["bg2"])
        self.ax_cycle = fig.add_subplot(111)
        self.ax_cycle.set_facecolor(COLORS["bg2"])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(top_frame, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_cycle = fig
        self.canvas_cycle = canvas
        info = tk.Label(top_frame, text=" dT/dt < 0 — охлаждение. Критическая 30°C/с — закалка. 50°C/с — мартенсит.", font=("Consolas", 8), fg=COLORS["text2"], bg=COLORS["bg2"], justify="left")
        info.pack(anchor="w", padx=5, pady=2)
        bottom_frame = tk.Frame(parent, bg=COLORS["bg2"])
        bottom_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig2 = Figure(figsize=(10, 3.5), facecolor=COLORS["bg2"])
        self.ax_cooling = fig2.add_subplot(111)
        fig2.tight_layout(pad=2)
        canvas2 = FigureCanvasTkAgg(fig2, master=bottom_frame)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        toolbar2 = tk.Frame(bottom_frame, bg=COLORS["bg3"])
        toolbar2.pack(fill="x")
        NavigationToolbar2Tk(canvas2, toolbar2)
        self.fig_cooling = fig2
        self.canvas_cooling = canvas2

    def _build_tab_components(self, parent):
        fig = Figure(figsize=(10, 6), facecolor=COLORS["bg2"])
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_comp_total = fig.add_subplot(gs[0, :])
        self.ax_comp_A = fig.add_subplot(gs[1, 0])
        self.ax_comp_B = fig.add_subplot(gs[1, 1])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_components = fig
        self.canvas_components = canvas

    def _build_tab_propagation(self, parent):
        ctrl = tk.Frame(parent, bg=COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=6)
        tk.Label(ctrl, text="Время, с:", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg2"]).pack(side="left", padx=6)
        self.var_t_anim = tk.DoubleVar(value=1.0)
        styled_entry(ctrl, self.var_t_anim, width=8).pack(side="left")
        styled_button(ctrl, "Показать", self._update_propagation, color=COLORS["accent"]).pack(side="left", padx=8)
        self.anim_time_label = tk.Label(ctrl, text="", font=("Consolas", 9), fg=COLORS["accent3"], bg=COLORS["bg2"])
        self.anim_time_label.pack(side="left", padx=10)
        fig = Figure(figsize=(10, 5), facecolor=COLORS["bg2"])
        self.ax_propagation = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_propagation = fig
        self.canvas_propagation = canvas

    def _build_tab_length(self, parent):
        fig = Figure(figsize=(10, 5), facecolor=COLORS["bg2"])
        self.ax_length = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_length = fig
        self.canvas_length = canvas

    def _build_tab_phases(self, parent):
        fig = Figure(figsize=(10, 5), facecolor=COLORS["bg2"])
        self.ax_phases = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_phases = fig
        self.canvas_phases = canvas

    def _build_tab_metallurgy(self, parent):
        """Вкладка с CCT-диаграммой и металлургическим анализом"""
        # Верхняя часть: CCT-диаграмма
        top_frame = tk.Frame(parent, bg=COLORS["bg2"])
        top_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig_cct = Figure(figsize=(10, 5), facecolor=COLORS["bg2"])
        self.ax_cct = fig_cct.add_subplot(111)
        fig_cct.tight_layout(pad=2)
        canvas_cct = FigureCanvasTkAgg(fig_cct, master=top_frame)
        canvas_cct.get_tk_widget().pack(fill="both", expand=True)
        toolbar_cct = tk.Frame(top_frame, bg=COLORS["bg3"])
        toolbar_cct.pack(fill="x")
        NavigationToolbar2Tk(canvas_cct, toolbar_cct)
        self.fig_cct = fig_cct
        self.canvas_cct = canvas_cct
        
        # Нижняя часть: параметры шва
        bottom_frame = tk.Frame(parent, bg=COLORS["bg2"])
        bottom_frame.pack(fill="x", padx=8, pady=4)
        
        info_frame = tk.Frame(bottom_frame, bg=COLORS["bg3"])
        info_frame.pack(fill="x", pady=4)
        
        self.lbl_t85 = tk.Label(info_frame, text="t₈/₅: — с", font=("Consolas", 10), fg=COLORS["accent"], bg=COLORS["bg3"])
        self.lbl_t85.pack(side="left", padx=10, pady=5)
        
        self.lbl_weld_width = tk.Label(info_frame, text="Ширина шва: — см", font=("Consolas", 10), fg=COLORS["accent2"], bg=COLORS["bg3"])
        self.lbl_weld_width.pack(side="left", padx=10, pady=5)
        
        self.lbl_haz_width = tk.Label(info_frame, text="Ширина ЗТВ: — см", font=("Consolas", 10), fg=COLORS["accent3"], bg=COLORS["bg3"])
        self.lbl_haz_width.pack(side="left", padx=10, pady=5)
        
        self.lbl_structure = tk.Label(info_frame, text="Структура: —", font=("Consolas", 10, "bold"), fg=COLORS["text"], bg=COLORS["bg3"])
        self.lbl_structure.pack(side="left", padx=10, pady=5)

    def _build_tab_heatmap(self, parent):
        top_frame = tk.Frame(parent, bg=COLORS["bg2"])
        top_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 4.5), facecolor=COLORS["bg2"])
        self.ax_heatmap = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(top_frame, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_heatmap = fig
        self.canvas_heatmap = canvas
        bottom_frame = tk.Frame(parent, bg=COLORS["bg2"])
        bottom_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig2 = Figure(figsize=(10, 3.5), facecolor=COLORS["bg2"])
        self.ax_gradient = fig2.add_subplot(111)
        fig2.tight_layout(pad=2)
        canvas2 = FigureCanvasTkAgg(fig2, master=bottom_frame)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        toolbar2 = tk.Frame(bottom_frame, bg=COLORS["bg3"])
        toolbar2.pack(fill="x")
        NavigationToolbar2Tk(canvas2, toolbar2)
        self.fig_gradient = fig2
        self.canvas_gradient = canvas2

    def _build_tab_profile(self, parent):
        ctrl = tk.Frame(parent, bg=COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=6)
        tk.Label(ctrl, text="Время, с:", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg2"]).pack(side="left", padx=6)
        self.var_t_profile = tk.DoubleVar(value=5.0)
        styled_entry(ctrl, self.var_t_profile, width=8).pack(side="left")
        styled_button(ctrl, "Обновить", self._update_profile, color=COLORS["accent"]).pack(side="left", padx=8)
        fig = Figure(figsize=(10, 5.5), facecolor=COLORS["bg2"])
        self.ax_profile = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_profile = fig
        self.canvas_profile = canvas

    def _build_tab_3d(self, parent):
        fig = Figure(figsize=(10, 6), facecolor=COLORS["bg2"])
        self.ax_3d = fig.add_subplot(111, projection="3d")
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_3d = fig
        self.canvas_3d = canvas

    def _build_tab_source(self, parent):
        top_frame = tk.Frame(parent, bg=COLORS["bg2"])
        top_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 3.5), facecolor=COLORS["bg2"])
        gs = fig.add_gridspec(1, 2, wspace=0.4)
        self.ax_src_rect = fig.add_subplot(gs[0])
        self.ax_src_gauss = fig.add_subplot(gs[1])
        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(top_frame, bg=COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_source = fig
        self.canvas_source = canvas
        info = tk.Label(top_frame, text=" Прямоугольный источник даёт T_max на 20-30% выше гауссова при том же q_max.", font=("Consolas", 8), fg=COLORS["warning"], bg=COLORS["bg2"], justify="left")
        info.pack(anchor="w", padx=5, pady=2)
        bottom_frame = tk.Frame(parent, bg=COLORS["bg2"])
        bottom_frame.pack(fill="both", expand=True, padx=4, pady=2)
        ctrl = tk.Frame(bottom_frame, bg=COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=2)
        tk.Label(ctrl, text="N гармоник:", font=("Consolas", 9), fg=COLORS["text2"], bg=COLORS["bg2"]).pack(side="left", padx=6)
        self.var_n_show = tk.IntVar(value=8)
        styled_entry(ctrl, self.var_n_show, width=5).pack(side="left")
        styled_button(ctrl, "Обновить", self._update_fourier_tab, color=COLORS["accent"]).pack(side="left", padx=8)
        fig2 = Figure(figsize=(10, 4), facecolor=COLORS["bg2"])
        gs2 = fig2.add_gridspec(2, 1, hspace=0.5)
        self.ax_fourier_modes = fig2.add_subplot(gs2[0])
        self.ax_fourier_conv = fig2.add_subplot(gs2[1])
        fig2.tight_layout(pad=2.5)
        canvas2 = FigureCanvasTkAgg(fig2, master=bottom_frame)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        self.fig_fourier = fig2
        self.canvas_fourier = canvas2

    def _build_tab_theory(self, parent):
        text_frame = tk.Frame(parent, bg=COLORS["bg2"])
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), bg=COLORS["bg3"], fg=COLORS["text"], relief="flat", padx=20, pady=20)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text_widget.pack(fill="both", expand=True)
        theory = """
ТЕОРЕТИЧЕСКАЯ БАЗА МОДЕЛИ
══════════════════════════════════════════════════════════════

УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ:
∂T/∂t = a · ∂²T/∂y² + q(y,t)/c_V

РЕШЕНИЕ МЕТОДОМ РЯДОВ ФУРЬЕ:
T(y,t) = (2/l) · Σ β_n · [ А + Б + В ] · cos(μ_n · y)

А = φ_n · exp(-a·μ²_n·t) — затухание начального условия
Б = -(a/λ) · ∫[q₁(τ) + q₂(τ)·(-1)ⁿ]·exp(-a·μ²_n·(t-τ))dτ — граничные потоки
В = (1/c_V) · ∫ q(η,τ)·exp(-a·μ²_n·(t-τ))·cos(μ_n·η) dη dτ — источник

ПАРАМЕТР t/₅:
Время охлаждения от 800°C до 500°C — ключевой параметр для сварки.
Определяет структуру металла шва и ЗТВ.

Критические значения t₈/₅ для углеродистых сталей:
• t₈/₅ < 2 с → Мартенсит (хрупкий)
• 2 < t₈/₅ < 10 с → Мартенсит + Бейнит
• 10 < t₈/₅ < 30 с → Бейнит
• 30 < t₈/₅ < 100 с → Перлит + Феррит
• t₈/₅ > 100 с → Грубый феррит

ШИРИНА ШВА И ЗТВ:
• Ширина шва: определяется по изотерме T_пл (зона плавления)
• Ширина ЗТВ: определяется по изотерме T_зак = AC₁ (зона термического влияния)

CCT-ДИАГРАММА:
Диаграмма непрерывного охлаждения показывает фазовые превращения
в зависимости от скорости охлаждения. Наложение термического цикла
на CCT позволяет предсказать структуру шва.
"""
        text_widget.insert("1.0", theory)
        text_widget.configure(state="disabled")

    def _get_params(self):
        l = self.var_l.get()
        a = self.var_a.get()
        c_V = self.var_cV.get()
        lam = self.var_lam.get()
        q_max = self.var_q_max.get()
        t_src = self.var_t_source.get()
        t_max = self.var_t_max.get()
        N_t = max(40, int(self.var_N_t.get()))
        y_A = self.var_y_A.get()
        N = max(10, int(self.var_N_terms.get()))
        T_melt = self.var_T_melt.get()
        T_harden = self.var_T_harden.get()
        T_init = self.var_T_init.get()
        q1_val = self.var_q1.get()
        q2_val = self.var_q2.get()
        src_type = self.var_src_type.get()
        if src_type == "rect":
            y1 = self.var_y1.get()
            y2 = self.var_y2.get()
            q_func = lambda eta: source_rect(eta, q_max, y1, y2)
        else:
            k_g = self.var_k_gauss.get()
            q_func = lambda eta: source_gauss(eta, q_max, k_g, l)
        phi_func = (lambda y: np.full_like(y, T_init)) if T_init != 0 else None
        q1_func = (lambda tau: np.full_like(tau, q1_val)) if q1_val != 0 else None
        q2_func = (lambda tau: np.full_like(tau, q2_val)) if q2_val != 0 else None
        return dict(l=l, a=a, c_V=c_V, lam=lam, q_max=q_max, q_func=q_func, t_source_end=t_src, phi_func=phi_func, q1_func=q1_func, q2_func=q2_func, t_max=t_max, N_t=N_t, y_A=y_A, N_terms=N, T_melt=T_melt, T_harden=T_harden, T_init=T_init, q1_val=q1_val, q2_val=q2_val, src_type=src_type)

    def _build_solver_kwargs(self, p):
        return dict(l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"], q_func=p["q_func"], t_source_end=p["t_source_end"], phi_func=p["phi_func"], q1_func=p["q1_func"], q2_func=p["q2_func"], N_terms=p["N_terms"])

    def _run_all(self):
        try:
            p = self._get_params()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.status_var.set(" Вычисление...")
        self.update()
        l = p["l"]
        t_max = p["t_max"]
        N_t = p["N_t"]
        y_A = p["y_A"]
        N_y = 100
        y_arr = np.linspace(0, l, N_y)
        t_arr = np.linspace(0.05, t_max, N_t)
        kw = self._build_solver_kwargs(p)
        try:
            T_field = compute_spatial_field(y_arr, t_arr, **kw)
            T_field += p["T_init"]
            i_yA = int(np.clip(np.round(y_A / l * (N_y - 1)), 0, N_y - 1))
            T_cycle = T_field[:, i_yA]
        except Exception as e:
            messagebox.showerror("Ошибка расчёта", str(e))
            self.status_var.set("❌ Ошибка")
            return
        self._y_arr = y_arr
        self._t_arr = t_arr
        self._T_field = T_field
        self._T_cycle = T_cycle
        self._params = p
        self.status_var.set("✅ Строю графики...")
        self.update()
        self._plot_cycle(t_arr, T_cycle, p)
        self._plot_cooling_rate(t_arr, T_cycle, p)
        self._plot_components(y_arr, t_arr, p)
        self._update_propagation()
        self._plot_length_effect(p)
        self._plot_phases(t_arr, T_cycle, p)
        self._plot_metallurgy(t_arr, T_cycle, y_arr, T_field, p)
        self._plot_heatmap(y_arr, t_arr, T_field, p)
        self._plot_gradient(y_arr, t_arr, T_field, p)
        self._plot_3d(y_arr, t_arr, T_field)
        self._plot_source(p)
        self._update_profile()
        self._update_fourier_tab()
        self._update_recommendations(self._analyze(t_arr, T_cycle, p))
        self.status_var.set(f"✅ Готово | T_max={T_field.max():.0f}°C")

    def _plot_cycle(self, t_arr, T_cycle, p):
        ax = self.ax_cycle
        ax.clear()
        apply_mpl_style(ax, f"Термический цикл y_A={p['y_A']} см", "Время t, с", "Температура T, °C")
        ax.plot(t_arr, T_cycle, color=COLORS["accent"], lw=2, label="T(y_A, t)")
        T_melt = p["T_melt"]
        T_harden = p["T_harden"]
        if T_cycle.max() > T_melt:
            ax.axhline(T_melt, color=COLORS["melt"], lw=1.2, ls="--", label=f"T_пл={T_melt:.0f}°C")
            ax.fill_between(t_arr, T_melt, T_cycle, where=T_cycle >= T_melt, alpha=0.18, color=COLORS["melt"], label="Плавление")
        if T_harden > 0 and T_cycle.max() > T_harden:
            ax.axhline(T_harden, color=COLORS["warning"], lw=1.2, ls="--", label=f"T_зак={T_harden:.0f}°C")
        ax.axvline(p["t_source_end"], color=COLORS["accent3"], lw=1, ls=":", label=f"Конец источника")
        idx_max = np.argmax(T_cycle)
        ax.scatter(t_arr[idx_max], T_cycle[idx_max], color=COLORS["accent2"], zorder=5, s=60)
        ax.annotate(f"  T_max={T_cycle[idx_max]:.0f}°C", xy=(t_arr[idx_max], T_cycle[idx_max]), color=COLORS["accent2"], fontsize=8, xytext=(6, -20), textcoords="offset points")
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_cycle.draw()

    def _plot_cooling_rate(self, t_arr, T_cycle, p):
        ax = self.ax_cooling
        ax.clear()
        dTdt = np.gradient(T_cycle, t_arr)
        apply_mpl_style(ax, "Скорость охлаждения dT/dt", "Время t, с", "dT/dt, °C/с")
        ax.fill_between(t_arr, 0, dTdt, alpha=0.3, color=COLORS["accent4"])
        ax.plot(t_arr, dTdt, color=COLORS["accent4"], lw=2, label="dT/dt")
        ax.axhline(-30, color=COLORS["warning"], lw=1, ls="--", label="Критическая (30°C/с)")
        ax.axhline(-50, color=COLORS["melt"], lw=1, ls="--", label="Закалка (50°C/с)")
        ax.axhline(0, color=COLORS["text2"], lw=0.5, ls="-")
        idx_cool = np.argmin(dTdt)
        ax.scatter(t_arr[idx_cool], dTdt[idx_cool], color=COLORS["accent2"], zorder=5, s=60)
        ax.annotate(f"  max={dTdt[idx_cool]:.1f}°C/с", xy=(t_arr[idx_cool], dTdt[idx_cool]), color=COLORS["accent2"], fontsize=8)
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_cooling.draw()

    def _plot_components(self, y_arr, t_arr, p):
        y_A = p["y_A"]
        y_pt = np.array([y_A])
        T_A_arr, T_B_arr, T_C_arr, T_total_arr = [], [], [], []
        kw = self._build_solver_kwargs(p)
        for t in t_arr:
            result = compute_temperature_detailed(y_pt, t, return_components=True, **kw)
            T_A_arr.append(result["A"][0])
            T_B_arr.append(result["B"][0])
            T_C_arr.append(result["C"][0])
            T_total_arr.append(result["total"][0])
        T_A_arr = np.array(T_A_arr)
        T_B_arr = np.array(T_B_arr)
        T_C_arr = np.array(T_C_arr)
        T_total_arr = np.array(T_total_arr)
        ax1 = self.ax_comp_total
        ax1.clear()
        apply_mpl_style(ax1, "Вклад компонент в точке A", "Время t, с", "Температура, °C")
        ax1.plot(t_arr, T_total_arr, color=COLORS["text"], lw=2, label="Всего T")
        ax1.plot(t_arr, T_A_arr, color=COLORS["accent"], lw=1.5, ls="--", label="A: начальное")
        ax1.plot(t_arr, T_B_arr, color=COLORS["accent2"], lw=1.5, ls="--", label="B: потоки")
        ax1.plot(t_arr, T_C_arr, color=COLORS["accent3"], lw=1.5, ls="--", label="C: источник")
        ax1.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax1.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        ax2 = self.ax_comp_A
        ax2.clear()
        apply_mpl_style(ax2, "A: Начальное условие", "Время t, с", "T_A, °C")
        ax2.plot(t_arr, T_A_arr, color=COLORS["accent"], lw=2)
        ax2.fill_between(t_arr, 0, T_A_arr, alpha=0.3, color=COLORS["accent"])
        ax2.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        ax3 = self.ax_comp_B
        ax3.clear()
        apply_mpl_style(ax3, "B: Граничные потоки", "Время t, с", "T_B, °C")
        ax3.plot(t_arr, T_B_arr, color=COLORS["accent2"], lw=2)
        ax3.fill_between(t_arr, 0, T_B_arr, alpha=0.3, color=COLORS["accent2"])
        ax3.axhline(0, color=COLORS["text2"], lw=0.5, ls="-")
        ax3.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_components.draw()

    def _update_propagation(self, *_):
        if self._T_field is None:
            return
        ax = self.ax_propagation
        ax.clear()
        p = self._params
        try:
            t_show = float(self.var_t_anim.get())
        except Exception:
            t_show = self._t_arr[0]
        idx_t = int(np.argmin(np.abs(self._t_arr - t_show)))
        t_actual = self._t_arr[idx_t]
        T_profile = self._T_field[idx_t, :]
        apply_mpl_style(ax, f"Распространение тепла при t={t_actual:.2f} с", "Координата y, см", "Температура T, °C")
        ax.plot(self._y_arr, T_profile, color=COLORS["accent"], lw=3, label="T(y)")
        ax.fill_between(self._y_arr, 0, T_profile, alpha=0.2, color=COLORS["accent"])
        if p["src_type"] == "rect":
            ax.axvspan(self.var_y1.get(), self.var_y2.get(), alpha=0.15, color=COLORS["accent2"], label="Зона источника")
        else:
            ax.axvline(p["l"]/2, color=COLORS["accent2"], lw=2, ls=":", label="Центр Гаусса")
        ax.axhline(p["T_melt"], color=COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax.axhline(p["T_harden"], color=COLORS["warning"], lw=1.5, ls="--", label=f"T_зак={p['T_harden']:.0f}°C")
        ax.axvline(p["y_A"], color=COLORS["accent3"], lw=1.5, ls=":", label=f"y_A={p['y_A']} см")
        idx_max = np.argmax(T_profile)
        ax.annotate(f"  T_max={T_profile[idx_max]:.0f}°C\n  y={self._y_arr[idx_max]:.1f} см", xy=(self._y_arr[idx_max], T_profile[idx_max]), color=COLORS["accent2"], fontsize=9, fontweight="bold", xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["bg3"], edgecolor=COLORS["accent2"]))
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.anim_time_label.config(text=f"t={t_actual:.2f} с | T_max={T_profile.max():.0f}°C")
        self.canvas_propagation.draw()

    def _plot_length_effect(self, p):
        ax = self.ax_length
        ax.clear()
        lengths = np.linspace(5, 30, 10)
        T_max_values = []
        kw_base = self._build_solver_kwargs(p)
        for l_test in lengths:
            if p["src_type"] == "rect":
                y1 = self.var_y1.get() * l_test / 10.0
                y2 = self.var_y2.get() * l_test / 10.0
                q_func = lambda eta: source_rect(eta, p["q_max"], y1, y2)
            else:
                k_g = self.var_k_gauss.get()
                q_func = lambda eta: source_gauss(eta, p["q_max"], k_g, l_test)
            kw = dict(kw_base, l=l_test, q_func=q_func)
            y_test = np.linspace(0, l_test, 50)
            t_test = np.array([p["t_source_end"] * 0.5])
            T_field = compute_spatial_field(y_test, t_test, **kw)
            T_max_values.append(T_field.max() + p["T_init"])
        apply_mpl_style(ax, "Влияние длины пластины на T_max", "Длина l, см", "T_max, °C")
        ax.plot(lengths, T_max_values, color=COLORS["accent"], lw=2, marker="o", markersize=6)
        ax.fill_between(lengths, 0, T_max_values, alpha=0.2, color=COLORS["accent"])
        ax.axvline(p["l"], color=COLORS["accent2"], lw=1.5, ls="--", label=f"Текущая l={p['l']} см")
        ax.axhline(p["T_melt"], color=COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_length.draw()

    def _plot_phases(self, t_arr, T_cycle, p):
        ax = self.ax_phases
        ax.clear()
        phases = []
        phase_colors = []
        for T_val in T_cycle:
            if T_val >= p["T_melt"]:
                phases.append("Жидкость")
                phase_colors.append(COLORS["melt"])
            elif p["T_harden"] > 0 and T_val >= p["T_harden"]:
                phases.append("Аустенит")
                phase_colors.append(COLORS["warning"])
            else:
                phases.append("Феррит+Перлит")
                phase_colors.append(COLORS["accent"])
        apply_mpl_style(ax, "Фазовые превращения", "Время t, с", "Температура T, °C")
        for i in range(len(t_arr)-1):
            ax.axvspan(t_arr[i], t_arr[i+1], ymin=0, ymax=1, alpha=0.15, color=phase_colors[i])
        ax.plot(t_arr, T_cycle, color=COLORS["text"], lw=2, label="T(t)")
        ax.axhline(p["T_melt"], color=COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax.axhline(p["T_harden"], color=COLORS["warning"], lw=1.5, ls="--", label=f"T_зак={p['T_harden']:.0f}°C")
        legend_elements = [Patch(facecolor=COLORS["melt"], alpha=0.3, label='Жидкость'), Patch(facecolor=COLORS["warning"], alpha=0.3, label='Аустенит'), Patch(facecolor=COLORS["accent"], alpha=0.3, label='Феррит+Перлит')]
        ax.legend(handles=legend_elements, fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"], loc='upper right')
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_phases.draw()

    def _plot_metallurgy(self, t_arr, T_cycle, y_arr, T_field, p):
        """Построение CCT-диаграммы и отображение металлургических параметров"""
        ax = self.ax_cct
        ax.clear()
        
        # Расчёт t₈/₅
        t85 = compute_t85(t_arr, T_cycle)
        
        # Расчёт ширины шва и ЗТВ
        weld_width = compute_weld_width(y_arr, T_field, p["T_melt"])
        haz_width = compute_haz_width(y_arr, T_field, p["T_harden"])
        
        # Предсказание структуры
        structure, struct_color = predict_structure(t85, T_cycle.max(), p["T_melt"])
        
        # Обновление меток
        if t85 is not None:
            self.lbl_t85.config(text=f"t₈/₅: {t85:.2f} с")
        else:
            self.lbl_t85.config(text="t₈/: —")
        
        self.lbl_weld_width.config(text=f"Ширина шва: {weld_width:.2f} см")
        self.lbl_haz_width.config(text=f"Ширина ЗТВ: {haz_width:.2f} см")
        self.lbl_structure.config(text=f"Структура: {structure}", fg=struct_color)
        
        # Построение CCT-диаграммы (упрощённая для углеродистых сталей)
        apply_mpl_style(ax, "CCT-диаграмма с термическим циклом", "Время, с (лог)", "Температура, °C")
        
        # Ось X в логарифмической шкале
        ax.set_xscale("log")
        
        # Зоны фазовых превращений (упрощённые данные для стали)
        # Ферритная зона
        ax.fill_between([0.1, 100], [700, 500], alpha=0.2, color="lightblue", label="Феррит")
        # Перлитная зона
        ax.fill_between([10, 1000], [600, 400], alpha=0.2, color="lightgreen", label="Перлит")
        # Бейнитная зона
        ax.fill_between([100, 10000], [400, 250], alpha=0.2, color="yellow", label="Бейнит")
        # Мартенситная зона (ниже M_s)
        ax.axhline(200, color="red", lw=1, ls="--", alpha=0.5)
        ax.text(0.5, 210, "M_s ≈ 200°C", fontsize=8, color="red")
        
        # Наложение термического цикла
        idx_max = np.argmax(T_cycle)
        t_cool = t_arr[idx_max:] - t_arr[idx_max] + 1  # Смещение для логарифмической шкалы
        T_cool = T_cycle[idx_max:]
        
        # Фильтрация только положительных времён
        valid = t_cool > 0
        if np.any(valid):
            ax.plot(t_cool[valid], T_cool[valid], color=COLORS["accent"], lw=2, label="Термический цикл")
            
            # Отметка t₈/
            if t85 is not None:
                t_800 = t_cool[valid][0] + (800 - T_cool[valid][0]) * (t85) / (T_cool[valid][0] - 500) if T_cool[valid][0] > 800 else t_cool[valid][0]
                ax.axvline(t_800 + t85, color=COLORS["accent2"], lw=1.5, ls="--", alpha=0.7, label=f"t₈/₅={t85:.2f}с")
        
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        ax.set_xlim(0.1, 10000)
        ax.set_ylim(0, 1600)
        
        self.canvas_cct.draw()

    def _plot_heatmap(self, y_arr, t_arr, T_field, p):
        ax = self.ax_heatmap
        ax.clear()
        if hasattr(self, 'cbar_heatmap') and self.cbar_heatmap is not None:
            try:
                self.cbar_heatmap.remove()
            except Exception:
                pass
            self.cbar_heatmap = None
        T_plot = np.clip(T_field, 0, None).T
        extent = [t_arr[0], t_arr[-1], y_arr[0], y_arr[-1]]
        im = ax.imshow(T_plot, aspect="auto", origin="lower", extent=extent, cmap=CMAP_HEAT)
        T_max_actual = np.max(T_field)
        T_min_actual = np.min(T_field)
        vmin = max(T_min_actual, p["T_init"])
        vmax = min(T_max_actual * 1.1, p["T_melt"] * 1.5)
        im.set_clim(vmin, vmax)
        self.cbar_heatmap = self.fig_heatmap.colorbar(im, ax=ax, pad=0.02)
        self.cbar_heatmap.set_label("T, °C", color=COLORS["text2"], fontsize=9)
        self.cbar_heatmap.ax.yaxis.set_tick_params(color=COLORS["text2"])
        plt.setp(self.cbar_heatmap.ax.yaxis.get_ticklabels(), color=COLORS["text2"])
        ax.axhline(p["y_A"], color=COLORS["accent3"], lw=1.5, ls="--", label=f"y_A={p['y_A']} см")
        try:
            ax.contour(t_arr, y_arr, T_plot, levels=[p["T_melt"]], colors=[COLORS["melt"]], linewidths=1.5, linestyles="--")
        except Exception:
            pass
        apply_mpl_style(ax, "T(y, t) — тепловая карта", "Время t, с", "Координата y, см")
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        self.canvas_heatmap.draw()

    def _plot_gradient(self, y_arr, t_arr, T_field, p):
        ax = self.ax_gradient
        ax.clear()
        if hasattr(self, 'cbar_gradient') and self.cbar_gradient is not None:
            try:
                self.cbar_gradient.remove()
            except Exception:
                pass
            self.cbar_gradient = None
        grad_T = np.gradient(T_field, axis=1)
        T_plot = np.abs(grad_T.T)
        extent = [t_arr[0], t_arr[-1], y_arr[0], y_arr[-1]]
        im = ax.imshow(T_plot, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        grad_max = np.max(T_plot)
        im.set_clim(0, grad_max * 1.1)
        self.cbar_gradient = self.fig_gradient.colorbar(im, ax=ax, pad=0.02)
        self.cbar_gradient.set_label("|∇T|, °C/см", color=COLORS["text2"], fontsize=9)
        self.cbar_gradient.ax.yaxis.set_tick_params(color=COLORS["text2"])
        plt.setp(self.cbar_gradient.ax.yaxis.get_ticklabels(), color=COLORS["text2"])
        ax.axhline(p["y_A"], color=COLORS["accent3"], lw=1.5, ls="--", label=f"y_A={p['y_A']} см")
        apply_mpl_style(ax, "|∇T(y, t)| — градиенты температур", "Время t, с", "Координата y, см")
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        self.canvas_gradient.draw()

    def _update_profile(self, *_):
        if self._T_field is None:
            return
        ax = self.ax_profile
        ax.clear()
        p = self._params
        try:
            t_show = float(self.var_t_profile.get())
        except Exception:
            t_show = self._t_arr[len(self._t_arr)//2]
        idx_t = int(np.argmin(np.abs(self._t_arr - t_show)))
        t_actual = self._t_arr[idx_t]
        T_profile = self._T_field[idx_t, :]
        apply_mpl_style(ax, f"Профиль T(y) при t={t_actual:.2f} с", "Координата y, см", "Температура T, °C")
        ax.plot(self._y_arr, T_profile, color=COLORS["accent"], lw=2)
        ax.fill_between(self._y_arr, 0, T_profile, alpha=0.12, color=COLORS["accent"])
        ax.axhline(p["T_melt"], color=COLORS["melt"], lw=1, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax.axhline(p["T_harden"], color=COLORS["warning"], lw=1, ls="--", label=f"T_зак={p['T_harden']:.0f}°C")
        ax.axvline(p["y_A"], color=COLORS["accent3"], lw=1, ls=":", label=f"y_A={p['y_A']} см")
        src = p["src_type"]
        if src == "rect":
            ax.axvspan(self.var_y1.get(), self.var_y2.get(), alpha=0.12, color=COLORS["accent2"], label="Зона источника")
        else:
            ax.axvline(p["l"]/2, color=COLORS["accent2"], lw=1, ls=":", label="Центр Гаусса")
        ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        ax.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_profile.draw()

    def _plot_3d(self, y_arr, t_arr, T_field):
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection="3d")
        ax.set_facecolor(COLORS["bg2"])
        self.fig_3d.patch.set_facecolor(COLORS["bg2"])
        step_y = max(1, len(y_arr)//40)
        step_t = max(1, len(t_arr)//40)
        Y = y_arr[::step_y]
        T = self._t_arr[::step_t]
        Z = T_field[::step_t, ::step_y]
        TT, YY = np.meshgrid(T, Y)
        Z_plot = np.clip(Z.T, 0, None)
        surf = ax.plot_surface(TT, YY, Z_plot, cmap=CMAP_HEAT, linewidth=0, antialiased=True, alpha=0.88)
        self.fig_3d.colorbar(surf, ax=ax, shrink=0.5, label="T, °C", pad=0.1)
        ax.set_xlabel("t, с", color=COLORS["text2"], fontsize=8, labelpad=6)
        ax.set_ylabel("y, см", color=COLORS["text2"], fontsize=8, labelpad=6)
        ax.set_zlabel("T, °C", color=COLORS["text2"], fontsize=8, labelpad=6)
        ax.set_title("3D: T(y, t)", color=COLORS["text"], fontsize=10, fontweight="bold")
        ax.tick_params(colors=COLORS["text2"], labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        self.canvas_3d.draw()

    def _plot_source(self, p):
        eta = np.linspace(0, p["l"], 600)
        ax1 = self.ax_src_rect
        ax1.clear()
        q_r = source_rect(eta, p["q_max"], self.var_y1.get(), self.var_y2.get())
        ax1.fill_between(eta, 0, q_r, alpha=0.4, color=COLORS["accent2"])
        ax1.plot(eta, q_r, color=COLORS["accent2"], lw=2)
        apply_mpl_style(ax1, "Прямоугольный источник", "y, см", "q(y), Вт/см³")
        ax1.set_ylim(bottom=-p["q_max"]*0.05)
        ax1.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        ax2 = self.ax_src_gauss
        ax2.clear()
        q_g = source_gauss(eta, p["q_max"], self.var_k_gauss.get(), p["l"])
        ax2.fill_between(eta, 0, q_g, alpha=0.4, color=COLORS["accent3"])
        ax2.plot(eta, q_g, color=COLORS["accent3"], lw=2)
        apply_mpl_style(ax2, f"Гауссов источник (k={self.var_k_gauss.get()})", "y, см", "q(y), Вт/см³")
        ax2.set_ylim(bottom=-p["q_max"]*0.05)
        ax2.grid(True, color=COLORS["border"], lw=0.5, alpha=0.6)
        for ax in [ax1, ax2]:
            ax.axhline(0.05 * p["q_max"], color=COLORS["text2"], lw=0.8, ls=":", label="5% q_max")
            ax.legend(fontsize=8, facecolor=COLORS["bg3"], labelcolor=COLORS["text"], edgecolor=COLORS["border"])
        self.canvas_source.draw()

    def _update_fourier_tab(self, *_):
        if self._T_field is None:
            return
        p = self._params
        N_show = min(max(1, self.var_n_show.get()), 20)
        l = p["l"]
        c_V = p["c_V"]
        eta = np.linspace(0, l, 600)
        q_eta = p["q_func"](eta)
        n_arr = np.arange(0, N_show + 1)
        I_n = []
        for n in n_arr:
            mn = np.pi * n / l
            cos_eta = np.cos(mn * eta)
            I_n.append(np.trapezoid(q_eta * cos_eta, eta))
        I_n = np.array(I_n)
        bn_arr = np.array([beta_n(n) for n in n_arr])
        coefs = (2/l) * bn_arr * I_n / c_V
        ax1 = self.ax_fourier_modes
        ax1.clear()
        colors_bar = [COLORS["accent"] if c >= 0 else COLORS["accent4"] for c in coefs]
        ax1.bar(n_arr, coefs, color=colors_bar, alpha=0.85)
        apply_mpl_style(ax1, "Коэффициенты Фурье", "Номер гармоники n", "Амплитуда, °C/с")
        ax1.grid(True, color=COLORS["border"], lw=0.5, alpha=0.5, axis="y")
        ax2 = self.ax_fourier_conv
        ax2.clear()
        y_arr = np.linspace(0, l, 200)
        t_test = min(p["t_source_end"], p["t_max"] * 0.5)
        kw = self._build_solver_kwargs(p)
        errors = []
        n_list = list(range(0, min(N_show*4, p["N_terms"]+1), 2))
        for N_cur in n_list:
            kw2 = dict(kw, N_terms=N_cur)
            T_cur = compute_temperature_detailed(y_arr, t_test, **kw2) + p["T_init"]
            T_full = self._T_field[int(np.argmin(np.abs(self._t_arr - t_test))), :]
            err = np.max(np.abs(T_cur - T_full)) if len(T_cur)==len(T_full) else 0
            errors.append(err)
        ax2.semilogy(n_list, [max(e, 0.001) for e in errors], color=COLORS["accent3"], lw=2, marker="o", markersize=4)
        apply_mpl_style(ax2, f"Сходимость ряда (t={t_test:.1f}с)", "Число членов ряда N", "Макс. погрешность, °C")
        ax2.grid(True, color=COLORS["border"], lw=0.5, alpha=0.5)
        self.canvas_fourier.draw()

    def _analyze(self, t_arr, T_cycle, p):
        T_melt = p["T_melt"]
        T_harden = p["T_harden"]
        T_max = T_cycle.max()
        t_at_max = t_arr[np.argmax(T_cycle)]
        lines = []
        lines.append(f"T_max = {T_max:.0f}°C")
        lines.append(f"t(T_max) = {t_at_max:.2f}с")
        above_melt = T_cycle >= T_melt
        if above_melt.any():
            dt = t_arr[1] - t_arr[0]
            t_melt = np.sum(above_melt) * dt
            lines.append(f"t > T_пл = {t_melt:.2f}с")
            lines.append(f"✓ ПЛАВЛЕНИЕ: да")
        else:
            lines.append(f"✗ ПЛАВЛЕНИЕ: нет")
        if T_harden > 0:
            above_hard = T_cycle >= T_harden
            if above_hard.any():
                dt = t_arr[1] - t_arr[0]
                t_hard = np.sum(above_hard) * dt
                lines.append(f"t > T_зак = {t_hard:.2f}с")
                above = np.where(above_hard)[0]
                if len(above) > 2:
                    i_cool = above[-2]
                    if i_cool + 1 < len(T_cycle):
                        dt_local = t_arr[i_cool+1] - t_arr[i_cool]
                        dT = T_cycle[i_cool+1] - T_cycle[i_cool]
                        v_cool = dT / dt_local
                        lines.append(f"v_охл(T_зак) = {v_cool:.1f}°C/с")
                        if v_cool < -50:
                            lines.append("⚠ Возможна закалка")
                        elif v_cool < -10:
                            lines.append("✓ Нормализация")
                        else:
                            lines.append("✓ Медленное охлаждение")
        return "\n".join(lines)

    def _update_recommendations(self, text):
        self.reco_text.configure(state="normal")
        self.reco_text.delete("1.0", "end")
        self.reco_text.insert("1.0", text)
        self.reco_text.configure(state="disabled")

    def _reset_params(self):
        self.var_l.set(10.0)
        self.var_a.set(0.1)
        self.var_cV.set(4.2)
        self.var_lam.set(0.5)
        self.var_density.set(7.85)
        self.var_T_melt.set(1510.0)
        self.var_T_harden.set(723.0)
        self.var_T_init.set(20.0)
        self.var_q_max.set(1e4)
        self.var_y1.set(4.5)
        self.var_y2.set(5.5)
        self.var_k_gauss.set(5.0)
        self.var_t_source.set(5.0)
        self.var_q1.set(0.0)
        self.var_q2.set(0.0)
        self.var_t_max.set(60.0)
        self.var_N_t.set(120)
        self.var_y_A.set(4.5)
        self.var_N_terms.set(80)
        self.var_src_type.set("rect")
        self.var_material.set("Сталь 20")
        self._toggle_source_fields()
        self._on_material_selected()
        messagebox.showinfo("Сброс", "Параметры сброшены")

    def _export_data(self):
        if self._T_field is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчёт")
            return
        try:
            fname = "welding_results.csv"
            with open(fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["t\\y"] + [f"{y:.3f}" for y in self._y_arr]
                writer.writerow(header)
                for i, t in enumerate(self._t_arr):
                    row = [f"{t:.4f}"] + [f"{v:.4f}" for v in self._T_field[i, :]]
                    writer.writerow(row)
            fname2 = "thermal_cycle.csv"
            with open(fname2, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t, с", "T, °C"])
                for t, T in zip(self._t_arr, self._T_cycle):
                    writer.writerow([f"{t:.4f}", f"{T:.4f}"])
            dTdt = np.gradient(self._T_cycle, self._t_arr)
            fname3 = "cooling_rate.csv"
            with open(fname3, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t, с", "dT/dt, °C/с"])
                for t, rate in zip(self._t_arr, dTdt):
                    writer.writerow([f"{t:.4f}", f"{rate:.4f}"])
            messagebox.showinfo("Экспорт", f"Сохранено:\n{fname}\n{fname2}\n{fname3}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    app = WeldingApp()
    app.mainloop()