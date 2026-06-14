"""
gui/app.py
Главный класс приложения WeldingApp
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
import traceback
from datetime import datetime
warnings.filterwarnings("ignore")

# Импорт из core
from core.math_engine import (mu_n, beta_n, source_rect, source_gauss,
                              compute_temperature_detailed, compute_spatial_field)
from core.analytics import (compute_t85, compute_weld_width, compute_haz_width, predict_structure)
from core import materials

# Импорт из plots
from gui import plots
from gui.plots import (FONT_UI, FONT_UI_BOLD, FONT_PLOT_LABEL, FONT_PLOT_TITLE,
                       COLORS_DARK, COLORS_LIGHT, CMAP_HEAT,
                       styled_entry, styled_button, separator, section_frame, apply_mpl_style,
                       normalize_color, map_color, apply_theme_recursive)


class WeldingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Моделирование и анализ теплового процесса сварки пластин распределенным источником энергии")
        self.geometry("1500x950")
        self.minsize(1300, 800)
        self.configure(bg=plots.COLORS["bg"])
        
        materials.load_materials()
        self._init_params()
        self._build_ui()
        self.after(300, self._run_all)

    def _update_highlights(self):
        if hasattr(self, 'tmax_frame'):
            self.tmax_frame.configure(bg=plots.COLORS["accent2"])
            self.tmax_label.configure(bg=plots.COLORS["accent2"], fg="#ffffff")
        if hasattr(self, 'frm6_highlight'):
            self.frm6_highlight.configure(highlightbackground=plots.COLORS["accent2"])
            self.hdr6.configure(bg=plots.COLORS["accent2"])
            self.hdr6_label.configure(bg=plots.COLORS["accent2"])
        if hasattr(self, 'cycle_hdr'):
            self.cycle_hdr.configure(bg=plots.COLORS["accent2"])
            self.cycle_hdr_label.configure(bg=plots.COLORS["accent2"])
        if hasattr(self, 'rec_hdr'):
            self.rec_hdr.configure(bg=plots.COLORS["accent2"])
            self.rec_hdr_label.configure(bg=plots.COLORS["accent2"])

    def _update_ttk_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=plots.COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=plots.COLORS["bg3"], foreground=plots.COLORS["text2"], 
                        font=FONT_UI_BOLD, padding=[8, 6], justify="center")
        style.map("TNotebook.Tab", background=[("selected", plots.COLORS["bg2"])], foreground=[("selected", plots.COLORS["accent"])])
        style.configure("TScrollbar", background=plots.COLORS["bg3"], troughcolor=plots.COLORS["bg2"], bordercolor=plots.COLORS["border"], arrowcolor=plots.COLORS["text"])
        style.map("TScrollbar", background=[("active", plots.COLORS["accent"]), ("!active", plots.COLORS["bg3"])])
        style.configure("TCombobox", fieldbackground=plots.COLORS["bg3"], background=plots.COLORS["bg3"], foreground=plots.COLORS["text"], bordercolor=plots.COLORS["border"], arrowcolor=plots.COLORS["text"])
        style.map("TCombobox", fieldbackground=[("readonly", plots.COLORS["bg3"])])

    def toggle_theme(self):
        if plots.CURRENT_THEME == "dark":
            plots.CURRENT_THEME = "light"
            plots.COLORS = plots.COLORS_LIGHT
        else:
            plots.CURRENT_THEME = "dark"
            plots.COLORS = plots.COLORS_DARK
        self.configure(bg=plots.COLORS["bg"])
        apply_theme_recursive(self)
        self._update_ttk_style()
        self._update_highlights()
        
        # Сбрасываем кэш тепловой карты анимации, чтобы пересоздать colorbar
        self._anim_im = None 
        
        for fig_attr in ["fig_cycle", "fig_anim", "fig_cooling", "fig_components", "fig_propagation", 
                         "fig_length", "fig_phases", "fig_cct", "fig_heatmap", 
                         "fig_3d", "fig_source", "fig_fourier", "fig_physical"]:
            if hasattr(self, fig_attr):
                getattr(self, fig_attr).patch.set_facecolor(plots.COLORS["bg2"])
        if hasattr(self, '_results') and self._results is not None:
            self._update_plots_from_results()

    def _init_params(self):
        self.var_l = tk.DoubleVar(value=10.0)
        self.var_a = tk.DoubleVar(value=0.1)
        self.var_cV = tk.DoubleVar(value=4.2)
        self.var_lam = tk.DoubleVar(value=0.5)
        self.var_T_melt = tk.DoubleVar(value=1510.0)
        self.var_T_harden = tk.DoubleVar(value=723.0)
        self.var_T_init = tk.DoubleVar(value=20.0)
        self.var_density = tk.DoubleVar(value=7.85)
        
        self.var_src_type = tk.StringVar(value="Прямоугольный")
        self.var_display_mode = tk.StringVar(value="rect")
        
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
        self._results = None
        self.cbar_heatmap = None
        self.cbar_gradient = None
        
        # Параметры анимации
        self._anim_im = None
        self.anim_playing = False
        self.anim_frame_idx = 0
        self.anim_speed_mult = 1.0
        self.anim_after_id = None

    def _add_highlight_header(self, parent, text, attr_name):
        hdr = tk.Frame(parent, bg=plots.COLORS["accent2"], height=32)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        lbl = tk.Label(hdr, text=f"  {text} ", font=("Consolas", 12, "bold"), fg="#ffffff", bg=plots.COLORS["accent2"], anchor="w")
        lbl.pack(side="left", padx=10, fill="y")
        setattr(self, attr_name + "_hdr", hdr)
        setattr(self, attr_name + "_hdr_label", lbl)

    def _build_ui(self):
        hdr = tk.Frame(self, bg=plots.COLORS["header"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        
        tk.Label(hdr, text="✨ МОДЕЛИРОВАНИЕ ТЕПЛОВОГО ПРОЦЕССА СВАРКИ ",
                 font=("Consolas", 14, "bold"), fg=plots.COLORS["accent"], bg=plots.COLORS["header"]).pack(side="left", padx=10, pady=14)
        
        tk.Label(hdr, text="| Автор: Жданов А. О.", font=FONT_UI, fg=plots.COLORS["accent2"], bg=plots.COLORS["header"]).pack(side="left", padx=5)
        
        disp_frame_hdr = tk.Frame(hdr, bg=plots.COLORS["header"])
        disp_frame_hdr.pack(side="left", padx=30)
        tk.Label(disp_frame_hdr, text="Отображать:", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["header"]).pack(side="left", padx=5)
        tk.Radiobutton(disp_frame_hdr, text="Равномерный", variable=self.var_display_mode, value="rect", font=FONT_UI, fg=plots.COLORS["text"], bg=plots.COLORS["header"], selectcolor=plots.COLORS["bg3"], command=self._update_plots_from_results).pack(side="left", padx=5)
        tk.Radiobutton(disp_frame_hdr, text="Гауссов", variable=self.var_display_mode, value="gauss", font=FONT_UI, fg=plots.COLORS["text"], bg=plots.COLORS["header"], selectcolor=plots.COLORS["bg3"], command=self._update_plots_from_results).pack(side="left", padx=5)

        styled_button(hdr, "🌓 Тема", self.toggle_theme, color=plots.COLORS["bg3"]).pack(side="right", padx=20)
        
        status_bar = tk.Frame(self, bg=plots.COLORS["bg3"], height=40)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Готов к расчёту")
        tk.Label(status_bar, textvariable=self.status_var, font=FONT_UI_BOLD, fg=plots.COLORS["text2"], bg=plots.COLORS["bg3"], anchor="w").pack(side="left", padx=12, pady=6)
        
        self.tmax_frame = tk.Frame(status_bar, bg=plots.COLORS["accent2"], padx=15, pady=4)
        self.tmax_frame.pack(side="right", fill="y", padx=5, pady=5)
        self.tmax_label = tk.Label(self.tmax_frame, text=" T_max = --- °C  ", font=("Consolas", 12, "bold"), fg="#ffffff", bg=plots.COLORS["accent2"])
        self.tmax_label.pack()
        
        main = tk.Frame(self, bg=plots.COLORS["bg"])
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg=plots.COLORS["bg"], width=360)
        left.pack(fill="y", side="left")
        left.pack_propagate(False)
        self._build_params_panel(left)
        separator(main, orient="vertical").pack(side="left", fill="y", padx=0)
        right = tk.Frame(main, bg=plots.COLORS["bg"])
        right.pack(fill="both", expand=True)
        self._build_tabs(right)

    def _build_params_panel(self, parent):
        scrollbar = ttk.Scrollbar(parent, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        canvas = tk.Canvas(parent, bg=plots.COLORS["bg"], highlightthickness=0, yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=plots.COLORS["bg"])
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        def _on_frame_configure(e): canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e): canvas.itemconfig(win_id, width=e.width)
        scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        def _scroll(e):
            if e.delta: canvas.yview_scroll(-1 * (e.delta // 120), "units")
            elif getattr(e, "num", 0) == 4: canvas.yview_scroll(-1, "units")
            elif getattr(e, "num", 0) == 5: canvas.yview_scroll(1, "units")
        def _bind_scroll_recursive(widget):
            widget.bind("<MouseWheel>", _scroll, add="+")
            widget.bind("<Button-4>", _scroll, add="+")
            widget.bind("<Button-5>", _scroll, add="+")
            for child in widget.winfo_children(): _bind_scroll_recursive(child)
        def _setup_scroll(*_):
            _bind_scroll_recursive(scroll_frame)
            canvas.bind("<MouseWheel>", _scroll, add="+")
        scroll_frame.bind("<Map>", lambda e: self.after(200, _setup_scroll))
        self._rebind_scroll = _setup_scroll
        
        p = scroll_frame
        tk.Label(p, text=" ПАРАМЕТРЫ МОДЕЛИ ", font=("Consolas", 12, "bold"), fg=plots.COLORS["accent"], bg=plots.COLORS["bg"]).pack(anchor="w", padx=10, pady=(12,4))
        
        # ==================================================================================
        # БЛОК ВЫБОРА МАТЕРИАЛА
        # ==================================================================================
        frm_mat, inn_mat = section_frame(p, "Выбор материала")
        frm_mat.pack(fill="x", padx=8, pady=4)
        
        self.var_mat_source = tk.StringVar(value="builtin")
        rb_frame = tk.Frame(inn_mat, bg=plots.COLORS["bg2"])
        rb_frame.pack(fill="x", pady=(5,0))
        tk.Radiobutton(rb_frame, text="📚 Из базы", variable=self.var_mat_source, value="builtin", 
                       font=FONT_UI, fg=plots.COLORS["text"], bg=plots.COLORS["bg2"], selectcolor=plots.COLORS["bg3"],
                       command=self._on_mat_source_changed).pack(side="left", padx=5)
        tk.Radiobutton(rb_frame, text="👤 Пользовательский", variable=self.var_mat_source, value="custom", 
                       font=FONT_UI, fg=plots.COLORS["text"], bg=plots.COLORS["bg2"], selectcolor=plots.COLORS["bg3"],
                       command=self._on_mat_source_changed).pack(side="left", padx=5)
        
        self.var_material = tk.StringVar(value="Сталь 20")
        self.material_combo = ttk.Combobox(inn_mat, textvariable=self.var_material, state="readonly", width=32, font=FONT_UI)
        self.material_combo.pack(pady=5)
        self.material_combo.bind("<<ComboboxSelected>>", self._on_material_selected)
        
        btn_mat_frame = tk.Frame(inn_mat, bg=plots.COLORS["bg2"])
        btn_mat_frame.pack(fill="x", pady=2)
        
        self.btn_apply_mat = styled_button(btn_mat_frame, "↻ Применить", self._apply_material_props, color=plots.COLORS["accent3"])
        self.btn_apply_mat.pack(side="left", expand=True, fill="x", padx=2)
        
        self.btn_save_mat = styled_button(btn_mat_frame, "💾 Сохранить", self._save_custom_material, color=plots.COLORS["bg3"])
        self.btn_save_mat.pack(side="left", expand=True, fill="x", padx=2)
        
        self.btn_delete_mat = styled_button(btn_mat_frame, " 🗑 ", self._delete_custom_material, color=plots.COLORS["accent4"])
        self.btn_delete_mat.pack(side="left", expand=True, fill="x", padx=2)
        
        # Кнопки управления базой материалов
        btn_db_frame = tk.Frame(inn_mat, bg=plots.COLORS["bg2"])
        btn_db_frame.pack(fill="x", pady=2)
        styled_button(btn_db_frame, "📤 Экспорт базы", self._export_builtin_materials, color=plots.COLORS["bg3"]).pack(side="left", expand=True, fill="x", padx=2)
        styled_button(btn_db_frame, "📥 Импорт базы", self._import_builtin_materials, color=plots.COLORS["bg3"]).pack(side="left", expand=True, fill="x", padx=2)
        
        self.mat_desc_label = tk.Label(inn_mat, text="", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"], wraplength=320, justify="left")
        self.mat_desc_label.pack(pady=5)
        
        self._on_mat_source_changed()
        # ==================================================================================
        
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
        
        tk.Label(inn2, text="Форма источника (расчет):", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"]).pack(anchor="w", pady=(10,0))
        self.src_type_combo = ttk.Combobox(inn2, textvariable=self.var_src_type, values=["Равномерный", "Гауссов"], state="readonly", width=20, font=FONT_UI)
        self.src_type_combo.set("Равномерный")
        self.src_type_combo.bind("<<ComboboxSelected>>", self._on_src_type_selected)
        self.src_type_combo.pack(pady=2)
        
        self._param_row(inn2, "q_max, Вт/см³", self.var_q_max)
        self._param_row(inn2, "t_ист, с", self.var_t_source)
        
        self.rect_frame = tk.Frame(inn2, bg=plots.COLORS["bg2"])
        self.rect_frame.pack(fill="x")
        self._param_row(self.rect_frame, "y₁, см", self.var_y1)
        self._param_row(self.rect_frame, "y₂, см", self.var_y2)
        
        self.gauss_frame = tk.Frame(inn2, bg=plots.COLORS["bg2"])
        self._param_row(self.gauss_frame, "k, 1/см² (для Гаусса)", self.var_k_gauss)
        
        self._on_src_type_selected()
        
        frm3, inn3 = section_frame(p, "Граничные потоки")
        frm3.pack(fill="x", padx=8, pady=4)
        self._param_row(inn3, "q₁, Вт/см² (y=0)", self.var_q1)
        self._param_row(inn3, "q₂, Вт/см² (y=l)", self.var_q2)
        tk.Label(inn3, text="(< 0 = охлаждение, > 0 = подогрев)", font=FONT_UI, fg=plots.COLORS["warning"], bg=plots.COLORS["bg2"]).pack(anchor="w")
        tk.Label(inn3, text="⚠ Для эффекта используйте |q| > 100", font=FONT_UI, fg=plots.COLORS["accent2"], bg=plots.COLORS["bg2"]).pack(anchor="w")
        
        frm4, inn4 = section_frame(p, "Параметры расчёта")
        frm4.pack(fill="x", padx=8, pady=4)
        self._param_row(inn4, "t_max, с", self.var_t_max)
        self._param_row(inn4, "N_t (точек по времени)", self.var_N_t)
        self._param_row(inn4, "N (членов ряда)", self.var_N_terms)
        
        frm5, inn5 = section_frame(p, "Точка наблюдения A")
        frm5.pack(fill="x", padx=8, pady=4)
        self._param_row(inn5, "y_A, см", self.var_y_A)
        
        btn_frame = tk.Frame(p, bg=plots.COLORS["bg"])
        btn_frame.pack(fill="x", padx=8, pady=10)
        
        self.btn_calculate = styled_button(btn_frame, "▶ РАССЧИТАТЬ", self._run_all, color=plots.COLORS["accent"])
        self.btn_calculate.pack(fill="x", pady=2)
        styled_button(btn_frame, "↺ Сброс", self._reset_params, color=plots.COLORS["bg3"]).pack(fill="x", pady=2)
        styled_button(btn_frame, "💾 Экспорт", self._export_data, color=plots.COLORS["bg3"]).pack(fill="x", pady=2)
        
        self.frm6_highlight = tk.Frame(p, bg=plots.COLORS["bg2"], highlightthickness=2, highlightbackground=plots.COLORS["accent2"])
        self.frm6_highlight.pack(fill="x", padx=8, pady=4)
        self.hdr6 = tk.Frame(self.frm6_highlight, bg=plots.COLORS["accent2"])
        self.hdr6.pack(fill="x")
        self.hdr6_label = tk.Label(self.hdr6, text="  📊 Анализ результата ", font=FONT_UI_BOLD, fg="#ffffff", bg=plots.COLORS["accent2"], pady=6)
        self.hdr6_label.pack(side="left")
        inn6 = tk.Frame(self.frm6_highlight, bg=plots.COLORS["bg2"])
        inn6.pack(fill="both", expand=True, padx=12, pady=8)
        self.reco_text = tk.Text(inn6, height=8, width=36, font=FONT_UI, bg=plots.COLORS["bg3"], fg=plots.COLORS["text2"], relief="flat", state="disabled", wrap="word")
        self.reco_text.pack(fill="both")
        self._update_recommendations("—")

    def _param_row(self, parent, label, var):
        row = tk.Frame(parent, bg=plots.COLORS["bg2"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"], width=28, anchor="w").pack(side="left")
        styled_entry(row, var, width=10).pack(side="right")

    def _on_src_type_selected(self, event=None):
        if self.var_src_type.get() == "Равномерный":
            self.rect_frame.pack(fill="x")
            self.gauss_frame.pack_forget()
        else:
            self.rect_frame.pack_forget()
            self.gauss_frame.pack(fill="x")
        if hasattr(self, "_rebind_scroll"):
            self.after(50, self._rebind_scroll)

    # ==================================================================================
    # ЛОГИКА УПРАВЛЕНИЯ МАТЕРИАЛАМИ
    # ==================================================================================
    def _on_mat_source_changed(self):
        if self.var_mat_source.get() == "builtin":
            if not materials.BUILTIN_MATERIALS:
                self.material_combo["values"] = ["(нет материалов)"]
                self.var_material.set("(нет материалов)")
            else:
                self.material_combo["values"] = materials.BUILTIN_MATERIALS
                if self.var_material.get() not in materials.BUILTIN_MATERIALS:
                    self.var_material.set(materials.BUILTIN_MATERIALS[0])
        else:
            custom_list = [m for m in materials.MATERIALS_DB.keys() if m not in materials.BUILTIN_MATERIALS]
            if not custom_list:
                self.material_combo["values"] = ["(нет сохраненных)"]
                self.var_material.set("(нет сохраненных)")
            else:
                self.material_combo["values"] = custom_list
                if self.var_material.get() not in custom_list:
                    self.var_material.set(custom_list[0])
        self._update_mat_buttons_state()
        self._on_material_selected()

    def _update_mat_buttons_state(self):
        mat_name = self.var_material.get()
        is_builtin = mat_name in materials.BUILTIN_MATERIALS
        is_empty = mat_name == "(нет сохраненных)" or mat_name == "(нет материалов)"
        
        if is_builtin or is_empty:
            self.btn_delete_mat.config(state="disabled")
        else:
            self.btn_delete_mat.config(state="normal")
            
        if is_empty:
            self.btn_apply_mat.config(state="disabled")
        else:
            self.btn_apply_mat.config(state="normal")

    def _delete_custom_material(self):
        mat_name = self.var_material.get()
        if mat_name in materials.BUILTIN_MATERIALS or mat_name == "(нет сохраненных)":
            return
            
        if messagebox.askyesno("Подтверждение", f"Удалить материал '{mat_name}'?\nЭто действие необратимо."):
            if mat_name in materials.MATERIALS_DB:
                del materials.MATERIALS_DB[mat_name]
            materials.save_custom_materials()
            self._on_mat_source_changed()
            messagebox.showinfo("Успех", f"Материал '{mat_name}' удален.")

    def _on_material_selected(self, event=None):
        mat_name = self.var_material.get()
        if mat_name in materials.MATERIALS_DB:
            props = materials.MATERIALS_DB[mat_name]
            info = f"{props['desc']}\n\nc_V = {props['c_V']}\na = {props['a']}\nλ = {props['lam']}\nT_пл = {props['T_melt']}°C\nρ = {props['density']}"
            self.mat_desc_label.config(text=info)
        else:
            self.mat_desc_label.config(text="Нет доступных пользовательских материалов.\nСохраните текущие параметры как новый материал.")
        self._update_mat_buttons_state()

    def _apply_material_props(self):
        mat_name = self.var_material.get()
        if mat_name not in materials.MATERIALS_DB: return
        props = materials.MATERIALS_DB[mat_name]
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
        dialog.configure(bg=plots.COLORS["bg"])
        dialog.transient(self)
        dialog.grab_set()
        tk.Label(dialog, text="Название: ", fg=plots.COLORS["text"], bg=plots.COLORS["bg"], font=FONT_UI_BOLD).pack(pady=10)
        name_entry = tk.Entry(dialog, width=40, font=FONT_UI, bg=plots.COLORS["bg3"], fg=plots.COLORS["text"])
        name_entry.pack(pady=5)
        tk.Label(dialog, text="Описание: ", fg=plots.COLORS["text2"], bg=plots.COLORS["bg"], font=FONT_UI).pack(pady=(15,5))
        desc_entry = tk.Entry(dialog, width=40, font=FONT_UI, bg=plots.COLORS["bg3"], fg=plots.COLORS["text2"])
        desc_entry.pack(pady=5)
        def save():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Внимание", "Введите название")
                return
            if name in materials.BUILTIN_MATERIALS:
                messagebox.showwarning("Внимание", "Нельзя перезаписать базовый материал.\nВыберите другое название.")
                return
            if name in materials.MATERIALS_DB:
                if not messagebox.askyesno("Подтверждение", f"Перезаписать '{name}'?"): return
            desc = desc_entry.get().strip() or "Пользовательский материал"
            materials.MATERIALS_DB[name] = {"c_V": self.var_cV.get(), "a": self.var_a.get(), "lam": self.var_lam.get(), "T_melt": self.var_T_melt.get(), "T_harden": self.var_T_harden.get(), "density": self.var_density.get(), "desc": desc}
            materials.save_custom_materials()
            
            self.var_mat_source.set("custom")
            self._on_mat_source_changed()
            self.var_material.set(name)
            self._on_material_selected()
            
            messagebox.showinfo("Успех", f"Материал '{name}' сохранён!")
            dialog.destroy()
        btn_frame = tk.Frame(dialog, bg=plots.COLORS["bg"])
        btn_frame.pack(pady=20)
        styled_button(btn_frame, "Сохранить", save, color=plots.COLORS["accent"]).pack(side="left", padx=10)
        styled_button(btn_frame, "Отмена", dialog.destroy, color=plots.COLORS["bg3"]).pack(side="left", padx=10)

    def _export_builtin_materials(self):
        try:
            materials.export_builtin_materials()
            messagebox.showinfo("Успех", "Базовые материалы экспортированы в builtin_materials.json")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _import_builtin_materials(self):
        filepath = filedialog.askopenfilename(
            title="Импорт базы материалов",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        try:
            materials.import_builtin_materials(filepath)
            self._on_mat_source_changed()
            messagebox.showinfo("Успех", f"База материалов импортирована из {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Ошибка импорта", str(e))
    # ==================================================================================

    def _build_tabs(self, parent):
        self._update_ttk_style()
        btn_frame = tk.Frame(parent, bg=plots.COLORS["bg3"], height=32)
        btn_frame.pack(fill="x")
        btn_frame.pack_propagate(False)
        def scroll_left():
            try:
                idx = self.notebook.index(self.notebook.select())
                if idx > 0: self.notebook.select(idx - 1)
            except Exception: pass
        def scroll_right():
            try:
                idx = self.notebook.index(self.notebook.select())
                if idx < self.notebook.index("end") - 1: self.notebook.select(idx + 1)
            except Exception: pass
        styled_button(btn_frame, "◀", scroll_left, color=plots.COLORS["bg3"]).pack(side="left", padx=2)
        styled_button(btn_frame, "▶", scroll_right, color=plots.COLORS["bg3"]).pack(side="left", padx=2)
        tk.Label(btn_frame, text="  Навигация", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg3"]).pack(side="left", padx=10)
        
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)
        
        tabs_config = [
            ("tab_cycle", "🟧 Терм. цикл", self._build_tab_cycle),
            ("tab_animation", "🎬 Анимация", self._build_tab_animation),
            ("tab_cooling", "📉 Охлаждение", self._build_tab_cooling),
            ("tab_source", "⚡ Источники", self._build_tab_source),
            ("tab_fourier", "📊 Ряд Фурье", self._build_tab_fourier),
            ("tab_heatmap", "🗺 Карта", self._build_tab_heatmap),
            ("tab_metallurgy", "🔬 Металлургия", self._build_tab_metallurgy),
            ("tab_physical", "🔬 Физ. схема", self._build_tab_physical),
            ("tab_components", "🧩 Компоненты", self._build_tab_components),
            ("tab_propagation", "🔥 Распростр.", self._build_tab_propagation),
            ("tab_length", "📏 Длина", self._build_tab_length),
            ("tab_phases", "🌙 Фазы", self._build_tab_phases),
            ("tab_3d", "🔷 3D", self._build_tab_3d),
            ("tab_recommendations", "🚩 Рекомендации", self._build_tab_recommendations),
            ("tab_theory", "📖 Теория", self._build_tab_theory),
        ]
        
        for attr_name, title, build_func in tabs_config:
            frame = tk.Frame(self.notebook, bg=plots.COLORS["bg2"])
            self.notebook.add(frame, text=title)
            setattr(self, attr_name, frame)
            build_func(frame)

    def _build_tab_cycle(self, parent):
        self._add_highlight_header(parent, "🌡 ТЕРМИЧЕСКИЙ ЦИКЛ", "cycle")
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_cycle = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_cycle = fig
        self.canvas_cycle = canvas

    # ==================================================================================
    # ВКЛАДКА АНИМАЦИИ
    # ==================================================================================
    def _build_tab_animation(self, parent):
        self._add_highlight_header(parent, "🎬 АНИМАЦИЯ ПРОЦЕССА", "anim")
        
        # Панель управления
        ctrl = tk.Frame(parent, bg=plots.COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=5)
        
        self.btn_anim_play = styled_button(ctrl, "▶ Play", self._toggle_anim, color=plots.COLORS["accent3"])
        self.btn_anim_play.pack(side="left", padx=5)
        
        styled_button(ctrl, "⏹ Reset", self._reset_anim, color=plots.COLORS["bg3"]).pack(side="left", padx=5)
        
        tk.Label(ctrl, text="Время:", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"]).pack(side="left", padx=(20, 5))
        
        self.anim_slider = tk.Scale(ctrl, from_=0, to=100, orient="horizontal", length=300, 
                                    bg=plots.COLORS["bg3"], fg=plots.COLORS["text"], troughcolor=plots.COLORS["bg2"],
                                    highlightthickness=0, command=self._on_anim_slider_change)
        self.anim_slider.pack(side="left", padx=5)
        
        tk.Label(ctrl, text="Скорость:", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"]).pack(side="left", padx=(20, 5))
        self.var_anim_speed = tk.StringVar(value="1.0")
        for spd in ["0.5", "1.0", "2.0"]:
            tk.Radiobutton(ctrl, text=f"x{spd}", variable=self.var_anim_speed, value=spd,
                           font=FONT_UI, fg=plots.COLORS["text"], bg=plots.COLORS["bg2"], selectcolor=plots.COLORS["bg3"],
                           command=self._update_anim_speed).pack(side="left", padx=2)
                           
        self.anim_info_label = tk.Label(ctrl, text="t = 0.00 с | T_max = 0 °C", font=FONT_UI_BOLD, fg=plots.COLORS["accent"], bg=plots.COLORS["bg2"])
        self.anim_info_label.pack(side="right", padx=10)

        # Область графиков
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        self.fig_anim = Figure(figsize=(10, 8), facecolor=plots.COLORS["bg2"])
        gs = self.fig_anim.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1.2, 1])
        
        self.ax_anim_profile = self.fig_anim.add_subplot(gs[0, :])
        self.ax_anim_physical = self.fig_anim.add_subplot(gs[1, 0])
        self.ax_anim_heatmap = self.fig_anim.add_subplot(gs[1, 1])
        
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=frame)
        self.canvas_anim.get_tk_widget().pack(fill="both", expand=True)

    def _toggle_anim(self):
        if self._T_field is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчёт")
            return
        self.anim_playing = not self.anim_playing
        if self.anim_playing:
            self.btn_anim_play.config(text="⏸ Pause", bg=plots.COLORS["accent4"])
            if self.anim_frame_idx >= len(self._t_arr) - 1:
                self.anim_frame_idx = 0
            self._run_anim_loop()
        else:
            self.btn_anim_play.config(text="▶ Play", bg=plots.COLORS["accent3"])
            if self.anim_after_id:
                self.after_cancel(self.anim_after_id)
                self.anim_after_id = None

    def _reset_anim(self):
        self.anim_playing = False
        self.btn_anim_play.config(text="▶ Play", bg=plots.COLORS["accent3"])
        if self.anim_after_id:
            self.after_cancel(self.anim_after_id)
            self.anim_after_id = None
        self.anim_frame_idx = 0
        if hasattr(self, 'anim_slider'):
            self.anim_slider.set(0)
        self._draw_anim_frame(0)

    def _on_anim_slider_change(self, val):
        if self._T_field is None: return
        idx = int(float(val))
        idx = max(0, min(idx, len(self._t_arr) - 1))
        self.anim_frame_idx = idx
        self._draw_anim_frame(idx)

    def _update_anim_speed(self):
        self.anim_speed_mult = float(self.var_anim_speed.get())

    def _run_anim_loop(self):
        if not self.anim_playing: return
        if self.anim_frame_idx >= len(self._t_arr) - 1:
            self.anim_playing = False
            self.btn_anim_play.config(text="▶ Play", bg=plots.COLORS["accent3"])
            return
            
        self._draw_anim_frame(self.anim_frame_idx)
        if hasattr(self, 'anim_slider'):
            self.anim_slider.set(self.anim_frame_idx)
        
        delay = max(10, int(50 / self.anim_speed_mult))
        self.anim_frame_idx += 1
        self.anim_after_id = self.after(delay, self._run_anim_loop)

    def _draw_anim_frame(self, idx):
        if self._T_field is None: return
        p = self._params
        t_current = self._t_arr[idx]
        T_profile = self._T_field[idx, :]
        T_max_curr = T_profile.max()
        
        self.anim_info_label.config(text=f"t = {t_current:.2f} с | T_max = {T_max_curr:.0f} °C")
        
        # 1. Эволюция профиля T(y)
        ax1 = self.ax_anim_profile
        ax1.clear()
        apply_mpl_style(ax1, f"Эволюция профиля T(y) при t = {t_current:.2f} с", "Координата y, см", "Температура T, °C")
        ax1.plot(self._y_arr, T_profile, color=plots.COLORS["accent"], lw=2.5)
        ax1.fill_between(self._y_arr, 0, T_profile, alpha=0.2, color=plots.COLORS["accent"])
        
        ax1.axhline(p["T_melt"], color=plots.COLORS["melt"], lw=1.5, ls="--", alpha=0.7, label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax1.axhline(p["T_harden"], color=plots.COLORS["warning"], lw=1.5, ls="--", alpha=0.7, label=f"T_зак={p['T_harden']:.0f}°C")
            
        if p["src_type"] == "rect":
            ax1.axvspan(self.var_y1.get(), self.var_y2.get(), alpha=0.15, color=plots.COLORS["accent2"], label="Зона источника")
        else:
            ax1.axvline(p["l"]/2, color=plots.COLORS["accent2"], lw=2, ls=":", label="Центр Гаусса")
            
        ax1.set_ylim(0, max(T_max_curr * 1.1, p["T_melt"] * 1.1))
        ax1.legend(fontsize=FONT_PLOT_LABEL-1, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"], loc='upper right')
        ax1.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        
        # 2. Физическая схема (Шов и ЗТВ)
        ax2 = self.ax_anim_physical
        ax2.clear()
        apply_mpl_style(ax2, "Формирование шва и ЗТВ", "Ширина, см", "Толщина, см")
        
        l = p["l"]
        plate_height = 2.0
        plate = Rectangle((-l/2, 0), l, plate_height, color=plots.COLORS["bg3"], edgecolor=plots.COLORS["text"], lw=2)
        ax2.add_patch(plate)
        
        melted = T_profile >= p["T_melt"]
        if np.any(melted):
            melt_idx = np.where(melted)[0]
            y_left = self._y_arr[melt_idx[0]]
            y_right = self._y_arr[melt_idx[-1]]
            w_weld = y_right - y_left
            weld_rect = Rectangle((y_left - l/2, 0), w_weld, plate_height, color=plots.COLORS["melt"], alpha=0.8)
            ax2.add_patch(weld_rect)
            
        if p["T_harden"] > 0:
            hardened = T_profile >= p["T_harden"]
            if np.any(hardened):
                haz_idx = np.where(hardened)[0]
                y_left_h = self._y_arr[haz_idx[0]]
                y_right_h = self._y_arr[haz_idx[-1]]
                w_haz = y_right_h - y_left_h
                haz_rect = Rectangle((y_left_h - l/2, 0), w_haz, plate_height, color=plots.COLORS["warning"], alpha=0.5)
                ax2.add_patch(haz_rect)
                
        ax2.set_xlim(-l/2 - 0.5, l/2 + 0.5)
        ax2.set_ylim(-0.2, plate_height + 0.2)
        ax2.set_aspect('equal')
        ax2.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        
        # 3. Тепловая карта (Time-lapse)
        ax3 = self.ax_anim_heatmap
        T_plot = np.clip(self._T_field[:idx+1, :], 0, None).T
        extent = [self._t_arr[0], t_current, self._y_arr[0], self._y_arr[-1]]
        
        if not hasattr(self, '_anim_im') or self._anim_im is None:
            if hasattr(ax3, '_anim_cbar'):
                try: ax3._anim_cbar.remove()
                except: pass
            ax3.clear()
            apply_mpl_style(ax3, "Тепловая карта (Time-lapse)", "Время t, с", "Координата y, см")
            vmin = 0
            vmax = np.clip(self._T_field.max(), p["T_melt"], None)
            self._anim_im = ax3.imshow(T_plot, aspect="auto", origin="lower", extent=extent, cmap=CMAP_HEAT, vmin=vmin, vmax=vmax)
            ax3._anim_cbar = self.fig_anim.colorbar(self._anim_im, ax=ax3, pad=0.02)
            ax3._anim_cbar.set_label("T, °C", color=plots.COLORS["text2"])
        else:
            self._anim_im.set_data(T_plot)
            self._anim_im.set_extent(extent)
            
        ax3.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        
        self.canvas_anim.draw_idle()
    # ==================================================================================

    def _build_tab_cooling(self, parent):
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_cooling = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_cooling = fig
        self.canvas_cooling = canvas

    def _build_tab_source(self, parent):
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        self.ax_src_rect = fig.add_subplot(gs[0])
        self.ax_src_gauss = fig.add_subplot(gs[1])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_source = fig
        self.canvas_source = canvas

    def _build_tab_fourier(self, parent):
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=2)
        ctrl = tk.Frame(frame, bg=plots.COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=2)
        tk.Label(ctrl, text="N гармоник: ", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"]).pack(side="left", padx=6)
        self.var_n_show = tk.IntVar(value=8)
        styled_entry(ctrl, self.var_n_show, width=5).pack(side="left")
        styled_button(ctrl, "Обновить", self._update_fourier_tab, color=plots.COLORS["accent"]).pack(side="left", padx=8)
        
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        gs = fig.add_gridspec(2, 1, hspace=0.4)
        self.ax_fourier_modes = fig.add_subplot(gs[0])
        self.ax_fourier_conv = fig.add_subplot(gs[1])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig_fourier = fig
        self.canvas_fourier = canvas

    def _build_tab_heatmap(self, parent):
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        fig = Figure(figsize=(10, 9), facecolor=plots.COLORS["bg2"])
        gs = fig.add_gridspec(2, 1, hspace=0.4, height_ratios=[1, 1])
        self.ax_heatmap = fig.add_subplot(gs[0])
        self.ax_gradient = fig.add_subplot(gs[1])
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_heatmap = fig
        self.canvas_heatmap = canvas

    def _build_tab_metallurgy(self, parent):
        top_frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        top_frame.pack(fill="both", expand=True, padx=4, pady=2)
        fig_cct = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_cct = fig_cct.add_subplot(111)
        fig_cct.tight_layout(pad=2)
        canvas_cct = FigureCanvasTkAgg(fig_cct, master=top_frame)
        canvas_cct.get_tk_widget().pack(fill="both", expand=True)
        toolbar_cct = tk.Frame(top_frame, bg=plots.COLORS["bg3"])
        toolbar_cct.pack(fill="x")
        NavigationToolbar2Tk(canvas_cct, toolbar_cct)
        self.fig_cct = fig_cct
        self.canvas_cct = canvas_cct
        bottom_frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        bottom_frame.pack(fill="x", padx=8, pady=4)
        info_frame = tk.Frame(bottom_frame, bg=plots.COLORS["bg3"])
        info_frame.pack(fill="x", pady=4)
        self.lbl_t85 = tk.Label(info_frame, text="t₈/₅: — с ", font=FONT_UI_BOLD, fg=plots.COLORS["accent"], bg=plots.COLORS["bg3"])
        self.lbl_t85.pack(side="left", padx=10, pady=5)
        self.lbl_weld_width = tk.Label(info_frame, text="Ширина шва: — см ", font=FONT_UI_BOLD, fg=plots.COLORS["accent2"], bg=plots.COLORS["bg3"])
        self.lbl_weld_width.pack(side="left", padx=10, pady=5)
        self.lbl_haz_width = tk.Label(info_frame, text="Ширина ЗТВ: — см ", font=FONT_UI_BOLD, fg=plots.COLORS["accent3"], bg=plots.COLORS["bg3"])
        self.lbl_haz_width.pack(side="left", padx=10, pady=5)
        self.lbl_structure = tk.Label(info_frame, text="Структура: — ", font=FONT_UI_BOLD, fg=plots.COLORS["text"], bg=plots.COLORS["bg3"])
        self.lbl_structure.pack(side="left", padx=10, pady=5)
        info = tk.Label(bottom_frame, text="ⓘ t₈/₅ (800→500°C) — стандарт для сталей. Для Al/Cu не применим. ", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"])
        info.pack(anchor="w", padx=5, pady=2)

    def _build_tab_physical(self, parent):
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_physical = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = tk.Frame(frame, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_physical = fig
        self.canvas_physical = canvas
        info = tk.Label(frame, text="Схематическое поперечное сечение сварного соединения (масштаб по Y увеличен для наглядности)", 
                        font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"])
        info.pack(pady=5)

    def _build_tab_components(self, parent):
        fig = Figure(figsize=(10, 6), facecolor=plots.COLORS["bg2"])
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_comp_total = fig.add_subplot(gs[0, :])
        self.ax_comp_A = fig.add_subplot(gs[1, 0])
        self.ax_comp_B = fig.add_subplot(gs[1, 1])
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_components = fig
        self.canvas_components = canvas

    def _build_tab_propagation(self, parent):
        ctrl = tk.Frame(parent, bg=plots.COLORS["bg2"])
        ctrl.pack(fill="x", padx=8, pady=6)
        tk.Label(ctrl, text="Время, с: ", font=FONT_UI, fg=plots.COLORS["text2"], bg=plots.COLORS["bg2"]).pack(side="left", padx=6)
        self.var_t_anim = tk.DoubleVar(value=1.0)
        styled_entry(ctrl, self.var_t_anim, width=8).pack(side="left")
        styled_button(ctrl, "Показать", self._update_propagation, color=plots.COLORS["accent"]).pack(side="left", padx=8)
        self.anim_time_label = tk.Label(ctrl, text=" ", font=FONT_UI_BOLD, fg=plots.COLORS["accent3"], bg=plots.COLORS["bg2"])
        self.anim_time_label.pack(side="left", padx=10)
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_propagation = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_propagation = fig
        self.canvas_propagation = canvas

    def _build_tab_length(self, parent):
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_length = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_length = fig
        self.canvas_length = canvas

    def _build_tab_phases(self, parent):
        fig = Figure(figsize=(10, 5), facecolor=plots.COLORS["bg2"])
        self.ax_phases = fig.add_subplot(111)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_phases = fig
        self.canvas_phases = canvas

    def _build_tab_3d(self, parent):
        fig = Figure(figsize=(10, 6), facecolor=plots.COLORS["bg2"])
        self.ax_3d = fig.add_subplot(111, projection="3d")
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        toolbar = tk.Frame(parent, bg=plots.COLORS["bg3"])
        toolbar.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar)
        self.fig_3d = fig
        self.canvas_3d = canvas

    def _build_tab_recommendations(self, parent):
        self._add_highlight_header(parent, "📋 РАСШИРЕННЫЕ РЕКОМЕНДАЦИИ", "rec")
        frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.rec_text_widget = tk.Text(frame, wrap=tk.WORD, font=FONT_UI, bg=plots.COLORS["bg3"], fg=plots.COLORS["text"], relief="flat", padx=20, pady=20)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.rec_text_widget.yview)
        self.rec_text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.rec_text_widget.pack(fill="both", expand=True)
        self.rec_text_widget.configure(state="disabled")

    def _build_tab_theory(self, parent):
        text_frame = tk.Frame(parent, bg=plots.COLORS["bg2"])
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=FONT_UI, bg=plots.COLORS["bg3"], fg=plots.COLORS["text"], relief="flat", padx=20, pady=20)
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

ПАРАМЕТР t₈/₅:
Время охлаждения от 800°C до 500°C — ключевой параметр для сварки сталей.
Определяет структуру металла шва и ЗТВ. Для цветных металлов (Al, Cu) не применим.
Критические значения t₈/₅ для углеродистых сталей:
• t₈/₅ < 2 с → Мартенсит (хрупкий)
• 2 < t₈/₅ < 10 с → Мартенсит + Бейнит
• 10 < t₈/₅ < 30 с → Бейнит
• 30 < t₈/₅ < 100 с → Перлит + Феррит
• t₈/₅ > 100 с → Грубый феррит

ШИРИНА ШВА И ЗТВ:
• Ширина шва: определяется по изотерме T_пл (зона плавления)
• Ширина ЗТВ: определяется по изотерме T_зак = AC₁ (зона термического влияния)
Для меди и алюминия используется температура рекристаллизации (~300°C).

CCT-ДИАГРАММА:
Диаграмма непрерывного охлаждения показывает фазовые превращения
в зависимости от скорости охлаждения. M_s — температура начала мартенситного превращения.
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
        phi_func = (lambda y: np.full_like(y, T_init)) if T_init != 0 else None
        q1_func = (lambda tau: np.full_like(tau, q1_val)) if q1_val != 0 else None
        q2_func = (lambda tau: np.full_like(tau, q2_val)) if q2_val != 0 else None
        return dict(l=l, a=a, c_V=c_V, lam=lam, q_max=q_max, t_source_end=t_src, 
                    phi_func=phi_func, q1_func=q1_func, q2_func=q2_func, t_max=t_max, N_t=N_t, 
                    y_A=y_A, N_terms=N, T_melt=T_melt, T_harden=T_harden, T_init=T_init, 
                    q1_val=q1_val, q2_val=q2_val, material=self.var_material.get())

    def _solve_scenario(self, p):
        l, a, c_V, lam = p["l"], p["a"], p["c_V"], p["lam"]
        t_max, N_t, N_terms = p["t_max"], p["N_t"], p["N_terms"]
        T_init = p["T_init"]
        
        y_arr = np.linspace(0, l, 100)
        t_arr = np.linspace(0.05, t_max, N_t)
        
        phi_func = (lambda y: np.full_like(y, T_init)) if T_init != 0 else None
        q1_func = (lambda tau: np.full_like(tau, p["q1_val"])) if p["q1_val"] != 0 else None
        q2_func = (lambda tau: np.full_like(tau, p["q2_val"])) if p["q2_val"] != 0 else None
        
        kw = dict(l=l, a=a, c_V=c_V, lam=lam, q_func=p["q_func"], t_source_end=p["t_source_end"],
                  phi_func=phi_func, q1_func=q1_func, q2_func=q2_func, N_terms=N_terms)
        
        T_field = compute_spatial_field(y_arr, t_arr, **kw) + T_init
        i_yA = int(np.clip(np.round(p["y_A"] / l * (len(y_arr) - 1)), 0, len(y_arr) - 1))
        T_cycle = T_field[:, i_yA]
        
        t85 = compute_t85(t_arr, T_cycle)
        weld_width = compute_weld_width(y_arr, T_field, p["T_melt"])
        haz_width_val, haz_width_text = compute_haz_width(y_arr, T_field, p["T_harden"], p.get("material", ""))
        structure, struct_color = predict_structure(t85, T_cycle.max(), p["T_melt"])
        
        return {
             "y_arr": y_arr, "t_arr": t_arr, "T_field": T_field, "T_cycle": T_cycle,
             "t85": t85, "weld_width": weld_width, "haz_width_val": haz_width_val, 
             "haz_width_text": haz_width_text, "structure": structure, "struct_color": struct_color,
             "T_max": T_field.max(), "i_yA": i_yA
        }

    def _run_all(self):
        self.btn_calculate.config(state="disabled", text="⏳ Вычисление...")
        self.update()
        
        try:
            p_base = self._get_params()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self.btn_calculate.config(state="normal", text="▶ РАССЧИТАТЬ")
            return
            
        try:
            self.status_var.set("⏳ Вычисление (Равномерный)...")
            self.update()
            
            p_rect = p_base.copy()
            p_rect["q_func"] = lambda eta: source_rect(eta, p_base["q_max"], self.var_y1.get(), self.var_y2.get())
            res_rect = self._solve_scenario(p_rect)
            
            self.status_var.set("⏳ Вычисление (Гауссов)...")
            self.update()
            
            p_gauss = p_base.copy()
            p_gauss["q_func"] = lambda eta: source_gauss(eta, p_base["q_max"], self.var_k_gauss.get(), p_base["l"])
            res_gauss = self._solve_scenario(p_gauss)
            
            self._results = {"rect": res_rect, "gauss": res_gauss, "params": p_base}
            self._y_arr = res_rect["y_arr"]
            self._t_arr = res_rect["t_arr"]
            
            self._update_plots_from_results()
            
            tmax_r = res_rect["T_max"]
            tmax_g = res_gauss["T_max"]
            self.tmax_label.config(text=f" T_max = {tmax_r:.0f}°C (Р) / {tmax_g:.0f}°C (Г)  ")
            self.status_var.set(f"✅ Готово | {p_base['material']}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Ошибка расчета", f"Произошла ошибка при расчете или отрисовке:\n{e}")
        finally:
            self.btn_calculate.config(state="normal", text="▶ РАССЧИТАТЬ")

    def _update_plots_from_results(self):
        if not hasattr(self, '_results') or self._results is None:
            return
            
        mode = self.var_display_mode.get()
        res = self._results[mode]
        p = self._results["params"].copy()
        
        src_type_val = "rect" if self.var_src_type.get() == "Равномерный" else "gauss"
        p["src_type"] = src_type_val
        
        if src_type_val == "rect":
            p["q_func"] = lambda eta: source_rect(eta, p["q_max"], self.var_y1.get(), self.var_y2.get())
        else:
            p["q_func"] = lambda eta: source_gauss(eta, p["q_max"], self.var_k_gauss.get(), p["l"])
        
        self._T_field = res["T_field"]
        self._T_cycle = res["T_cycle"]
        self._params = p
        
        self._plot_cycle(res["t_arr"], res["T_cycle"], p)
        self._plot_cooling_rate(res["t_arr"], res["T_cycle"], p)
        self._plot_components(res["y_arr"], res["t_arr"], p)
        self._update_propagation()
        self._plot_length_effect(p)
        self._plot_phases(res["t_arr"], res["T_cycle"], p)
        self._plot_metallurgy(res["t_arr"], res["T_cycle"], res["y_arr"], res["T_field"], p)
        self._plot_physical(res, p)
        self._plot_heatmap(res["y_arr"], res["t_arr"], res["T_field"], p)
        self._plot_gradient(res["y_arr"], res["t_arr"], res["T_field"], p)
        self._plot_3d(res["y_arr"], res["t_arr"], res["T_field"])
        self._plot_source(p)
        self._update_fourier_tab()
        
        # Обновляем анимацию
        if hasattr(self, 'canvas_anim'):
            self._anim_im = None # Сбрасываем кэш тепловой карты
            if hasattr(self, 'anim_slider'):
                self.anim_slider.config(to=len(res["t_arr"])-1)
            self._draw_anim_frame(0)
        
        self._update_recommendations(self._analyze(res["t_arr"], res["T_cycle"], p))
        
        res_rect = self._results["rect"]
        res_gauss = self._results["gauss"]
        ext_recs = self._generate_extended_recommendations(p, res, res_rect, res_gauss)
        self.rec_text_widget.configure(state="normal")
        self.rec_text_widget.delete("1.0", "end")
        self.rec_text_widget.insert("1.0", ext_recs)
        self.rec_text_widget.configure(state="disabled")

    def _plot_cycle(self, t_arr, T_cycle, p):
        ax = self.ax_cycle
        ax.clear()
        apply_mpl_style(ax, f"Термический цикл y_A={p['y_A']} см", "Время t, с", "Температура T, °C")
        ax.plot(t_arr, T_cycle, color=plots.COLORS["accent"], lw=2.5, label="T(y_A, t)")
        T_melt = p["T_melt"]
        T_harden = p["T_harden"]
        if T_cycle.max() > T_melt:
            ax.axhline(T_melt, color=plots.COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={T_melt:.0f}°C")
            ax.fill_between(t_arr, T_melt, T_cycle, where=T_cycle >= T_melt, alpha=0.18, color=plots.COLORS["melt"], label="Плавление")
        if T_harden > 0 and T_cycle.max() > T_harden:
            ax.axhline(T_harden, color=plots.COLORS["warning"], lw=1.5, ls="--", label=f"T_зак={T_harden:.0f}°C")
        ax.axvline(p["t_source_end"], color=plots.COLORS["accent3"], lw=1, ls=":", label=f"Конец источника")
        idx_max = np.argmax(T_cycle)
        ax.scatter(t_arr[idx_max], T_cycle[idx_max], color=plots.COLORS["accent2"], zorder=5, s=80)
        ax.annotate(f"  T_max={T_cycle[idx_max]:.0f}°C", xy=(t_arr[idx_max], T_cycle[idx_max]), color=plots.COLORS["accent2"], fontsize=FONT_PLOT_LABEL, fontweight="bold", xytext=(6, -20), textcoords="offset points")
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_cycle.draw()

    def _plot_cooling_rate(self, t_arr, T_cycle, p):
        ax = self.ax_cooling
        ax.clear()
        dTdt = np.gradient(T_cycle, t_arr) 
        apply_mpl_style(ax, "Скорость охлаждения dT/dt", "Время t, с", "dT/dt, °C/с")
        ax.fill_between(t_arr, 0, dTdt, alpha=0.3, color=plots.COLORS["accent4"])
        ax.plot(t_arr, dTdt, color=plots.COLORS["accent4"], lw=2, label="dT/dt")
        ax.axhline(-30, color=plots.COLORS["warning"], lw=1.5, ls="--", label="Критическая (30°C/с)")
        ax.axhline(-50, color=plots.COLORS["melt"], lw=1.5, ls="--", label="Закалка (50°C/с)")
        ax.axhline(0, color=plots.COLORS["text2"], lw=0.5, ls="-")
        idx_cool = np.argmin(dTdt)
        ax.scatter(t_arr[idx_cool], dTdt[idx_cool], color=plots.COLORS["accent2"], zorder=5, s=80)
        ax.annotate(f"  max={dTdt[idx_cool]:.1f}°C/с", xy=(t_arr[idx_cool], dTdt[idx_cool]), color=plots.COLORS["accent2"], fontsize=FONT_PLOT_LABEL, fontweight="bold")
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_cooling.draw()

    def _plot_components(self, y_arr, t_arr, p):
        y_A = p["y_A"]
        y_pt = np.array([y_A])
        T_A_arr, T_B_arr, T_C_arr, T_total_arr = [], [], [], []
        kw = dict(l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"], q_func=p["q_func"], t_source_end=p["t_source_end"], phi_func=p["phi_func"], q1_func=p["q1_func"], q2_func=p["q2_func"], N_terms=p["N_terms"])
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
        ax1.plot(t_arr, T_total_arr, color=plots.COLORS["text"], lw=2, label="Всего T")
        ax1.plot(t_arr, T_A_arr, color=plots.COLORS["accent"], lw=1.5, ls="--", label="A: начальное")
        ax1.plot(t_arr, T_B_arr, color=plots.COLORS["accent2"], lw=1.5, ls="--", label="B: потоки")
        ax1.plot(t_arr, T_C_arr, color=plots.COLORS["accent3"], lw=1.5, ls="--", label="C: источник")
        ax1.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax1.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        ax2 = self.ax_comp_A
        ax2.clear()
        apply_mpl_style(ax2, "A: Начальное условие", "Время t, с", "T_A, °C")
        ax2.plot(t_arr, T_A_arr, color=plots.COLORS["accent"], lw=2)
        ax2.fill_between(t_arr, 0, T_A_arr, alpha=0.3, color=plots.COLORS["accent"])
        ax2.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        ax3 = self.ax_comp_B
        ax3.clear()
        apply_mpl_style(ax3, "B: Граничные потоки", "Время t, с", "T_B, °C")
        ax3.plot(t_arr, T_B_arr, color=plots.COLORS["accent2"], lw=2)
        ax3.fill_between(t_arr, 0, T_B_arr, alpha=0.3, color=plots.COLORS["accent2"])
        ax3.axhline(0, color=plots.COLORS["text2"], lw=0.5, ls="-")
        ax3.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_components.draw()

    def _update_propagation(self, *_):
        if self._T_field is None: return
        ax = self.ax_propagation
        ax.clear()
        p = self._params
        try: t_show = float(self.var_t_anim.get())
        except Exception: t_show = self._t_arr[0]
        idx_t = int(np.argmin(np.abs(self._t_arr - t_show)))
        t_actual = self._t_arr[idx_t]
        T_profile = self._T_field[idx_t, :]
        apply_mpl_style(ax, f"Распространение тепла при t={t_actual:.2f} с", "Координата y, см", "Температура T, °C")
        ax.plot(self._y_arr, T_profile, color=plots.COLORS["accent"], lw=3, label="T(y)")
        ax.fill_between(self._y_arr, 0, T_profile, alpha=0.2, color=plots.COLORS["accent"])
        if p["src_type"] == "rect":
            ax.axvspan(self.var_y1.get(), self.var_y2.get(), alpha=0.15, color=plots.COLORS["accent2"], label="Зона источника")
        else:
            ax.axvline(p["l"]/2, color=plots.COLORS["accent2"], lw=2, ls=":", label="Центр Гаусса")
        ax.axhline(p["T_melt"], color=plots.COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax.axhline(p["T_harden"], color=plots.COLORS["warning"], lw=1.5, ls="--", label=f"T_зак={p['T_harden']:.0f}°C")
        ax.axvline(p["y_A"], color=plots.COLORS["accent3"], lw=1.5, ls=":", label=f"y_A={p['y_A']} см")
        idx_max = np.argmax(T_profile)
        ax.annotate(f"  T_max={T_profile[idx_max]:.0f}°C\n  y={self._y_arr[idx_max]:.1f} см", xy=(self._y_arr[idx_max], T_profile[idx_max]), color=plots.COLORS["accent2"], fontsize=FONT_PLOT_LABEL, fontweight="bold", xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", facecolor=plots.COLORS["bg3"], edgecolor=plots.COLORS["accent2"]))
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.anim_time_label.config(text=f"t={t_actual:.2f} с | T_max={T_profile.max():.0f}°C")
        self.canvas_propagation.draw()

    def _plot_length_effect(self, p):
        ax = self.ax_length
        ax.clear()
        lengths = np.linspace(5, 30, 10)
        T_max_values = []
        kw_base = dict(l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"], t_source_end=p["t_source_end"], phi_func=p["phi_func"], q1_func=p["q1_func"], q2_func=p["q2_func"], N_terms=p["N_terms"])
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
        ax.plot(lengths, T_max_values, color=plots.COLORS["accent"], lw=2, marker="o", markersize=6)
        ax.fill_between(lengths, 0, T_max_values, alpha=0.2, color=plots.COLORS["accent"])
        ax.axvline(p["l"], color=plots.COLORS["accent2"], lw=1.5, ls="--", label=f"Текущая l={p['l']} см")
        ax.axhline(p["T_melt"], color=plots.COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_length.draw()

    def _plot_phases(self, t_arr, T_cycle, p):
        ax = self.ax_phases
        ax.clear()
        phases = []
        phase_colors = []
        for T_val in T_cycle:
            if T_val >= p["T_melt"]:
                phases.append("Жидкость")
                phase_colors.append(plots.COLORS["melt"])
            elif p["T_harden"] > 0 and T_val >= p["T_harden"]:
                phases.append("Аустенит")
                phase_colors.append(plots.COLORS["warning"])
            else:
                phases.append("Феррит+Перлит")
                phase_colors.append(plots.COLORS["accent"])
        apply_mpl_style(ax, "Фазовые превращения", "Время t, с", "Температура T, °C")
        for i in range(len(t_arr)-1):
            ax.axvspan(t_arr[i], t_arr[i+1], ymin=0, ymax=1, alpha=0.15, color=phase_colors[i])
        ax.plot(t_arr, T_cycle, color=plots.COLORS["text"], lw=2, label="T(t)")
        ax.axhline(p["T_melt"], color=plots.COLORS["melt"], lw=1.5, ls="--", label=f"T_пл={p['T_melt']:.0f}°C")
        if p["T_harden"] > 0:
            ax.axhline(p["T_harden"], color=plots.COLORS["warning"], lw=1.5, ls="--", label=f"T_зак={p['T_harden']:.0f}°C")
        legend_elements = [Patch(facecolor=plots.COLORS["melt"], alpha=0.3, label='Жидкость'), Patch(facecolor=plots.COLORS["warning"], alpha=0.3, label='Аустенит'), Patch(facecolor=plots.COLORS["accent"], alpha=0.3, label='Феррит+Перлит')]
        ax.legend(handles=legend_elements, fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"], loc='upper right')
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_phases.draw()

    def _plot_metallurgy(self, t_arr, T_cycle, y_arr, T_field, p):
        ax = self.ax_cct
        ax.clear()
        t85 = compute_t85(t_arr, T_cycle)
        weld_width = compute_weld_width(y_arr, T_field, p["T_melt"])
        haz_width_val, haz_width_text = compute_haz_width(y_arr, T_field, p["T_harden"], p.get("material", ""))
        structure, struct_color = predict_structure(t85, T_cycle.max(), p["T_melt"])
        if t85 is not None: self.lbl_t85.config(text=f"t₈/₅: {t85:.2f} с")
        else: self.lbl_t85.config(text="t₈/₅: N/A (T_max < 800°C)")
        self.lbl_weld_width.config(text=f"Ширина шва: {weld_width:.3f} см")
        self.lbl_haz_width.config(text=f"Ширина ЗТВ: {haz_width_text}")
        self.lbl_structure.config(text=f"Структура: {structure}", fg=struct_color)
        apply_mpl_style(ax, "CCT-диаграмма с термическим циклом", "Время, с (лог)", "Температура, °C")
        ax.set_xscale("log")
        ax.fill_between([0.1, 100], [700, 500], alpha=0.2, color="lightblue", label="Феррит")
        ax.fill_between([10, 1000], [600, 400], alpha=0.2, color="lightgreen", label="Перлит")
        ax.fill_between([100, 10000], [400, 250], alpha=0.2, color="yellow", label="Бейнит")
        ax.axhline(200, color="red", lw=1.5, ls="--", alpha=0.7)
        ax.text(0.5, 215, "M_s ≈ 200°C (начало мартенситного превращения)", fontsize=FONT_PLOT_LABEL, color="red", fontweight="bold")
        idx_max = np.argmax(T_cycle)
        t_cool = t_arr[idx_max:] - t_arr[idx_max] + 1
        T_cool = T_cycle[idx_max:]
        valid = t_cool > 0
        if np.any(valid):
            ax.plot(t_cool[valid], T_cool[valid], color=plots.COLORS["accent"], lw=2.5, label="Термический цикл")
        max_time = max(100, t85 * 5) if t85 else 10000
        ax.set_xlim(0.1, max_time)
        ax.set_ylim(0, 1600)
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        ax.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        self.canvas_cct.draw()

    def _plot_physical(self, res, p):
        ax = self.ax_physical
        ax.clear()
        weld_w = res["weld_width"]
        haz_w = res["haz_width_val"]
        l = p["l"]
        plate_height = 2.0 
        plate = Rectangle((-l/2, 0), l, plate_height, color=plots.COLORS["bg3"], edgecolor=plots.COLORS["text"], lw=2, label="Основной металл")
        ax.add_patch(plate)
        if haz_w > 0:
            haz = Rectangle((-haz_w/2, 0), haz_w, plate_height, color=plots.COLORS["warning"], alpha=0.5, label=f"ЗТВ ({haz_w:.3f} см)")
            ax.add_patch(haz)
        if weld_w > 0:
            weld = Rectangle((-weld_w/2, 0), weld_w, plate_height, color=plots.COLORS["melt"], alpha=0.8, label=f"Шов ({weld_w:.3f} см)")
            ax.add_patch(weld)
        ax.axvline(0, color=plots.COLORS["accent"], lw=2, ls="--", label="Центр источника")
        ax.set_xlim(-l/2 - 1, l/2 + 1)
        ax.set_ylim(-0.5, plate_height + 0.5)
        ax.set_aspect('equal')
        apply_mpl_style(ax, "Схематическое поперечное сечение сварного соединения", "Ширина, см", "Толщина, см (масштаб x5)")
        ax.legend(loc="upper right", fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        self.canvas_physical.draw()

    def _plot_heatmap(self, y_arr, t_arr, T_field, p):
        ax = self.ax_heatmap
        ax.clear()
        if self.cbar_heatmap is not None:
            try: self.cbar_heatmap.remove()
            except Exception: pass
            self.cbar_heatmap = None
        T_plot = np.clip(T_field, 0, None).T
        extent = [t_arr[0], t_arr[-1], y_arr[0], y_arr[-1]]
        im = ax.imshow(T_plot, aspect="auto", origin="lower", extent=extent, cmap=CMAP_HEAT)
        self.cbar_heatmap = self.fig_heatmap.colorbar(im, ax=ax, pad=0.02)
        self.cbar_heatmap.set_label("T, °C", color=plots.COLORS["text2"], fontsize=FONT_PLOT_LABEL)
        self.cbar_heatmap.ax.yaxis.set_tick_params(color=plots.COLORS["text2"])
        plt.setp(self.cbar_heatmap.ax.yaxis.get_ticklabels(), color=plots.COLORS["text2"])
        ax.axhline(p["y_A"], color=plots.COLORS["accent3"], lw=1.5, ls="--", label=f"y_A={p['y_A']} см")
        try: ax.contour(t_arr, y_arr, T_plot, levels=[p["T_melt"]], colors=[plots.COLORS["melt"]], linewidths=1.5, linestyles="--")
        except Exception: pass
        apply_mpl_style(ax, "T(y, t) — тепловая карта", "Время t, с", "Координата y, см")
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        self.canvas_heatmap.draw()

    def _plot_gradient(self, y_arr, t_arr, T_field, p):
        ax = self.ax_gradient
        ax.clear()
        if self.cbar_gradient is not None:
            try: self.cbar_gradient.remove()
            except Exception: pass
            self.cbar_gradient = None
        grad_T = np.gradient(T_field, axis=1)
        T_plot = np.abs(grad_T.T)
        extent = [t_arr[0], t_arr[-1], y_arr[0], y_arr[-1]]
        im = ax.imshow(T_plot, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        self.cbar_gradient = self.fig_heatmap.colorbar(im, ax=ax, pad=0.02)
        self.cbar_gradient.set_label("|∇T|, °C/см", color=plots.COLORS["text2"], fontsize=FONT_PLOT_LABEL)
        self.cbar_gradient.ax.yaxis.set_tick_params(color=plots.COLORS["text2"])
        plt.setp(self.cbar_gradient.ax.yaxis.get_ticklabels(), color=plots.COLORS["text2"])
        ax.axhline(p["y_A"], color=plots.COLORS["accent3"], lw=1.5, ls="--", label=f"y_A={p['y_A']} см")
        apply_mpl_style(ax, "|∇T(y, t)| — градиенты температур", "Время t, с", "Координата y, см")
        ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        self.canvas_heatmap.draw()

    def _plot_3d(self, y_arr, t_arr, T_field):
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection="3d")
        ax.set_facecolor(plots.COLORS["bg2"])
        self.fig_3d.patch.set_facecolor(plots.COLORS["bg2"])
        step_y = max(1, len(y_arr)//40)
        step_t = max(1, len(t_arr)//40)
        Y = y_arr[::step_y]
        T = self._t_arr[::step_t]
        Z = T_field[::step_t, ::step_y]
        TT, YY = np.meshgrid(T, Y)
        Z_plot = np.clip(Z.T, 0, None)
        surf = ax.plot_surface(TT, YY, Z_plot, cmap=CMAP_HEAT, linewidth=0, antialiased=True, alpha=0.88)
        self.fig_3d.colorbar(surf, ax=ax, shrink=0.5, label="T, °C", pad=0.1)
        ax.set_xlabel("t, с", color=plots.COLORS["text2"], fontsize=FONT_PLOT_LABEL, labelpad=6)
        ax.set_ylabel("y, см", color=plots.COLORS["text2"], fontsize=FONT_PLOT_LABEL, labelpad=6)
        ax.set_zlabel("T, °C", color=plots.COLORS["text2"], fontsize=FONT_PLOT_LABEL, labelpad=6)
        ax.set_title("3D: T(y, t)", color=plots.COLORS["text"], fontsize=FONT_PLOT_TITLE, fontweight="bold")
        ax.tick_params(colors=plots.COLORS["text2"], labelsize=FONT_PLOT_LABEL-2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        self.canvas_3d.draw()

    def _plot_source(self, p):
        eta = np.linspace(0, p["l"], 600)
        ax1 = self.ax_src_rect
        ax1.clear()
        q_r = source_rect(eta, p["q_max"], self.var_y1.get(), self.var_y2.get())
        ax1.fill_between(eta, 0, q_r, alpha=0.4, color=plots.COLORS["accent2"])
        ax1.plot(eta, q_r, color=plots.COLORS["accent2"], lw=2)
        apply_mpl_style(ax1, "Равномерный источник", "y, см", "q(y), Вт/см³")
        ax1.set_ylim(bottom=-p["q_max"]*0.05)
        ax1.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        ax2 = self.ax_src_gauss
        ax2.clear()
        q_g = source_gauss(eta, p["q_max"], self.var_k_gauss.get(), p["l"])
        ax2.fill_between(eta, 0, q_g, alpha=0.4, color=plots.COLORS["accent3"])
        ax2.plot(eta, q_g, color=plots.COLORS["accent3"], lw=2)
        apply_mpl_style(ax2, f"Гауссов источник (k={self.var_k_gauss.get()})", "y, см", "q(y), Вт/см³")
        ax2.set_ylim(bottom=-p["q_max"]*0.05)
        ax2.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.6)
        for ax in [ax1, ax2]:
            ax.axhline(0.05 * p["q_max"], color=plots.COLORS["text2"], lw=0.8, ls=":", label="5% q_max")
            ax.legend(fontsize=FONT_PLOT_LABEL, facecolor=plots.COLORS["bg3"], labelcolor=plots.COLORS["text"], edgecolor=plots.COLORS["border"])
        self.canvas_source.draw()

    def _update_fourier_tab(self, *_):
        if self._T_field is None: return
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
        colors_bar = [plots.COLORS["accent"] if c >= 0 else plots.COLORS["accent4"] for c in coefs]
        ax1.bar(n_arr, coefs, color=colors_bar, alpha=0.85)
        apply_mpl_style(ax1, "Коэффициенты Фурье", "Номер гармоники n", "Амплитуда, °C/с")
        ax1.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.5, axis="y")
        ax2 = self.ax_fourier_conv
        ax2.clear()
        y_arr = np.linspace(0, l, 200)
        t_test = min(p["t_source_end"], p["t_max"] * 0.5)
        kw = dict(l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"], q_func=p["q_func"], t_source_end=p["t_source_end"], phi_func=p["phi_func"], q1_func=p["q1_func"], q2_func=p["q2_func"], N_terms=p["N_terms"])
        errors = []
        n_list = list(range(0, min(N_show*4, p["N_terms"]+1), 2))
        for N_cur in n_list:
            kw2 = dict(kw, N_terms=N_cur)
            T_cur = compute_temperature_detailed(y_arr, t_test, **kw2) + p["T_init"]
            T_full = self._T_field[int(np.argmin(np.abs(self._t_arr - t_test))), :]
            err = np.max(np.abs(T_cur - T_full)) if len(T_cur)==len(T_full) else 0
            errors.append(err)
        ax2.semilogy(n_list, [max(e, 0.001) for e in errors], color=plots.COLORS["accent3"], lw=2, marker="o", markersize=4)
        apply_mpl_style(ax2, f"Сходимость ряда (t={t_test:.1f}с)", "Число членов ряда N", "Макс. погрешность, °C")
        ax2.grid(True, color=plots.COLORS["border"], lw=0.5, alpha=0.5)
        self.canvas_fourier.draw()

    def _generate_extended_recommendations(self, p, res_current, res_rect, res_gauss):
        lines = []
        lines.append("=" * 60)
        lines.append("📊 РАСШИРЕННЫЙ АНАЛИЗ И РЕКОМЕНДАЦИИ")
        lines.append("=" * 60)
        lines.append(" ")
        
        lines.append("1. СРАВНЕНИЕ МОДЕЛЕЙ ИСТОЧНИКА")
        lines.append("-" * 40)
        tmax_r = res_rect["T_max"]
        tmax_g = res_gauss["T_max"]
        lines.append(f"Максимальная температура (Равномерный): {tmax_r:.0f} °C")
        lines.append(f"Максимальная температура (Гауссов):       {tmax_g:.0f} °C")
        if tmax_r > tmax_g:
            diff = ((tmax_r - tmax_g) / tmax_g) * 100
            lines.append(f"⚠️ Равномерная модель дает T_max на {diff:.1f}% выше.")
            lines.append("   Это связано с тем, что она концентрирует всю энергию на узком участке.")
            lines.append("   Гауссова модель физически точнее описывает реальные сварочные дуги/лазеры.")
        lines.append(" ")
        
        t85 = res_current["t85"]
        weld_width = res_current["weld_width"]
        T_max = res_current["T_max"]
        
        lines.append("2. АНАЛИЗ УСЛОВИЙ ЗАКАЛКИ (для выбранного режима)")
        lines.append("-" * 40)
        if t85 is not None:
            lines.append(f"Время охлаждения t₈/₅ = {t85:.2f} с")
            if t85 < 2:
                lines.append("⚠️ ОЧЕНЬ ВЫСОКАЯ СКОРОСТЬ ОХЛАЖДЕНИЯ!")
                lines.append("Риск образования хрупкого мартенсита и холодных трещин.")
                lines.append("✅ РЕКОМЕНДАЦИЯ:")
                lines.append("   - Увеличить предварительный подогрев пластины (T₀).")
                lines.append("   - Снизить скорость охлаждения (увеличить q_max или уменьшить k).")
                lines.append("   - Рассмотреть последующую термическую обработку (отпуск).")
            elif t85 < 10:
                lines.append("⚠️ Высокая скорость охлаждения. Возможно образование мартенсита и бейнита.")
                lines.append("✅ РЕКОМЕНДАЦИЯ: Для ответственных конструкций рекомендуется отпуск.")
            elif t85 < 30:
                lines.append("✅ БЛАГОПРИЯТНЫЙ РЕЖИМ.")
                lines.append("Ожидается структура бейнита. Хорошее сочетание прочности и вязкости.")
            elif t85 < 100:
                lines.append("ℹ️ Медленное охлаждение. Структура: перлит + феррит.")
                lines.append("✅ РЕКОМЕНДАЦИЯ: Если требуется высокая прочность, нужно ускорить охлаждение.")
            else:
                lines.append("⚠️ ОЧЕНЬ МЕДЛЕННОЕ ОХЛАЖДЕНИЕ.")
                lines.append("Риск роста зерна и снижения ударной вязкости.")
        else:
            lines.append("ℹ️ Параметр t₈/₅ не рассчитан (T_max < 800°C).")
            lines.append("Фазовые превращения в аустенитной области не произошли.")
        lines.append(" ")
        
        lines.append("3. ГЕОМЕТРИЯ СВАРНОГО СОЕДИНЕНИЯ")
        lines.append("-" * 40)
        lines.append(f"Ширина шва (зона плавления): {weld_width:.3f} см")
        lines.append(f"Ширина ЗТВ: {res_current['haz_width_text']}")
        if weld_width == 0:
            lines.append("⚠️ ПЛАВЛЕНИЕ НЕ ПРОИЗОШЛО!")
            lines.append("✅ РЕКОМЕНДАЦИЯ: Увеличить q_max или время действия источника.")
        lines.append(" ")
        
        lines.append("4. ТЕПЛОВОЕ ВОЗДЕЙСТВИЕ")
        lines.append("-" * 40)
        lines.append(f"Максимальная температура: {T_max:.0f} °C")
        if T_max > p["T_melt"] * 1.5:
            lines.append("⚠️ ПЕРЕГРЕВ! T_max значительно превышает T_пл.")
            lines.append("Возможно выгорание легирующих элементов, рост зерна.")
            lines.append("✅ РЕКОМЕНДАЦИЯ: Снизить q_max или увеличить k (для Гаусса).")
        elif T_max > p["T_melt"]:
            lines.append("✅ Нормальный режим сварки.")
        else:
            lines.append("ℹ️ Нагрев ниже температуры плавления (пайка/отпуск).")
        lines.append(" ")
        
        lines.append("5. ИНТЕРЕСНЫЕ СВЕДЕНИЯ")
        lines.append("-" * 40)
        lines.append("• Для меди и алюминия классическая закалка невозможна из-за")
        lines.append("  отсутствия фазовых превращений. Их ЗТВ определяется температурой")
        lines.append("  рекристаллизации (~300°C).")
        lines.append("• Параметр t₈/₅ (800->500°C) является стандартом только для")
        lines.append("  углеродистых сталей. Для титана и алюминиевых сплавов")
        lines.append("  используются другие критерии.")
        lines.append("• Высокие градиенты температур (|∇T|) ведут к возникновению")
        lines.append("  остаточных напряжений и деформаций пластины.")
        
        return "\n".join(lines)

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
                        if v_cool < -50: lines.append("⚠ Возможна закалка")
                        elif v_cool < -10: lines.append("✓ Нормализация")
                        else: lines.append("✓ Медленное охлаждение")
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
        self.var_src_type.set("Прямоугольный")
        self.var_display_mode.set("rect")
        self.var_material.set("Сталь 20")
        self.var_mat_source.set("builtin")
        self._on_src_type_selected()
        self._on_mat_source_changed()
        messagebox.showinfo("Сброс", "Параметры сброшены")

    def _export_data(self):
        if self._T_field is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчёт")
            return
        
        try:
            # Определяем путь к корню проекта (welding_sim/)
            # app.py лежит в gui/, поэтому поднимаемся на уровень выше
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            exports_dir = os.path.join(project_root, "exports")
            os.makedirs(exports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f"export_{timestamp}"
            full_export_path = os.path.join(exports_dir, folder_name)
            os.makedirs(full_export_path, exist_ok=True)
            
            mode = self.var_display_mode.get()
            res = self._results[mode]
            p = self._results["params"].copy()
            src_type_val = "rect" if self.var_src_type.get() == "Прямоугольный" else "gauss"
            p["src_type"] = src_type_val
            
            mat_name = p.get('material', 'N/A')
            density = materials.MATERIALS_DB.get(mat_name, {}).get('density', 'N/A')
            
            # 1. Save parameters.txt
            with open(os.path.join(full_export_path, "parameters.txt"), "w", encoding="utf-8") as f:
                f.write("WELDING SIMULATION PARAMETERS\n")
                f.write("="*40 + "\n")
                f.write(f"Material: {mat_name}\n")
                f.write(f"Plate length (l): {p['l']} cm\n")
                f.write(f"Thermal diffusivity (a): {p['a']} cm^2/s\n")
                f.write(f"Volumetric heat capacity (c_V): {p['c_V']} J/(cm^3*C)\n")
                f.write(f"Thermal conductivity (lam): {p['lam']} W/(cm*C)\n")
                f.write(f"Density (rho): {density} g/cm^3\n")
                f.write(f"Melting temp (T_melt): {p['T_melt']} C\n")
                f.write(f"Quenching temp (T_harden): {p['T_harden']} C\n")
                f.write(f"Initial temp (T_init): {p['T_init']} C\n")
                f.write("-"*40 + "\n")
                f.write(f"Source type: {p['src_type']}\n")
                f.write(f"Max power (q_max): {p['q_max']} W/cm^3\n")
                f.write(f"Source time (t_source): {p['t_source_end']} s\n")
                if p['src_type'] == 'rect':
                    f.write(f"Source y1: {self.var_y1.get()} cm\n")
                    f.write(f"Source y2: {self.var_y2.get()} cm\n")
                else:
                    f.write(f"Gauss coefficient (k): {self.var_k_gauss.get()} 1/cm^2\n")
                f.write("-"*40 + "\n")
                f.write(f"Boundary flux q1: {p['q1_val']} W/cm^2\n")
                f.write(f"Boundary flux q2: {p['q2_val']} W/cm^2\n")
                f.write("-"*40 + "\n")
                f.write(f"Max time (t_max): {p['t_max']} s\n")
                f.write(f"Time steps (N_t): {p['N_t']}\n")
                f.write(f"Fourier terms (N_terms): {p['N_terms']}\n")
                f.write(f"Observation point (y_A): {p['y_A']} cm\n")

            # 2. Save recommendations.txt
            recs = self.rec_text_widget.get("1.0", "end").strip()
            with open(os.path.join(full_export_path, "recommendations.txt"), "w", encoding="utf-8") as f:
                f.write(recs)
                
            # 3. Save thermal_cycle.csv (English headers, ; delimiter for Excel)
            dTdt = np.gradient(res["T_cycle"], res["t_arr"])
            with open(os.path.join(full_export_path, "thermal_cycle.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Time t, s", "Temperature T, C", "Cooling rate dT/dt, C/s"])
                for t, T, rate in zip(res["t_arr"], res["T_cycle"], dTdt):
                    writer.writerow([f"{t:.4f}", f"{T:.4f}", f"{rate:.4f}"])
                    
            # 4. Save all plots as PNG
            plots_to_save = [
                ("fig_cycle", "thermal_cycle.png"),
                ("fig_cooling", "cooling_rate.png"),
                ("fig_source", "sources.png"),
                ("fig_fourier", "fourier.png"),
                ("fig_heatmap", "heatmap.png"),
                ("fig_cct", "metallurgy.png"),
                ("fig_physical", "physical.png"),
                ("fig_components", "components.png"),
                ("fig_propagation", "propagation.png"),
                ("fig_length", "length.png"),
                ("fig_phases", "phases.png"),
                ("fig_3d", "3d_surface.png")
            ]
            
            for fig_attr, filename in plots_to_save:
                if hasattr(self, fig_attr):
                    fig = getattr(self, fig_attr)
                    try:
                        fig.canvas.draw()
                        fig.savefig(os.path.join(full_export_path, filename), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
                    except Exception as e:
                        print(f"Error saving {filename}: {e}")
                        
            messagebox.showinfo("Экспорт", f"Данные успешно экспортированы в папку:\n{os.path.abspath(full_export_path)}")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Ошибка экспорта", str(e))