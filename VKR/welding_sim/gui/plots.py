"""
gui/plots.py
Константы стилей и вспомогательные функции для визуализации
"""
import tkinter as tk

# ==============================================================================
# 1. НАСТРОЙКИ ТЕМ, ЦВЕТОВ И ШРИФТОВ
# ==============================================================================
FONT_UI = ("Consolas", 11)
FONT_UI_BOLD = ("Consolas", 11, "bold")
FONT_PLOT_LABEL = 10
FONT_PLOT_TITLE = 12

COLORS_DARK = {
    "bg": "#0f1117", "bg2": "#1a1d27", "bg3": "#232736",
    "border": "#2e3347", "accent": "#4f8ef7", "accent2": "#f7954f",
    "accent3": "#4ff7a0", "accent4": "#f74f7a", "text": "#e8ecf4",
    "text2": "#8a93aa", "warning": "#f7c14f", "melt": "#ff4444",
    "header": "#1e2235",
}

COLORS_LIGHT = {
    "bg": "#f0f2f5", "bg2": "#ffffff", "bg3": "#e1e4e8",
    "border": "#d1d5da", "accent": "#0366d6", "accent2": "#d07020",
    "accent3": "#28a745", "accent4": "#d73a49", "text": "#24292e",
    "text2": "#586069", "warning": "#b08800", "melt": "#cb2431",
    "header": "#ffffff",
}

CURRENT_THEME = "dark"
COLORS = COLORS_DARK
CMAP_HEAT = "plasma"

def styled_entry(parent, textvariable, width=12):
    return tk.Entry(parent, textvariable=textvariable, width=width, font=FONT_UI,
                    bg=COLORS["bg3"], fg=COLORS["text"], insertbackground=COLORS["accent"],
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=COLORS["border"], highlightcolor=COLORS["accent"])

def styled_button(parent, text, command, color=None, **kw):
    bg = color or COLORS["accent"]
    return tk.Button(parent, text=text, command=command, font=FONT_UI_BOLD,
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
        tk.Label(hdr, text=f"  {title} ", font=FONT_UI_BOLD,
                 fg=COLORS["accent"], bg=COLORS["header"], pady=6).pack(side="left")
    inner = tk.Frame(outer, bg=COLORS["bg2"])
    inner.pack(fill="both", expand=True, padx=12, pady=8)
    return outer, inner

def apply_mpl_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS["bg2"])
    ax.tick_params(colors=COLORS["text2"], labelsize=FONT_PLOT_LABEL)
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    ax.set_xlabel(xlabel, color=COLORS["text2"], fontsize=FONT_PLOT_LABEL+1)
    ax.set_ylabel(ylabel, color=COLORS["text2"], fontsize=FONT_PLOT_LABEL+1)
    ax.set_title(title, color=COLORS["text"], fontsize=FONT_PLOT_TITLE, fontweight="bold", pad=10)

def normalize_color(widget, color):
    try:
        r, g, b = widget.winfo_rgb(color)
        return f"#{r>>8:02x}{g>>8:02x}{b>>8:02x}"
    except tk.TclError:
        return color.lower() if isinstance(color, str) else color

def map_color(widget, color_str):
    if not color_str:
        return color_str
    norm = normalize_color(widget, color_str)
    old_theme = COLORS_DARK if CURRENT_THEME == "light" else COLORS_LIGHT
    new_theme = COLORS_LIGHT if CURRENT_THEME == "light" else COLORS_DARK
    for key, val in old_theme.items():
        if normalize_color(widget, val) == norm:
            return new_theme[key]
    return color_str

def apply_theme_recursive(widget):
    try:
        color_props = ["bg", "fg", "highlightbackground", "highlightcolor", 
                       "insertbackground", "activebackground", "activeforeground", 
                       "selectcolor", "troughcolor"]
        for prop in color_props:
            try:
                if prop in widget.keys():
                    old_val = widget.cget(prop)
                    new_val = map_color(widget, old_val)
                    widget.configure(**{prop: new_val})
            except tk.TclError:
                pass
        for child in widget.winfo_children():
             apply_theme_recursive(child)
    except tk.TclError:
        pass