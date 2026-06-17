import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Настройка стиля
plt.style.use('default')
fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_aspect('equal')

# Цвета
COLOR_PRESENTATION = '#E3F2FD'  # светло-голубой
COLOR_BUSINESS = '#FFF3E0'      # светло-оранжевый
COLOR_DATA = '#E8F5E9'          # светло-зелёный
COLOR_CROSS = '#F3E5F5'         # светло-фиолетовый
BORDER_COLOR = '#1565C0'
TEXT_COLOR = '#0D47A1'

def create_rounded_rect(x, y, width, height, color, label, sublabels=None):
    """Создаёт прямоугольник с закруглёнными углами"""
    rect = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        linewidth=2,
        edgecolor=BORDER_COLOR,
        facecolor=color,
        zorder=1
    )
    ax.add_patch(rect)
    
    # Заголовок слоя
    ax.text(x + width/2, y + height - 0.4, label, 
            ha='center', va='top', 
            fontsize=12, fontweight='bold', 
            color=TEXT_COLOR)
    
    # Подписи компонентов
    if sublabels:
        for i, sublabel in enumerate(sublabels):
            ax.text(x + width/2, y + height - 1.2 - i*0.7, 
                   sublabel, ha='center', va='top',
                   fontsize=10, color=TEXT_COLOR,
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', 
                           edgecolor=BORDER_COLOR, 
                           linewidth=1.5))

# ============ СЛОЙ ПРЕДСТАВЛЕНИЯ ============
create_rounded_rect(1, 8.5, 7, 3, COLOR_PRESENTATION, 
                   "Слой представления (GUI)",
                   ["Компоненты интерфейса (tkinter)",
                    "Визуализация графиков (matplotlib)",
                    "Управление параметрами (ползунки, кнопки)"])

# ============ БИЗНЕС-СЛОЙ ============
create_rounded_rect(1, 5, 7, 3, COLOR_BUSINESS,
                   "Бизнес-слой (Ядро расчёта)",
                   ["Расчёт температурного поля (КИП)",
                    "Анализ метрик (ширина шва, t₈/₅)",
                    "Работа с материалами (JSON)"])

# ============ СЛОЙ ДОСТУПА К ДАННЫМ ============
create_rounded_rect(1, 2, 7, 2.5, COLOR_DATA,
                   "Слой данных",
                   ["База материалов (JSON файлы)",
                    "Экспорт результатов (CSV, PNG, TXT)"])

# ============ СКВОЗНАЯ ФУНКЦИОНАЛЬНОСТЬ ============
cross_func = FancyBboxPatch(
    (8.5, 2), 3, 9.5,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    linewidth=2,
    edgecolor=BORDER_COLOR,
    facecolor=COLOR_CROSS,
    zorder=1
)
ax.add_patch(cross_func)

ax.text(10, 11, "Сквозная функциональность", 
       ha='center', va='top',
       fontsize=12, fontweight='bold', 
       color=TEXT_COLOR, rotation=90)

cross_labels = ["Тестирование (pytest)",
               "Векторизация (NumPy)",
               "Обработка ошибок"]

for i, label in enumerate(cross_labels):
    ax.text(10, 9.5 - i*1.2, label,
           ha='center', va='top',
           fontsize=10, color=TEXT_COLOR,
           rotation=90,
           bbox=dict(boxstyle="round,pad=0.3",
                    facecolor='white',
                    edgecolor=BORDER_COLOR,
                    linewidth=1.5))

# ============ СТРЕЛКИ СВЯЗИ ============
# Стрелка между слоями
arrow_props = dict(arrowstyle='->', color=BORDER_COLOR, 
                  linewidth=2, mutation_scale=20)

# Между GUI и ядром
ax.annotate('', xy=(4.5, 8.3), xytext=(4.5, 8.7),
           arrowprops=arrow_props)
ax.annotate('', xy=(4.5, 8.7), xytext=(4.5, 8.3),
           arrowprops=arrow_props)

# Между ядром и данными
ax.annotate('', xy=(4.5, 4.8), xytext=(4.5, 5.2),
           arrowprops=arrow_props)
ax.annotate('', xy=(4.5, 5.2), xytext=(4.5, 4.8),
           arrowprops=arrow_props)

# Подпись к стрелкам
ax.text(5, 8.5, "Запрос расчёта", fontsize=9, 
       color=TEXT_COLOR, ha='left', va='center')
ax.text(5, 7.8, "Результаты", fontsize=9, 
       color=TEXT_COLOR, ha='left', va='center')

ax.text(5, 5, "Чтение/запись", fontsize=9, 
       color=TEXT_COLOR, ha='left', va='center')
ax.text(5, 4.5, "материалов", fontsize=9, 
       color=TEXT_COLOR, ha='left', va='center')

# ============ ЗАГОЛОВОК ============
ax.text(6, 11.5, "Трёхуровневая архитектура приложения\nмоделирования теплового процесса сварки",
       ha='center', va='top',
       fontsize=14, fontweight='bold',
       color='#0D47A1')

# ============ ДОПОЛНИТЕЛЬНЫЕ ЭЛЕМЕНТЫ ============
# Блок "Пользователь"
user_rect = patches.Rectangle((2.5, 11.2), 2, 0.5,
                             facecolor='#FFECB3',
                             edgecolor='#FF6F00',
                             linewidth=2)
ax.add_patch(user_rect)
ax.text(3.5, 11.45, "Пользователь",
       ha='center', va='center',
       fontsize=10, fontweight='bold',
       color='#E65100')

# Стрелка от пользователя к GUI
ax.annotate('', xy=(3.5, 11.1), xytext=(3.5, 11.7),
           arrowprops=dict(arrowstyle='->', 
                          color='#FF6F00',
                          linewidth=2))

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('architecture_diagram.pdf', bbox_inches='tight')
plt.show()

print("Схема архитектуры сохранена:")
print("  - architecture_diagram.png (растр, 300 dpi)")
print("  - architecture_diagram.pdf (вектор)")