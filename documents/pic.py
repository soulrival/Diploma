import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи (из твоей программы)
l = 10.0          # ширина пластины, см
q_max = 10000.0   # максимальная мощность, Вт/см³
y1, y2 = 4.5, 5.5 # границы равномерного источника
k = 5.0           # коэффициент для гауссова источника

# Сетка по координате
y = np.linspace(0, l, 600)

# Функции источников (взяты прямо из core/math_engine.py)
def source_uniform(y, q_max, y1, y2):
    """Равномерное (кусочно-постоянное) распределение мощности"""
    return np.where((y > y1) & (y <= y2), q_max, 0.0)

def source_gauss(y, q_max, k, l):
    """Гауссово (нормальное) распределение мощности"""
    return q_max * np.exp(-k * (y - l / 2) ** 2)

q_uniform = source_uniform(y, q_max, y1, y2)
q_gauss = source_gauss(y, q_max, k, l)

# Настройка стиля для диплома (ГОСТ-подобный)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Создание рисунка с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# а) Равномерное распределение
ax1.fill_between(y, 0, q_uniform, color='#4f8ef7', alpha=0.3, label='Область действия источника')
ax1.plot(y, q_uniform, color='#4f8ef7', lw=2.5, label='q(y)')
ax1.axvline(y1, color='black', lw=1, ls='--', alpha=0.7)
ax1.axvline(y2, color='black', lw=1, ls='--', alpha=0.7)
ax1.annotate(f'$y_1={y1}$', xy=(y1, q_max), xytext=(y1-0.8, q_max+800),
             fontsize=11, ha='center')
ax1.annotate(f'$y_2={y2}$', xy=(y2, q_max), xytext=(y2+0.8, q_max+800),
             fontsize=11, ha='center')
ax1.set_xlabel('y, см')
ax1.set_ylabel('q(y), Вт/см³')
ax1.set_title('а) Равномерное распределение')
ax1.set_xlim(0, l)
ax1.set_ylim(-500, q_max * 1.15)
ax1.legend(loc='upper right')

# б) Гауссово распределение
ax2.fill_between(y, 0, q_gauss, color='#f7954f', alpha=0.3, label='Область действия источника')
ax2.plot(y, q_gauss, color='#f7954f', lw=2.5, label='q(y)')
ax2.axvline(l/2, color='black', lw=1, ls='--', alpha=0.7)
ax2.annotate(f'центр: $y={l/2}$', xy=(l/2, q_max), xytext=(l/2+1.5, q_max-1500),
             fontsize=11, ha='center',
             arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax2.set_xlabel('y, см')
ax2.set_ylabel('q(y), Вт/см³')
ax2.set_title(f'б) Гауссово распределение ($k={k}$ 1/см²)')
ax2.set_xlim(0, l)
ax2.set_ylim(-500, q_max * 1.15)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('sources_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('sources_distribution.pdf', bbox_inches='tight')  # векторный формат для Word
plt.show()

print("Графики сохранены:")
print("  - sources_distribution.png (растр, 300 dpi)")
print("  - sources_distribution.pdf (вектор, для Word)")