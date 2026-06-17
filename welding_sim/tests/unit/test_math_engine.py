"""
tests/unit/test_math_engine.py
Модульное тестирование математического ядра (White-box testing).
Проверяет корректность вычислений, граничные условия и сходимость ряда Фурье.
"""
import pytest
import numpy as np
from core.math_engine import (
    mu_n, beta_n, source_rect, source_gauss,
    compute_temperature_detailed, compute_spatial_field
)

class TestBasicMathFunctions:
    """Тесты базовых математических функций преобразования"""
    
    def test_mu_n_calculation(self):
        """Проверка расчёта собственных чисел μ_n = π * n / l"""
        assert mu_n(0, 10.0) == 0.0
        assert np.isclose(mu_n(1, 10.0), np.pi / 10.0)
        assert np.isclose(mu_n(2, 10.0), 2 * np.pi / 10.0)
    
    def test_beta_n_calculation(self):
        """Проверка нормирующих коэффициентов β_n"""
        assert beta_n(0) == 0.5
        assert beta_n(1) == 1.0
        assert beta_n(10) == 1.0


class TestSourceModels:
    """Тесты моделей распределения мощности источника тепла"""
    
    def test_source_rect_boundaries(self):
        """Прямоугольный источник: q = q_max внутри (y1, y2], 0 снаружи"""
        # Условие в реализации: (eta > y1) & (eta <= y2)
        # Левая граница НЕ включается, правая включается
        eta = np.array([4.0, 4.5, 4.6, 5.0, 5.5, 6.0])
        q = source_rect(eta, q_max=1000.0, y1=4.5, y2=5.5)
        
        assert q[0] == 0.0      # y < y1 (снаружи)
        assert q[1] == 0.0      # y == y1 (НЕ включается по условию >)
        assert q[2] == 1000.0   # y = 4.6 (внутри)
        assert q[3] == 1000.0   # y = 5.0 (внутри)
        assert q[4] == 1000.0   # y == y2 (включается по условию <=)
        assert q[5] == 0.0      # y > y2 (снаружи)

    def test_source_gauss_peak_and_symmetry(self):
        """Гауссов источник: пик в центре, симметричное убывание"""
        l = 10.0
        eta = np.linspace(0, l, 1001)  # Нечётное количество для точного центра
        q = source_gauss(eta, q_max=1000.0, k=5.0, l=l)
        
        # 1. Пик должен быть в центре (l/2)
        peak_idx = np.argmax(q)
        assert np.isclose(eta[peak_idx], l / 2.0, atol=0.02)
        
        # 2. Значение в пике должно быть равно q_max
        assert np.isclose(q[peak_idx], 1000.0, rtol=1e-3)
        
        # 3. Симметрия: используем действительно симметричные точки относительно центра
        center_idx = len(eta) // 2  # = 500, соответствует eta=5.0
        # Берём точки на одинаковом расстоянии от центра
        offset = 50
        left_val = q[center_idx - offset]
        right_val = q[center_idx + offset]
        # Проверяем симметрию с разумным допуском
        assert np.isclose(left_val, right_val, rtol=1e-2)


class TestTemperatureSolver:
    """Тесты решателя задачи теплопроводности"""
    
    def test_initial_condition_preservation(self, default_steel_params):
        """
        Физическая проверка: при t -> 0 температура должна стремиться к T_init.
        """
        y_points = np.array([2.0, 5.0, 8.0])
        t_val = 0.001  # Очень малое время
        params = default_steel_params
        
        q_func = lambda eta: source_rect(eta, params["q_max"], 4.5, 5.5)
        phi_func = lambda y: np.full_like(y, params["T_init"])
        
        T = compute_temperature_detailed(
            y_points, t_val, 
            l=params["l"], a=params["a"], c_V=params["c_V"], lam=params["lam"],
            q_func=q_func, t_source_end=params["t_source_end"],
            phi_func=phi_func, N_terms=params["N_terms"], 
            N_eta=params["N_eta"], N_tau=params["N_tau"]
        )
        
        # Температура должна быть близка к начальной (с допуском на численную погрешность)
        assert np.allclose(T, params["T_init"], atol=5.0)

    def test_physical_correctness_no_negative_temp(self, default_steel_params, spatial_grids):
        """
        Физическая проверка: температура не может быть ниже начальной (если T_init >= 0),
        так как источник только нагревает, а граничные условия изолирующие.
        """
        t_arr, y_arr = spatial_grids
        params = default_steel_params
        q_func = lambda eta: source_gauss(eta, params["q_max"], 5.0, params["l"])
        phi_func = lambda y: np.full_like(y, params["T_init"])
        
        T_field = compute_spatial_field(
            y_arr, t_arr,
            l=params["l"], a=params["a"], c_V=params["c_V"], lam=params["lam"],
            q_func=q_func, t_source_end=params["t_source_end"],
            phi_func=phi_func, N_terms=params["N_terms"]
        )
        
        # Минимальная температура во всём поле не должна быть ниже T_init (с небольшим допуском)
        assert np.min(T_field) >= params["T_init"] - 1.0

    def test_fourier_series_convergence(self, default_steel_params):
        """
        Математическая проверка: решение должно сходиться при увеличении числа мод N_terms.
        Разница между N=60 и N=80 должна быть меньше, чем между N=20 и N=40.
        """
        y_points = np.array([5.0]) # Центр пластины
        t_val = 5.0
        params = default_steel_params
        q_func = lambda eta: source_rect(eta, params["q_max"], 4.5, 5.5)
        
        # Базовые параметры для вызова
        base_kwargs = {
            "l": params["l"], "a": params["a"], "c_V": params["c_V"], "lam": params["lam"],
            "q_func": q_func, "t_source_end": params["t_source_end"],
            "N_eta": params["N_eta"], "N_tau": params["N_tau"]
        }
        
        T_20 = compute_temperature_detailed(y_points, t_val, N_terms=20, **base_kwargs)[0]
        T_40 = compute_temperature_detailed(y_points, t_val, N_terms=40, **base_kwargs)[0]
        T_80 = compute_temperature_detailed(y_points, t_val, N_terms=80, **base_kwargs)[0]
        
        diff_20_40 = abs(T_40 - T_20)
        diff_40_80 = abs(T_80 - T_40)
        
        # Ожидаем, что при увеличении N погрешность меняется всё меньше (сходимость)
        assert diff_40_80 < diff_20_40, "Ряд Фурье не демонстрирует ожидаемой сходимости"

    def test_spatial_field_shape(self, default_steel_params, spatial_grids):
        """Проверка размерности выходного массива пространственно-временного поля"""
        t_arr, y_arr = spatial_grids
        params = default_steel_params
        q_func = lambda eta: source_rect(eta, params["q_max"], 4.5, 5.5)
        
        T_field = compute_spatial_field(
            y_arr, t_arr,
            l=params["l"], a=params["a"], c_V=params["c_V"], lam=params["lam"],
            q_func=q_func, t_source_end=params["t_source_end"], N_terms=params["N_terms"]
        )
        
        assert T_field.shape == (len(t_arr), len(y_arr))