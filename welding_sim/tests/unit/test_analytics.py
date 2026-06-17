"""
tests/unit/test_analytics.py
Модульное тестирование модуля аналитики.
Проверяет расчёт метрик (t85, ширина шва, ЗТВ, структура) на граничных условиях.
"""
import pytest
import numpy as np
from core.analytics import (
    compute_t85, compute_weld_width, compute_haz_width, predict_structure
)

class TestT85Calculation:
    """Тесты расчёта времени охлаждения t₈/₅"""
    
    def test_t85_none_for_low_temperature(self):
        """Если максимальная температура < 800°C, t85 должен быть None"""
        t_arr = np.linspace(0, 60, 120)
        T_cycle = np.full(120, 500.0)  # Постоянная температура 500°C
        assert compute_t85(t_arr, T_cycle) is None

    def test_t85_valid_cycle(self):
        """Для реалистичного термического цикла t85 должен быть положительным и конечным"""
        t_arr = np.linspace(0, 60, 200)
        # Синтетический цикл: быстрый нагрев до 1500, медленное остывание до 20
        T_cycle = np.concatenate([
            np.linspace(20, 1500, 50),
            np.linspace(1500, 20, 150)
        ])
        t85 = compute_t85(t_arr, T_cycle)
        
        assert t85 is not None
        assert t85 > 0
        assert t85 < 60.0  # Не может превышать общее время моделирования

class TestWeldWidth:
    """Тесты определения ширины сварного шва"""
    
    def test_weld_width_no_melting(self):
        """Если температура нигде не достигает T_пл, ширина шва = 0.0"""
        y_arr = np.linspace(0, 10, 100)
        T_field = np.full((10, 100), 1000.0)  # Максимум 1000°C
        T_melt = 1510.0
        
        width = compute_weld_width(y_arr, T_field, T_melt)
        assert width == 0.0

    def test_weld_width_with_melting(self):
        """Ширина шва должна корректно определяться по изотерме"""
        y_arr = np.linspace(0, 10, 100) # Шаг 0.1 см
        T_field = np.zeros((10, 100))
        # Создаём зону плавления от y=4.0 до y=6.0 (индексы 40..60)
        T_field[:, 40:61] = 1600.0  
        T_melt = 1510.0
        
        width = compute_weld_width(y_arr, T_field, T_melt)
        assert width > 0.0
        # Ожидаемая ширина: 6.0 - 4.0 = 2.0 см (с погрешностью дискретизации)
        assert abs(width - 2.0) < 0.15

class TestHAZWidth:
    """Тесты определения ширины зоны термического влияния (ЗТВ)"""
    
    def test_haz_width_no_hardening_steel(self):
        """Для материала без закалки (T_harden <= 0), не являющегося Al/Cu, ЗТВ = N/A"""
        y_arr = np.linspace(0, 10, 100)
        T_field = np.full((10, 100), 1000.0)
        T_harden = 0.0
        
        width, text = compute_haz_width(y_arr, T_field, T_harden, material_name="Неизвестный сплав")
        assert width == 0.0
        assert "N/A" in text

    def test_haz_width_aluminum_recristallization(self):
        """Для алюминия ЗТВ определяется по температуре рекристаллизации (~300°C)"""
        y_arr = np.linspace(0, 10, 100)
        T_field = np.zeros((10, 100))
        T_field[:, 30:71] = 400.0  # Выше 300°C
        T_harden = 0.0
        
        width, text = compute_haz_width(y_arr, T_field, T_harden, material_name="Алюминий АМг5")
        assert width > 0.0
        assert "T_рекр" in text

class TestStructurePrediction:
    """Тесты прогнозирования микроструктуры по параметру t₈/₅"""
    
    @pytest.mark.parametrize("t85_val, expected_keyword, expected_color", [
        (1.5, "Мартенсит", "red"),
        (5.0, "Мартенсит", "orange"), # Мартенсит + Бейнит
        (20.0, "Бейнит", "yellow"),
        (50.0, "Перлит", "green"),    # Перлит + Феррит
        (150.0, "Грубый феррит", "lightgreen"),
    ])
    def test_structure_prediction_ranges(self, t85_val, expected_keyword, expected_color):
        """Проверка всех диапазонов классификации структуры"""
        structure, color = predict_structure(t85=t85_val, T_max=1600, T_melt=1510)
        assert expected_keyword in structure
        assert color == expected_color

    def test_structure_no_melting(self):
        """Если T_max < T_melt, структура должна быть 'Без плавления'"""
        structure, color = predict_structure(t85=10, T_max=1000, T_melt=1510)
        assert "Без плавления" in structure
        assert color == "blue"

    def test_structure_insufficient_data(self):
        """Если t85 = None, но плавление было, данных недостаточно для классификации"""
        structure, color = predict_structure(t85=None, T_max=1600, T_melt=1510)
        assert "Недостаточно" in structure