"""
tests/integration/test_simulation_pipeline.py
Интеграционное тестирование (Black-box testing).
Проверяет корректность работы всего конвейера: от задания параметров 
до расчёта температурного поля и извлечения физических метрик.
"""
import pytest
import numpy as np
from core.math_engine import compute_spatial_field, source_rect, source_gauss
from core.analytics import compute_t85, compute_weld_width, compute_haz_width, predict_structure
from core import materials


class TestFullPipelineScenarios:
    """Тесты полного цикла моделирования для различных сценариев"""

    def test_pipeline_steel_rectangular_source(self, default_steel_params, spatial_grids):
        """
        Сценарий 1: Стандартная сварка стали 20 прямоугольным источником.
        Ожидаемый результат: Плавление происходит, ширина шва > 0, t85 рассчитывается.
        
        Примечание: В 1D-модели поперечного сечения (ширина пластины L=10 см) 
        чрезмерная мощность источника приводит к нереалистичному перегреву (>5000°C), 
        так как отсутствует отвод тепла вглубь и вдоль шва (3D-эффекты). 
        Для корректной проверки алгоритма t85 используются физически обоснованные 
        параметры: умеренная мощность (3000 Вт/см³) и естественное остывание 
        за счёт теплопроводности в пределах 100 секунд.
        """
        _, y_arr = spatial_grids
        p = default_steel_params.copy()
        
        # 1. Физически корректная мощность для 1D-модели поперечного сечения
        p["q_max"] = 3000.0  # Вт/см³ (вместо 10000, чтобы избежать перегрева >5000°C)
        
        # 2. Время моделирования 100 секунд (достаточно для остывания 10 см стали)
        t_arr = np.linspace(0.1, 100.0, 200)
        
        # 3. Граничные условия: теплоизоляция (тепло рассеивается внутри области)
        p["q1_val"] = 0.0
        p["q2_val"] = 0.0
        
        q_func = lambda eta: source_rect(eta, p["q_max"], 4.5, 5.5)
        phi_func = lambda y: np.full_like(y, p["T_init"])
        q1_func = lambda tau: np.full_like(tau, p["q1_val"])
        q2_func = lambda tau: np.full_like(tau, p["q2_val"])

        # 4. Запуск решателя (ядро)
        T_field = compute_spatial_field(
            y_arr, t_arr,
            l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"],
            q_func=q_func, t_source_end=p["t_source_end"],
            phi_func=phi_func, q1_func=q1_func, q2_func=q2_func, N_terms=p["N_terms"]
        ) + p["T_init"]

        # 5. Запуск аналитики (постобработка)
        T_max_global = np.max(T_field)
        
        # Контрольная точка: центр пластины (y = 5.0 см, индекс 50)
        idx_monitor = 50
        t85 = compute_t85(t_arr, T_field[:, idx_monitor])
        
        weld_w = compute_weld_width(y_arr, T_field, p["T_melt"])
        haz_w, haz_text = compute_haz_width(y_arr, T_field, p["T_harden"], "Сталь 20")
        
        T_max_at_point = np.max(T_field[:, idx_monitor])
        structure, color = predict_structure(t85, T_max_at_point, p["T_melt"])

        # 6. Физические проверки (Assertions)
        assert T_max_global > p["T_melt"], "Глобальная температура должна превысить температуру плавления"
        assert T_max_global < 3000.0, "Температура не должна быть нереалистично высокой (признак некорректных параметров для 1D)"
        assert weld_w > 0.0, "При превышении T_melt ширина шва должна быть больше 0"
        assert t85 is not None, f"Параметр t85 должен быть рассчитан (Макс. темп. в точке: {T_max_at_point:.1f}°C)"
        assert 1.0 < t85 < 150.0, f"Значение t85={t85} выходит за физические пределы для стали"
        assert haz_w > weld_w, "Зона термического влияния должна быть шире самого шва"
        assert any(s in structure for s in ["Мартенсит", "Бейнит", "Перлит"]), f"Некорректный прогноз структуры: {structure}"

    def test_pipeline_no_melting_low_power(self, default_steel_params, spatial_grids):
        """
        Сценарий 2: Недостаточная мощность источника (нагрев без плавления).
        Ожидаемый результат: Плавления нет, ширина шва = 0, t85 = None.
        """
        t_arr, y_arr = spatial_grids
        p = default_steel_params.copy()
        p["q_max"] = 500.0  # Занижаем мощность в 20 раз
        
        q_func = lambda eta: source_rect(eta, p["q_max"], 4.5, 5.5)
        phi_func = lambda y: np.full_like(y, p["T_init"])

        T_field = compute_spatial_field(
            y_arr, t_arr,
            l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"],
            q_func=q_func, t_source_end=p["t_source_end"],
            phi_func=phi_func, N_terms=p["N_terms"]
        ) + p["T_init"]

        T_max = np.max(T_field)
        weld_w = compute_weld_width(y_arr, T_field, p["T_melt"])
        t85 = compute_t85(t_arr, T_field[:, 50])

        # Проверки
        assert T_max < p["T_melt"], "При низкой мощности плавления быть не должно"
        assert weld_w == 0.0, "Ширина шва должна быть строго 0.0"
        assert t85 is None, "Параметр t85 не должен рассчитываться, если T_max < 800°C"

    def test_pipeline_aluminum_recristallization(self, default_steel_params, spatial_grids):
        """
        Сценарий 3: Сварка алюминия (материал без закалки).
        Ожидаемый результат: ЗТВ определяется по температуре рекристаллизации, 
        а не по T_harden.
        """
        t_arr, y_arr = spatial_grids
        p = default_steel_params.copy()
        
        # Переопределяем параметры под Алюминий АМг5
        p["T_melt"] = 660.0
        p["T_harden"] = 0.0
        p["a"] = 0.8
        p["c_V"] = 2.4
        p["lam"] = 1.2
        p["q_max"] = 8000.0  # Адаптированная мощность

        q_func = lambda eta: source_gauss(eta, p["q_max"], 5.0, p["l"])
        phi_func = lambda y: np.full_like(y, p["T_init"])

        T_field = compute_spatial_field(
            y_arr, t_arr,
            l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"],
            q_func=q_func, t_source_end=p["t_source_end"],
            phi_func=phi_func, N_terms=p["N_terms"]
        ) + p["T_init"]

        T_max = np.max(T_field)
        haz_w, haz_text = compute_haz_width(y_arr, T_field, p["T_harden"], "Алюминий АМг5")

        # Проверки
        assert T_max > p["T_melt"], "Алюминий должен расплавиться"
        assert haz_w > 0.0, "ЗТВ должна быть определена"
        assert "T_рекр" in haz_text, "Для алюминия ЗТВ должна маркироваться по температуре рекристаллизации"

    def test_pipeline_gaussian_vs_rectangular_gradients(self, default_steel_params, spatial_grids):
        """
        Сценарий 4: Сравнение градиентов температур.
        Гауссов источник должен давать более плавный профиль (меньший максимальный градиент), 
        чем прямоугольный при схожей общей энергии.
        """
        t_arr, y_arr = spatial_grids
        p = default_steel_params.copy()
        p["N_terms"] = 60 # Немного увеличим для точности градиентов
        
        # Прямоугольный источник
        q_rect = lambda eta: source_rect(eta, p["q_max"], 4.5, 5.5)
        T_rect = compute_spatial_field(
            y_arr, t_arr, l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"],
            q_func=q_rect, t_source_end=p["t_source_end"], N_terms=p["N_terms"]
        ) + p["T_init"]
        grad_rect = np.max(np.abs(np.gradient(T_rect, axis=1)))

        # Гауссов источник (схожая интегральная мощность)
        q_gauss = lambda eta: source_gauss(eta, p["q_max"], 5.0, p["l"])
        T_gauss = compute_spatial_field(
            y_arr, t_arr, l=p["l"], a=p["a"], c_V=p["c_V"], lam=p["lam"],
            q_func=q_gauss, t_source_end=p["t_source_end"], N_terms=p["N_terms"]
        ) + p["T_init"]
        grad_gauss = np.max(np.abs(np.gradient(T_gauss, axis=1)))

        # Проверка: градиент у Гаусса должен быть меньше (процесс более плавный)
        assert grad_gauss < grad_rect, "Гауссов источник должен обеспечивать меньшие градиенты температур"


class TestDataConsistency:
    """Тесты согласованности данных между модулями"""

    def test_materials_db_consistency(self):
        """
        Проверка того, что все материалы в базе имеют корректные типы данных 
        и логически непротиворечивые значения (например, T_melt > T_harden).
        """
        for mat_name, props in materials.MATERIALS_DB.items():
            assert props["T_melt"] > 0, f"Температура плавления {mat_name} некорректна"
            assert props["a"] > 0, f"Температуропроводность {mat_name} некорректна"
            assert props["c_V"] > 0, f"Теплоемкость {mat_name} некорректна"
            
            # Для сталей температура закалки должна быть ниже температуры плавления
            if props["T_harden"] > 0:
                assert props["T_harden"] < props["T_melt"], \
                    f"Ошибка в {mat_name}: T_harden ({props['T_harden']}) >= T_melt ({props['T_melt']})"