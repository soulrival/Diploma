"""
tests/conftest.py
Глобальные фикстуры (fixtures) для модульных и интеграционных тестов.
"""
import pytest
import numpy as np
import os
import json

from core import materials
from core.math_engine import compute_spatial_field, source_rect, source_gauss

@pytest.fixture
def default_steel_params():
    """Стандартный набор параметров для стали 20 (базовый сценарий)"""
    return {
        "l": 10.0,
        "a": 0.1,
        "c_V": 4.2,
        "lam": 0.5,
        "q_max": 10000.0,
        "t_source_end": 5.0,
        "T_init": 20.0,
        "T_melt": 1510.0,
        "T_harden": 723.0,
        "N_terms": 40,  # Уменьшено для ускорения тестов, но достаточно для сходимости
        "N_eta": 200,
        "N_tau": 100
    }

@pytest.fixture
def spatial_grids():
    """
    Стандартные сетки по времени и координате.
    Время увеличено до 60.0 с, чтобы температура успела упасть ниже 500°C 
    для корректного расчёта параметра t85 (имитация реальных условий из GUI).
    """
    t_arr = np.linspace(0.1, 60.0, 120)
    y_arr = np.linspace(0.0, 10.0, 100)
    return t_arr, y_arr

@pytest.fixture
def temp_materials_dir(tmp_path):
    """
    Фикстура для изолированного тестирования работы с JSON-файлами.
    Создаёт временную директорию и меняет рабочую директорию, 
    чтобы не засорять реальный проект тестовыми файлами.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    # Инициализируем чистую базу в временной папке
    materials.MATERIALS_DB = materials.HARDCODED_MATERIALS_DB.copy()
    materials.BUILTIN_MATERIALS = list(materials.HARDCODED_MATERIALS_DB.keys())
    
    yield tmp_path
    
    # Возвращаем рабочую директорию обратно после тестов
    os.chdir(original_cwd)