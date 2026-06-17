"""
tests/unit/test_materials.py
Тестирование модуля управления базой данных материалов.
Проверяет целостность данных, сохранение и защиту встроенных материалов.
"""
import pytest
import os
import json
from core import materials

class TestMaterialsDatabaseIntegrity:
    """Тесты целостности исходной базы данных"""
    
    def test_hardcoded_materials_exist(self):
        """Проверка наличия всех базовых материалов"""
        expected_materials = ["Сталь 20", "Сталь 45", "Сталь 09Г2С", "Алюминий АМг5", "Титан ВТ6", "Медь М1"]
        for mat in expected_materials:
            assert mat in materials.HARDCODED_MATERIALS_DB

    def test_material_properties_structure(self):
        """Проверка, что каждый материал содержит все необходимые ключи"""
        required_keys = {"c_V", "a", "lam", "T_melt", "T_harden", "density", "desc"}
        for mat_name, props in materials.HARDCODED_MATERIALS_DB.items():
            assert set(props.keys()) == required_keys, f"Материал {mat_name} имеет неполную структуру"

class TestMaterialsOperations:
    """Тесты операций CRUD с материалами (с использованием изолированной директории)"""
    
    def test_get_material_properties(self, temp_materials_dir):
        """Корректное получение свойств существующего и несуществующего материала"""
        props = materials.get_material_properties("Сталь 20")
        assert props is not None
        assert props["c_V"] == 4.2
        
        props_none = materials.get_material_properties("Несуществующий материал")
        assert props_none is None

    def test_add_custom_material(self, temp_materials_dir):
        """Добавление нового пользовательского материала и его сохранение в JSON"""
        test_mat = {
            "c_V": 3.5, "a": 0.15, "lam": 0.6, 
            "T_melt": 1400, "T_harden": 600, "density": 7.5, "desc": "Тест"
        }
        materials.add_custom_material("Тестовый сплав", test_mat)
        
        # Проверка в памяти
        assert "Тестовый сплав" in materials.MATERIALS_DB
        assert materials.MATERIALS_DB["Тестовый сплав"]["c_V"] == 3.5
        
        # Проверка на диске
        assert os.path.exists("custom_materials.json")
        with open("custom_materials.json", "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert "Тестовый сплав" in saved_data

    def test_add_custom_material_prevents_overwrite(self, temp_materials_dir):
        """Попытка добавить материал с именем встроенного должна вызывать ошибку"""
        test_mat = {"c_V": 1.0, "a": 1.0, "lam": 1.0, "T_melt": 1000, "T_harden": 500, "density": 7.0, "desc": "Test"}
        with pytest.raises(ValueError, match="Нельзя перезаписать базовый материал"):
            materials.add_custom_material("Сталь 20", test_mat)

    def test_delete_custom_material(self, temp_materials_dir):
        """Удаление пользовательского материала"""
        test_mat = {"c_V": 3.5, "a": 0.15, "lam": 0.6, "T_melt": 1400, "T_harden": 600, "density": 7.5, "desc": "Test"}
        materials.add_custom_material("Удаляемый сплав", test_mat)
        assert "Удаляемый сплав" in materials.MATERIALS_DB
        
        materials.delete_custom_material("Удаляемый сплав")
        assert "Удаляемый сплав" not in materials.MATERIALS_DB
        
        # Проверка, что он удалён и из файла
        with open("custom_materials.json", "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert "Удаляемый сплав" not in saved_data

    def test_delete_builtin_material_prevented(self, temp_materials_dir):
        """Попытка удалить встроенный материал должна вызывать ошибку"""
        with pytest.raises(ValueError, match="Нельзя удалить базовый материал"):
            materials.delete_custom_material("Сталь 20")

    def test_export_and_import_builtin(self, temp_materials_dir):
        """Экспорт и импорт базы встроенных материалов (проверка логики полной замены базы)"""
        # 1. Экспорт текущей базы
        materials.export_builtin_materials()
        assert os.path.exists("builtin_materials.json")
        
        # 2. Создаём фейковый файл для импорта (с одним материалом)
        fake_import_data = {
            "Импортная Сталь": {"c_V": 4.0, "a": 0.1, "lam": 0.5, "T_melt": 1500, "T_harden": 700, "density": 7.8, "desc": "Imported"}
        }
        with open("fake_import.json", "w", encoding="utf-8") as f:
            json.dump(fake_import_data, f)
            
        # 3. Импортируем (по логике программы это ПОЛНОСТЬЮ ЗАМЕНЯЕТ список BUILTIN_MATERIALS)
        materials.import_builtin_materials("fake_import.json")
        
        # 4. Проверяем, что список встроенных материалов теперь состоит ТОЛЬКО из импортированных
        assert len(materials.BUILTIN_MATERIALS) == len(fake_import_data)
        assert "Импортная Сталь" in materials.BUILTIN_MATERIALS
        
        # 5. Проверяем, что материал успешно добавлен в общий словарь
        assert "Импортная Сталь" in materials.MATERIALS_DB
        assert materials.MATERIALS_DB["Импортная Сталь"]["c_V"] == 4.0