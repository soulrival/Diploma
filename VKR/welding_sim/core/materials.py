"""
core/materials.py
Управление базой данных материалов (JSON)
"""
import json
import os

# Захардкоженная база (Fallback, если builtin_materials.json отсутствует)
HARDCODED_MATERIALS_DB = {
    "Сталь 20": {"c_V": 4.2, "a": 0.1, "lam": 0.5, "T_melt": 1510, "T_harden": 723, "density": 7.85, "desc": "Конструкционная углеродистая сталь"},
    "Сталь 45": {"c_V": 4.1, "a": 0.09, "lam": 0.48, "T_melt": 1490, "T_harden": 723, "density": 7.85, "desc": "Среднеуглеродистая сталь"},
    "Сталь 09Г2С": {"c_V": 4.0, "a": 0.11, "lam": 0.45, "T_melt": 1480, "T_harden": 723, "density": 7.82, "desc": "Низколегированная сталь"},
    "Алюминий АМг5": {"c_V": 2.4, "a": 0.8, "lam": 1.2, "T_melt": 660, "T_harden": 0, "density": 2.7, "desc": "Алюминиево-магниевый сплав"},
    "Титан ВТ6": {"c_V": 3.6, "a": 0.04, "lam": 0.15, "T_melt": 1660, "T_harden": 995, "density": 4.43, "desc": "Титановый сплав (Ti-6Al-4V)"},
    "Медь М1": {"c_V": 3.45, "a": 1.1, "lam": 3.8, "T_melt": 1083, "T_harden": 0, "density": 8.94, "desc": "Электролитическая медь"},
}

# Глобальные переменные, будут инициализированы в load_materials
MATERIALS_DB = HARDCODED_MATERIALS_DB.copy()
BUILTIN_MATERIALS = list(HARDCODED_MATERIALS_DB.keys())

def load_materials():
    global MATERIALS_DB, BUILTIN_MATERIALS
    MATERIALS_DB = HARDCODED_MATERIALS_DB.copy()
    BUILTIN_MATERIALS = list(HARDCODED_MATERIALS_DB.keys())
    
    if os.path.exists("builtin_materials.json"):
        try:
            with open("builtin_materials.json", "r", encoding="utf-8") as f:
                builtin_json = json.load(f)
            MATERIALS_DB.update(builtin_json)
            BUILTIN_MATERIALS = list(builtin_json.keys())
        except Exception as e:
            print(f"Warning loading builtin_materials.json: {e}")
            
    if os.path.exists("custom_materials.json"):
        try:
            with open("custom_materials.json", "r", encoding="utf-8") as f:
                custom = json.load(f)
            for k, v in custom.items():
                if k not in BUILTIN_MATERIALS:
                    MATERIALS_DB[k] = v
        except Exception as e:
            print(f"Warning loading custom_materials.json: {e}")

def save_custom_materials():
    try:
        custom = {k: v for k, v in MATERIALS_DB.items() if k not in BUILTIN_MATERIALS}
        with open("custom_materials.json", "w", encoding="utf-8") as f:
            json.dump(custom, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Ошибка сохранения материалов: {e}")

def export_builtin_materials():
    data = {k: MATERIALS_DB[k] for k in BUILTIN_MATERIALS if k in MATERIALS_DB}
    with open("builtin_materials.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def import_builtin_materials(filepath):
    global MATERIALS_DB, BUILTIN_MATERIALS
    with open(filepath, "r", encoding="utf-8") as f:
        imported = json.load(f)
    if not isinstance(imported, dict):
        raise ValueError("Файл не содержит словарь материалов")
    
    with open("builtin_materials.json", "w", encoding="utf-8") as f:
        json.dump(imported, f, indent=2, ensure_ascii=False)
    
    custom_mats = {k: v for k, v in MATERIALS_DB.items() if k not in BUILTIN_MATERIALS}
    
    MATERIALS_DB = HARDCODED_MATERIALS_DB.copy()
    MATERIALS_DB.update(imported)
    BUILTIN_MATERIALS = list(imported.keys())
    
    for k, v in custom_mats.items():
        if k not in BUILTIN_MATERIALS:
            MATERIALS_DB[k] = v
            
    save_custom_materials()

# === НОВЫЕ ФУНКЦИИ (добавлены для тестов и GUI) ===

def get_material_properties(mat_name):
    """Получение свойств материала по имени"""
    if mat_name in MATERIALS_DB:
        return MATERIALS_DB[mat_name]
    return None

def add_custom_material(name, properties):
    """Добавление пользовательского материала"""
    if name in BUILTIN_MATERIALS:
        raise ValueError(f"Нельзя перезаписать базовый материал '{name}'")
    MATERIALS_DB[name] = properties
    save_custom_materials()

def delete_custom_material(name):
    """Удаление пользовательского материала"""
    if name in BUILTIN_MATERIALS:
        raise ValueError(f"Нельзя удалить базовый материал '{name}'")
    if name in MATERIALS_DB:
        del MATERIALS_DB[name]
        save_custom_materials()