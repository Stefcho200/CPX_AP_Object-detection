"""Loads project.json - the ONE file to edit when you want the app to
recognize a different set of objects. See README.md.

- app_title:   name shown in the main window.
- training:    epochs / patience / imgsz / batch / warm_start used by
               "Run training".
- twin_groups: lists of class names that look identical and must be told
               apart by reading printed text (OCR). Leave as [] for
               ordinary objects.

What the model detects is NOT set here: every folder under data/ is one
class. Create classes from the app (Training -> Train model on new product).
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PATH = os.path.join(BASE_DIR, "project.json")

DEFAULTS = {
    "app_title": "Object Recognizer",
    "training": {
        "epochs": 100,
        "patience": 20,
        "imgsz": 640,
        "batch": 8,
        "warm_start": True,
    },
    "twin_groups": [],
}


def load_project():
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if not os.path.exists(PROJECT_PATH):
        return cfg
    try:
        with open(PROJECT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[project_config] could not read project.json, using defaults: {e}")
        return cfg
    for key, value in data.items():
        if key == "training" and isinstance(value, dict):
            cfg["training"].update(value)
        else:
            cfg[key] = value
    return cfg


def app_title():
    return load_project()["app_title"]


def training_settings():
    return load_project()["training"]


def twin_groups():
    """Returns the twin groups as a list of sets."""
    return [set(g) for g in load_project().get("twin_groups", []) if g]
