import importlib, subprocess, sys

pkgs = ["pandas","numpy","sklearn","xgboost","imblearn","matplotlib","plotly","yaml","shap"]
print("Python:", sys.version)
print("-"*60)
for pkg in pkgs:
    try:
        m = importlib.import_module(pkg if pkg!="yaml" else "yaml")
        print(f"{pkg:12s} ✅ OK\tversion: {getattr(m,'__version__','n/a')}")
    except ImportError:
        print(f"{pkg:12s} ❌ MISSING — Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print(f"{pkg:12s} ✅ Installed successfully.")
print("-"*60)
print("✅ All dependencies checked and installed.")
