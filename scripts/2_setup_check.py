"""
Automated Setup Check (Version 2)
==================================
نسخة محسّنة للفحص التلقائي للمكتبات
تعرض معلومات تفصيلية مع خيار التثبيت التلقائي
"""

import importlib
import subprocess
import sys
import platform


def check_python_version():
    """فحص إصدار Python"""
    print("=" * 70)
    print("🐍 Python Environment Check")
    print("=" * 70)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print()
    
    # التأكد من إصدار Python 3.7+
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 7):
        print("⚠️  WARNING: Python 3.7+ is recommended")
        print(f"   Current version: {version_info.major}.{version_info.minor}")
        return False
    return True


def check_package(package_name, import_name=None):
    """
    فحص حزمة واحدة
    
    Args:
        package_name: اسم الحزمة في pip
        import_name: اسم الحزمة عند الاستيراد (إذا كان مختلفاً)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✅ {package_name:20s} v{version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name:20s} NOT INSTALLED")
        return False


def install_package(package_name):
    """تثبيت حزمة مفقودة"""
    print(f"\n📦 Installing {package_name}...")
    try:
        # ترقية pip أولاً
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # تثبيت الحزمة
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"   ✅ {package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install {package_name}")
        print(f"      Error: {e}")
        return False


def main():
    """البرنامج الرئيسي"""
    # فحص Python
    if not check_python_version():
        print("\n⚠️  Please upgrade Python to version 3.7 or higher")
        return
    
    # قائمة الحزم المطلوبة
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("imbalanced-learn", "imblearn"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("streamlit", "streamlit"),
        ("pyyaml", "yaml"),
        ("shap", "shap"),
    ]
    
    print("\n" + "=" * 70)
    print("📚 Checking Required Packages")
    print("=" * 70)
    
    # فحص جميع الحزم
    missing_packages = []
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    # إذا كانت هناك حزم مفقودة
    if missing_packages:
        print("\n" + "=" * 70)
        print(f"⚠️  Found {len(missing_packages)} missing package(s)")
        print("=" * 70)
        
        # خيار التثبيت التلقائي
        response = input("\n❓ Install missing packages automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("\n🔧 Installing missing packages...")
            failed = []
            for pkg in missing_packages:
                if not install_package(pkg):
                    failed.append(pkg)
            
            if failed:
                print("\n" + "=" * 70)
                print("❌ Installation Failed")
                print("=" * 70)
                print("The following packages could not be installed:")
                for pkg in failed:
                    print(f"  - {pkg}")
                print("\nPlease install them manually:")
                print(f"  pip install {' '.join(failed)}")
            else:
                print("\n" + "=" * 70)
                print("✅ All packages installed successfully!")
                print("=" * 70)
        else:
            print("\n📝 To install missing packages manually, run:")
            print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("\n" + "=" * 70)
        print("✅ All required packages are installed!")
        print("=" * 70)
    
    print("\n🎉 Setup check complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

