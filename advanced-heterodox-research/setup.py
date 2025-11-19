"""
Setup script for Advanced Heterodox Economic Research Toolkit

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced Heterodox Economic Research Toolkit"

setup(
    name="heterodox-econ-toolkit",
    version="1.0.0",
    author="Claude",
    author_email="",
    description="Production-quality toolkit for heterodox economic research and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/heterodox-econ-toolkit",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Economics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "gui": ["PyQt6>=6.2.0"],
        "full": [
            "PyQt6>=6.2.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "sympy>=1.9",
            "networkx>=2.6",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.9b0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sfc-builder=pyqt_applications.sfc_model_builder:main",
        ],
    },
    package_data={
        "": ["*.json", "*.yaml", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "economics",
        "heterodox",
        "post-keynesian",
        "marxian",
        "kaleckian",
        "sfc",
        "stock-flow-consistent",
        "political-economy",
        "macroeconomics",
        "economic-modeling",
    ],
)
