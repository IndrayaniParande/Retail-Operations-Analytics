from setuptools import setup, find_packages

setup(
    name="olist-analytics",
    version="1.0.0",
    description="End-to-End Retail Analytics Pipeline on Olist Brazilian E-Commerce Dataset",
    author="Your Name",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
