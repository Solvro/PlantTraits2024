from setuptools import setup, find_packages

setup(
    name="planttraits",
    version="0.0.1",
    description="Codebase for Plant Traits Challenge",
    python_requires=">=3.8,<3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch ~= 2.1",
        "jupyter ~= 1.0.0",
        "pytorch-lightning ~= 2.1",
        "lightning ~= 2.3",
        "torchmetrics ~= 1.2",
        "torchvision ~= 0.16",
        "scipy ~= 1.10",
        "numpy ~= 1.24",
        "wandb",
        "scikit-image",
        "tensorboard",
        "python-dotenv==1.0.1",
    ],
    extras_require={
        "jupyter": [
            "jupyterlab~=3.6",
            "pandas ~= 2.0",
            "matplotlib ~= 3.7",
            "seaborn ~= 0.12",
            "plotly ~= 5.14",
            "torchsummary==1.5.1",
            "transformers ~= 4.47.1",
            "evaluate ~= 0.4.3",
            "transformers[torch]",
            "datasets[vision]",
        ],
        "lint": [
            "ruff ~= 0.1",
            "pre-commit ~= 2.20",
        ],
        "test": [
            "pytest ~= 7.1",
            "pytest-cases ~= 3.6",
            "pytest-cov ~= 3.0",
            "pytest-xdist ~= 2.5",
            "pytest-sugar ~= 0.9",
        ],
        "dev": [
            "planttraits[jupyter, lint, test]",
        ],
    },
)
