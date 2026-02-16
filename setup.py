from setuptools import setup, find_packages

setup(
    name="CascadeTorch",
    version="2.0",
    description="Calibrated inference of spiking from calcium ΔF/F data using deep networks with PyTorch",
    author="Peter Rupprecht",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch>=2.5",  # pip install CPU and GPU tensorflow
        "h5py",
        "seaborn",
        "ruamel.yaml",
    ],
)
