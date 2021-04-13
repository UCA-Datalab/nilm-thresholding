import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nilm_thresholding",
    use_scm_version=True,
    version="1.1.2",
    author="Daniel Precioso Garcelán",
    author_email="daniel.precioso@uca.es",
    description="Non-Intrusive Load Monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniprec/nilm-thresholding",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.toml']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, <=3.8.2',
    setup_requires=['setuptools_scm'],
    install_requires=[
        "Keras==2.3.1",
        "matplotlib==3.2.1",
        "numpy==1.18.1",
        "pandas==0.24.2",
        "scikit-learn==0.22.2.post1",
        "seaborn==0.10.0",
        "tables==3.6.1",
        "tensorflow==2.3.1",
        "toml==0.10.2",
        "torch==1.4.0",
        "torchvision==0.5.0",
        "typer==0.3.2"
    ]
)

