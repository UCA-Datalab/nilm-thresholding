import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nilm-thresholding",
    use_scm_version=True,
    # version="0.1.0",
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
        # TODO: Add requirements
    ]
)

