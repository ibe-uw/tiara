import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tiara",
    version="1.0.0",
    description="A tool for classifying metagenomic data",
    author="Michał Karlicki and Stanisław Antonowicz",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.7",
    keywords="machine-learning computational-biology",
    install_requires=[
        "biopython~=1.78",
        "joblib~=0.17.0",
        "numba~=0.52.0rc3",
        "numpy~=1.19.4",
        "skorch~=0.9.0",
        "torch~=1.7.0",
        "tqdm~=4.54.0",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": ["tiara=tiara.main:main", "tiara-test=tiara.test.test:test",]
    },
)
