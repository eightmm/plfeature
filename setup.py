"""
Setup script for plfeature package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plfeature",
    version="0.1.0",
    author="Jaemin Sim",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for extracting features from molecular and protein structures for protein-ligand modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eightmm/plfeature",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "protein-featurizer=plfeature.protein_featurizer.main:main",
            "pdb-standardize=plfeature.protein_featurizer.pdb_standardizer:main",
            "extract-protein-features=plfeature.protein_featurizer.residue_featurizer:main",
            "extract-mol-features=plfeature.molecule_featurizer.molecular_feature:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="protein-ligand molecule pdb smiles feature-extraction bioinformatics chemistry structural-biology machine-learning drug-discovery gnn",
    project_urls={
        "Bug Reports": "https://github.com/eightmm/plfeature/issues",
        "Source": "https://github.com/eightmm/plfeature",
    },
)