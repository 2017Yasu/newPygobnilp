import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygobnilp", 
    version="1.0",
    author="James Cussens",
    author_email="james.cussens@york.ac.uk",
    description="Bayesian network learning with the Gurobi MIP solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/jamescussens/pygobnilp/",
    license = "GPLv3",
    project_urls = {
        'Documentation': 'https://pygobnilp.readthedocs.io/en/latest/',
        'Source': "https://bitbucket.org/jamescussens/pygobnilp/"
    },
    install_requires = ['scipy','pygraphviz','matplotlib',
                        'networkx','pandas','numpy','scikit-learn',
                        'numba'], #gurobipy not in pypi
    scripts = ['rungobnilp.py'],
    packages=['pygobnilp'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
