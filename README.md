# Bayesian PEP inversion

This repository contains code and results associated with a Bayesian framework for paleomagnetic Euler pole inversion and apparent polar wander path analyses. The associated paper in JGR-Solid Earth is: 

Rose, I., Zhang, Y., and  Swanson-Hysell, N.L. (2022), **Bayesian paleomagnetic Euler pole inversion for paleogeographic reconstruction and analysis**, *JGR: Solid Earth*, doi:10.1029/2021JB023890.
https://doi.org/10.1029/2021JB023890

The manuscript files be found in this repository: https://github.com/Swanson-Hysell-Group/Bayesian_PEP_inversion_manuscript 

## Repository structure

### bayesian_pep

This folder contains the Python library.

### code_output

This folder contains visualizations as well as raw PyMC output from the Jupyter notebooks .

### data 

This folder contains compiled data used for generating examples of apparent polar wander path inversions.

### pygplates 

This folder contains the pygplates library which is licensed with a CC BY 3.0 license and distributed by the EarthByte group: https://www.earthbyte.org/category/resources/software-workflows/pygplates/

### Jupyter Notebooks

Jupyter notebooks within the main folder of the repository demonstrate the functions of the package, conduct analysis, and develop visualizations that are presented in the accompanying manuscript.

## Setting up Python environment

Conda users can use the _environment.yml_ file in the repository to generate a working Conda Python environment  that can execute the code in this repository. 



In terminal navigate to the repository where the environment.yml file is and type this in the command:

`conda env create -f environment.yml`



after the environment is installed, activate the environment:

`conda activate pymc3_env`



Then install ipykernel:

`pip install ipykernel`



Install the new kernel:

```python
ipython kernel install --user --name=pymc3_env
```

There is an issue with the package shapely which can produce error of `IllegalArgumentException: ` when plotting figures.

To solve this, use the command below in your terminal within the activated `pymc3_env` to install the shapely package with no binary files.

`pip uninstall shapely`

`pip install shapely --no-binary shapely`

With these steps completed, a new kernel will be available in your Jupyter Notebook or Jupyter Lab from the base environment that can be used to run the notebooks.

To see and interact with the notebooks in this study you can open a new terminal window and type:

`jupyter lab`

When Jupyter Lab launches, you can navigate to the notebooks associated with this study. When you open them, you can change the kernel using the Change Kernel... option in the Kernel menu:

<img width="456" alt="Screen Shot 2022-09-14 at 9 17 34 AM" src="https://user-images.githubusercontent.com/4332322/190208187-2c1a46be-0eed-4538-b8d4-1c41512a1edc.png">

You could select the `pymc3_env` kernel:

<img width="383" alt="Screen Shot 2022-09-14 at 9 07 18 AM" src="https://user-images.githubusercontent.com/4332322/190206250-36d1b87e-befa-4a8d-91ab-35ee16db253c.png">

With that kernel selected, you should be good to go.
