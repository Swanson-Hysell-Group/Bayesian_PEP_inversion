# Bayesian PEP inversion

This repository is for storing code related to Bayesian framework of doing paleomagnetic Euler pole inversion and apparent polar wander path analyses.





### setting up Python environment

Conda users can use the _environment.yml_ file in the repository to generate a working Conda Python environment  that can execute the code in this repository. 



In terminal navigate to the repository where the environment.yml file is and type this in the command:

`conda env creat -f environment.yml`



after the environment is installed, activate the environment:

`conda activate pymc3_env`



Then install ipykernel:

`pip install ipykernel`



Install the new kernel:

```python
ipython kernel install --user --name=pymc3_env
```



Now you should be able to see the new kernel available in your Jupyter Notebook or Jupyter Lab from the base environment.
