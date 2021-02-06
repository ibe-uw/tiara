# Detailed installation

For someone who doesn't use Python, the easiest way to install **tiara** is to use a conda environment.
This way your existing Python-dependent programs won't be affected by, for example, updating some libraries.

## Python installation

1. Follow the installation instruction for your OS on: https://docs.conda.io/en/latest/miniconda.html. 
Install Python >=3.7 version. (Most tested on 3.8).

2. After you installed (mini)conda and verified it works, create new virtual environment, for example `tiara-env`:
    ```bash
    conda create --name tiara-env python=3.8
    ```
3. After the environment is crated, run `conda activate tiara-env`.

You can of course install **tiara** in any Python environment (`venv`, `virtualenv` etc),
but it should be a fresh one.


## **Tiara** installation

Now you can install **tiara** in your environment. There are two ways to handle dependencies:
- You can install them by hand: in the `tiara-env` run `conda install numpy pytorch numba joblib` 
and then `conda install -c conda-forge tqdm biopython skorch`. 
(Or just simply `pip install -r requirements.txt`)
- You can let the setup.py script install them for you.
It could potentially cause problems with your conda environment though, 
but if you use it only for **tiara**, everything should be fine. This is a recommended option.

Now you can clone this repository, navigate to the directory containing setup.py and run it:

```bash
git clone https://github.com/ibe-uw/tiara.git
cd tiara
python setup.py install
```

Verify the installation with `tiara-test` (it may take a while).

[Back to README](README.md)

