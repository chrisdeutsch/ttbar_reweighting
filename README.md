# Quick Setup

```bash
# Instructions tested on Ubuntu 20.04

# Check out package
git clone git@github.com:chrisdeutsch/ttbar_reweighting.git

# Create virtual environment
virtualenv ttbar_reweighting_env

# Activate virtual environment
source ttbar_reweighting_env/bin/activate

# Install requirements
pip install -r ttbar_reweighting/requirements.txt

# Set paths
PATH="$(readlink -e ttbar_reweighting/scripts):${PATH}"
if [[ -z "${PYTHONPATH}" ]]; then
    export PYTHONPATH="$(readlink -e ttbar_reweighting)"
else
    PYTHONPATH="$(readlink -e ttbar_reweighting):${PYTHONPATH}"
fi


# Ready to reweight
reweight_nn.py /cephfs/user/s6crdeut/fr_ntups/ntups_v3
```
