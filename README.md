# Individual mixing patterns in networks
Python class for Bayesian analysis of mixing patterns in networks.

The basic usage would be
```python
import individual_mixing

DM = individual_mixing.DirichletModel(k,g)
DM.fit()
```
where `k` is the edge matrix and `g` is the group vector.
See `example.py` for an example.
