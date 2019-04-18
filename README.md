# Individual mixing patterns in networks
Python class for Bayesian analysis of mixing patterns in networks.

This code implements the methods described in:<br>
[Mixing patterns and individual differences in networks](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.042306)
[[arXiv]](https://arxiv.org/abs/1810.01432)
<br>
George T. Cantwell and M. E. J. Newman<br>
Phys. Rev. E 99, 042306<br>



The basic usage would be
```python
import individual_mixing

DM = individual_mixing.DirichletModel(k,g)
DM.fit()
```
where `k` is the edge matrix and `g` is the group vector, as described in the original paper.
See `example.py` for an example.
