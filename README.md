![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![pypi](https://img.shields.io/pypi/v/yamvp.svg)](https://pypi.org/project/yamvp/)
# YAMVP - Yet Another Matplotlib Venn-diagram Plotter
## Overview
This module provides a function to create matplotlib figures containing ellipse-based Venn-diagramms with up to 5 Classes.

## Installation
```
pip install yamvp
```

## Usage
### Basic usage
```python
from yamvp import venn
import numpy as np

#Fill a 4-dim 2x2x2x2 array with random values
rand4 = np.random.randint(0, 1000, size=(2, 2, 2, 2))
    
#Set the value at A∩C to 42
rand4[1,0,1,0] = 42

#Create the Venn-diagram
fig = venn(rand4, ["A", "B", "C", "D"])

#save the figure
fig.savefig("rand4_demo.png", dpi=350, bbox_inches="tight")
plt.close(fig)
```
![rand4_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand4_demo.png?raw=true "rand4_demo")



#### n=1:
```python
```
![venn1_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn1_demo.png?raw=true "venn1_demo")

#### n=2:
```python
```
![venn2_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn2_demo.png?raw=true "venn2_demo")

#### n=3:
```python
```
![venn3_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn3_demo.png?raw=true "venn3_demo")

#### n=4:
```python
```
![venn4_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo.png?raw=true "venn4_demo")

#### n=5:
```python
```
![venn5_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn5_demo.png?raw=true "venn5_demo")

## License
This project is licensed under the MIT License (c) 2025 Bálint Csanády, aielte-research. See the LICENSE file for details.