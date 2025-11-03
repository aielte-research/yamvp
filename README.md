![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![pypi](https://img.shields.io/pypi/v/yamvp.svg)](https://pypi.org/project/yamvp/)
# YAMVP - Yet Another Matplotlib Venn-diagram Plotter
## Overview
This module provides a function to create matplotlib figures containing ellipse-based Venn-diagramms with up to 5 Classes.

The main goal was to provide a simple interface, and good looks.

Class colors are averaged instead of simply using alpha, making the intersection colors invariant to the order of the ellipses drawn.  

## Installation
```
pip install yamvp
```

## Basic usage
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
fig.savefig("rand4_demo.png", dpi=100, bbox_inches="tight")
plt.close(fig)
```
![rand4_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand4_demo.png?raw=true "rand4_demo")

## n=1
```python
venn([None, "A"], ["Alpha"], outfile = "venn1_demo.png")
```
![venn1_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn1_demo.png?raw=true "venn1_demo")

## n=2
```python
venn([[None, "B"], ["A", "AB"]], ["Alpha", "Beta"], outfile = "venn2_demo.png")
```
![venn2_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn2_demo.png?raw=true "venn2_demo")

## n=3
```python
vals3 = [
    [[None, "C"], ["B", "BC"]],
    [["A", "AC"], ["AB", "ABC"]],
]
venn(vals3, ["Alpha", "Beta", "Gamma"], outfile = "venn3_demo.png")
```
![venn3_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn3_demo.png?raw=true "venn3_demo")

## n=4
```python
vals4 = np.empty((2, 2, 2, 2), dtype=object)
for i, yA in enumerate(("", "A")):
    for j, yB in enumerate(("", "B")):
        for k, yC in enumerate(("", "C")):
            for l, yD in enumerate(("", "D")):
                vals4[i, j, k, l] = yA + yB + yC + yD
vals4[0,0,0,0] = None

venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], outfile = "venn4_demo.png")
```
![venn4_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo.png?raw=true "venn4_demo")

## n=5
```python
vals5 = np.empty((2, 2, 2, 2, 2), dtype=object)
for i, yA in enumerate(("", "A")):
    for j, yB in enumerate(("", "B")):
        for k, yC in enumerate(("", "C")):
            for l, yD in enumerate(("", "D")):
                for m, yE in enumerate(("", "E")):
                    vals5[i, j, k, l, m] = yA + yB + yC + yD + yE
vals5[0,0,0,0,0] = None

venn(vals5, ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], outfile = "venn5_demo.png")
```
![venn5_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn5_demo.png?raw=true "venn5_demo")

## Custom colors
```python
venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], colors=["red", "green", "blue", "yellow"], outfile = "venn4_demo_colors.png")
```
![venn4_demo_colors](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo_colors.png?raw=true "venn4_demo_colors")

## License
This project is licensed under the MIT License (c) 2025 Bálint Csanády, aielte-research. See the LICENSE file for details.