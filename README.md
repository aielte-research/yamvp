![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![pypi](https://img.shields.io/pypi/v/yamvp.svg)](https://pypi.org/project/yamvp/)
# YAMVP - Yet Another Matplotlib Venn-diagram Plotter
## Overview
This module provides a function for creating Matplotlib figures with ellipse-based Venn diagrams for up to five classes.
Its primary aim is a simple interface and visually appealing results.

Features:
- Custom color mixing callbacks.
- Area-proportional option for $n=2$.

## Installation
```
pip install yamvp
```

## Basic Usage
```python
import numpy as np
import matplotlib.pyplot as plt

from yamvp import venn

# Fill a 4D 2x2x2x2 array with random values
rand4 = np.random.randint(0, 1000, size=(2, 2, 2, 2))
    
# Set the value at A∩B to 42
rand4[1,1,0,0] = 42

# Create the Venn-diagram
fig = venn(rand4, ["A", "B", "C", "D"])

# Save the figure
fig.savefig("rand4_demo.png", dpi=100, bbox_inches="tight")
plt.close(fig)
```
![rand4_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand4_demo.png?raw=true "rand4_demo")

## $n=2$
```python
venn([[None, "B"], ["A", "AB"]], ["Alpha", "Beta"], outfile = "venn2_demo.png")
```
![venn2_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn2_demo.png?raw=true "venn2_demo")

## $n=3$
```python
vals3 = [
    [[None, "C"], ["B", "BC"]],
    [["A", "AC"], ["AB", "ABC"]],
]
venn(vals3, ["Alpha", "Beta", "Gamma"], outfile = "venn3_demo.png")
```
![venn3_demo](https://github.com/aielte-research/yamvp/blob/master/img/venn3_demo.png?raw=true "venn3_demo")

## $n=4$
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

## $n=5$
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

## Additional Color Mixing Options
### Average
Each intersection’s color is the mean of the corresponding class colors.
```python
venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing = "average", outfile="venn4_demo_colors_average_mixing.png")
```
![venn4_demo_colors_average_mixing](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo_colors_average_mixing.png?raw=true "venn4_demo_colors_average_mixing")

### Alpha Stacking
This is what would happen if we simply stacked the ellipses with opacity = 0.5. The result depends on the order in which the ellipses are drawn. 
```python
venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing = "alpha", outfile = "venn4_demo_colors_alpha_mixing.png")      
```
![venn4_demo_colors_alpha_mixing](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo_colors_alpha_mixing.png?raw=true "venn4_demo_colors_alpha_mixing")

### Custom Mixing Callback 
```python
def color_mix_multiply(colors):
    arr = np.stack([np.array(c, float) for c in colors], axis=0)
    return np.prod(arr, axis=0)

venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing=color_mix_multiply, outfile="venn4_demo_colors_multiply_mixing.png", text_color="white")     
```
![venn4_demo_colors_multiply_mixing](https://github.com/aielte-research/yamvp/blob/master/img/venn4_demo_colors_multiply_mixing.png?raw=true "venn4_demo_colors_multiply_mixing")

## Custom Class Colors
```python
venn(vals3, ["Alpha", "Beta", "Gamma"], colors=["red", "green", "blue"], outfile = "venn3_demo_colors.png")
```
![venn3_demo_colors](https://github.com/aielte-research/yamvp/blob/master/img/venn3_demo_colors.png?raw=true "venn3_demo_colors")

## Area-Proportional Option
The `area_proportional` flag is ignored if $n > 3$ or if the input data does not contain positive numbers.
### $n=2$
For $n = 2$, it is possible to draw the diagram with areas proportional to the given data. 
```python
rand2 = np.random.randint(0, 1000, size=(2, 2))
venn(rand2, ["Alpha", "Beta"], area_proportional=True, outfile = "rand2_demo.png")
```
![rand2_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand2_demo.png?raw=true "rand2_demo")
### $n=3$
For $n = 3$, it is generally NOT possible to draw the diagram area-proportionally using circles only.
However we can try morphing the circles into ellipses.
```python
rand3 = np.array(np.random.randint(0, 1000, size=(2, 2, 2)), dtype=object)
rand3[0,0,0] = None
venn(rand3, ["Alpha", "Beta", "Gamma"], area_proportional=True, outfile = "rand3_demo.png")
```
![rand3_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand3_demo.png?raw=true "rand3_demo")

The optimizer is based on a monte-carlo greedy hill climing algorithm, it tries to find a solution with sufficient loss, and selects the best result on using a quality score. 

![rand3_demo](https://github.com/aielte-research/yamvp/blob/master/img/rand3_demo.gif?raw=true "rand3_demo")

## License
This project is licensed under the MIT License (c) 2025 Bálint Csanády, aielte-research. See the LICENSE file for details.
