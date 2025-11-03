import os
import matplotlib.pyplot as plt
import numpy as np
from yamvp import venn

if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    
    # 4-set Random
    rand4 = np.random.randint(0, 1000, size=(2, 2, 2, 2))
    
    rand4[1,0,1,0] = 42

    fig = venn(rand4, ["A", "B", "C", "D"])
    fig.savefig("img/rand4_demo.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    # 1-set
    venn([None, "A"], ["Alpha"], outfile = "img/venn1_demo.png")

    # 2-set
    venn([[None, "B"], ["A", "AB"]], ["Alpha", "Beta"], outfile = "img/venn2_demo.png")
    
    # 3-set
    vals3 = [
        [[None, "C"], ["B", "BC"]],
        [["A", "AC"], ["AB", "ABC"]],
    ]
    venn(vals3, ["Alpha", "Beta", "Gamma"], outfile = "img/venn3_demo.png")

    # 4-set
    vals4 = np.empty((2, 2, 2, 2), dtype=object)
    for i, yA in enumerate(("", "A")):
        for j, yB in enumerate(("", "B")):
            for k, yC in enumerate(("", "C")):
                for l, yD in enumerate(("", "D")):
                    vals4[i, j, k, l] = yA + yB + yC + yD
    vals4[0,0,0,0] = None
    #vals4[0,0,0,0] = "Other"

    venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], outfile = "img/venn4_demo.png")

    # 5-set
    vals5 = np.empty((2, 2, 2, 2, 2), dtype=object)
    for i, yA in enumerate(("", "A")):
        for j, yB in enumerate(("", "B")):
            for k, yC in enumerate(("", "C")):
                for l, yD in enumerate(("", "D")):
                    for m, yE in enumerate(("", "E")):
                        vals5[i, j, k, l, m] = yA + yB + yC + yD + yE
    vals5[0,0,0,0,0] = None
    #vals5[0,0,0,0,0] = "Other"
    
    venn(vals5, ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], outfile = "img/venn5_demo.png")
    
    # 4-set Colors
    venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], colors=["red", "green", "blue", "yellow"], outfile = "img/venn4_demo_colors.png")
