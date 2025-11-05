import os
import matplotlib.pyplot as plt
import numpy as np
from yamvp import venn

if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    
    # 4-set Random
    rand4 = np.random.randint(0, 1000, size=(2, 2, 2, 2))
    
    rand4[1,1,0,0] = 42

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
    venn(vals3, ["Alpha", "Beta", "Gamma"], colors=["red", "green", "blue"], outfile = "img/venn3_demo_colors.png")
    
    # color mixing    
    venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing = "average", outfile="img/venn4_demo_colors_average_mixing.png")
    
    venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing = "alpha", outfile = "img/venn4_demo_colors_alpha_mixing.png")       
    
    def color_mix_multiply(colors):
        arr = np.stack([np.array(c, float) for c in colors], axis=0)
        return np.prod(arr, axis=0)
    
    venn(vals4, ["Alpha", "Beta", "Gamma", "Delta"], color_mixing=color_mix_multiply, outfile="img/venn4_demo_colors_multiply_mixing.png", text_color="white")
    
    # 2-set Area Proportional
    rand2 = np.random.randint(0, 1000, size=(2, 2))
    fig = venn(rand2, ["Alpha", "Beta"], area_proportional=True)
    fig.savefig("img/rand2_demo.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    rand2 = np.random.randint(0, 1000, size=(2, 2))
    rand2[1,1] = 0
    fig = venn(rand2, ["Alpha", "Beta"], area_proportional=True)
    fig.savefig("img/rand2_demo2.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    rand2 = np.random.randint(0, 1000, size=(2, 2))
    rand2[1,0] = 0
    fig = venn(rand2, ["Alpha", "Beta"], area_proportional=True)
    fig.savefig("img/rand2_demo3.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    rand2[0,1] = 0
    fig = venn(rand2, ["Alpha", "Beta"], area_proportional=True)
    fig.savefig("img/rand2_demo4.png", dpi=100, bbox_inches="tight")
    plt.close(fig)