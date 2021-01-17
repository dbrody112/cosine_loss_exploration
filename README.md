# Cosine Ohem Loss Exploration
Note: Python under should now work and the Jupyter Notebook is messy but also works. Use any one of these two or both for reference. For any version as well you must run the code in folder with the chexpert directory. The chexpert dataset requires permission but is completely downloadable here : https://stanfordmlgroup.github.io/competitions/chexpert/



<h2>Objective:  Optimization of Confidence Levels in Small Datasets</h2>

The big idea around the project was to create an OHEM that uses cosine loss+cross entropy as the top-k loss instead of the original cross entropy (cosine OHEM loss). The basis of this idea was that the top-k loss inside of the cosine OHEM loss would be confident from the cosine entropy and act as a confidence metric so that any values that would be too confident were penalized. In this way, we could find a way to prevent overconfidence in small datasets. Unfortunately our method backfired and became overly confident, incorrectly shifting the weights and choosing the wrong losses. For more info feel free to read cosine_ohem_loss_paper.pdf.

<h3>OHEM Loss</h3>

![ohem](https://user-images.githubusercontent.com/59486373/104828037-36267b00-5833-11eb-8e63-12c1eba829bd.png)

<h4><strong>Figure 1: Formulation of OHEM Loss with Input and Top-K Loss Highlighted</strong></h4>

[12/17/2020] Update: cosine ohem on the affine transformed chexpert actually showed that subtracting the cosine loss from the cross entropy provides a higher correlation from train to test and therefore overfits later than the normal ohem while adding cross entropy from the cosine loss does the exact opposite. This is shown in these figures:

![subplot_graph_cosine_ohem_overfitting](https://user-images.githubusercontent.com/59486373/102566213-0e65bb00-40ad-11eb-9e70-f07d2d84fd5d.png)


