# Cosine Ohem Loss Exploration
Note: Python should now work and the Jupyter Notebook is messy but also works. Use any one of these two or both for reference. For any version as well you must run the code in a directory containing the chexpert dataset. The chexpert dataset requires permission but is completely downloadable here : https://stanfordmlgroup.github.io/competitions/chexpert/


![ohem](https://user-images.githubusercontent.com/59486373/104828037-36267b00-5833-11eb-8e63-12c1eba829bd.png)

<h4><strong>Figure 1: Formulation of OHEM Loss with Input and Top-K Loss Highlighted</strong></h4>

![cosine_ohem](https://user-images.githubusercontent.com/59486373/104856875-d8516c00-58e2-11eb-84ed-6ce40a3c53b1.png)

<h4><strong>Figure 2: Formulation of Cosine OHEM Top-K Loss</strong></h4>

<h2>Objective:  Optimization of Confidence Levels in Small Datasets</h2>

The big idea around the project was to create an OHEM that uses cosine loss+cross entropy as the top-k loss (referred to as cosine OHEM loss) instead of the original cross entropy for the top-k loss as shown in Figure 1.  The cosine loss was very effective in <i>Deep Learning on Small Datasets without Pre-Training using Cosine Loss</i> by Joachim Denzler et al. (https://arxiv.org/pdf/1901.09054v1.pdf). The basis of this idea was that the top-k loss inside of the cosine OHEM loss would act as a confidence metric so that any values that would be too confident were largely penalized. In this way, we could find a way to prevent overconfidence in small datasets. Unfortunately, for the time allotted for the project, our method backfired and became overly confident, incorrectly shifting the weights and choosing the wrong losses. 

<i>*For more info feel free to read cosine_ohem_loss_paper.pdf.</i>

[12/17/2020] Update: cosine ohem on the affine transformed chexpert actually showed that subtracting the cosine loss from the cross entropy provides a higher correlation from training accuracy to testing accuracy and therefore overfits later than the normal ohem. Adding cross entropy to the cosine loss, however, does the exact opposite. This is shown in these figures over 15 epochs (with epochs on x-axis and accuracy on y-axis):

![ohemmm](https://user-images.githubusercontent.com/59486373/104858047-92000b00-58ea-11eb-9cba-24532ce3501a.png)

<i>*Note: Testing accuracy was only monitored every three epochs so the testing accuracy curve does not have as many points as the training accuracy curve</i>
