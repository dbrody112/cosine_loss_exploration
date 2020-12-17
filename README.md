# cosine_loss_exploration
Note: Python under should now work and the Jupyter Notebook is messy but also works. Use any one of these two or both for reference. For any version as well you must run the code in folder with the chexpert directory. The chexpert dataset requires permission but is completely downloadable here : https://stanfordmlgroup.github.io/competitions/chexpert/

Project : On optimization of confidence levels in small datasets
The big idea around the project was to create an OHEM that uses cosine loss+cross entropy as the top-k loss instead of the original cross entropy. The basis of this idea was that the top-k loss would be confident from the cosine entropy and act as a confidence metric so that any values that would be too confident were penalized. Unfortunately our method backfired and became overly confident, incorrectly shifting the weights and choosing the wrong 
