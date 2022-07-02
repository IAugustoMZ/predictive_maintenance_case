# Project Chart - Predictive Maintenance Problem

## **Business Problem Understandment**

The problem that we face is that the maintenance costs must be reduced due to the paper market becoming more challenging, which is affected by the international cellulose price downfall.

Therefore, a strategic way to reduce maintenance cost is to have a predictive approach to tell the team when is best to stop the machinery and make the necessary repairs, before it presents a failure. The actual maintenance is done in a fixed frequency by calendar time, which canbe too early or too late.

By building a data-driven model, one could use it as support to see if a specific asset has a high probability of failure or how many cycles that asset has left before the failure. This could leverage the decision making about which asset must be prioritized for maintenance and thus, reducing costs of unexpected stops, for example.

## **Data Problem Definition**

Having the necessary data for the assets, there are two data problems that can be defined as the following:

- a **regression** problem to predict how many cycles each asset has left before the failure
    - the performance measure for selecting the models will be the $R^2$ score, since we do not have information about the cost / impact of the error of the model. We are interested to explain and understand why one asset have a higher RUL (remaining useful life) than others, and this is related to target's variance, which is well captured by the $R^2$. However, the mean absolute error will be also monitored, since we do not want a high bias model. A model with high bias could lead to innacurate RUL estimations.
- a **classification** problem to predict the probabilitity of an asset to have a failure after 20 cycles of operation
    - in this type of problem, we considered that a false negative (predicted no failure but the asset failed)  - which is captured by *recall score* has a higher weight than a false positive (predicted failure but the asset did not failed) - which is captured by *precision score*. However we do not want a model with high false positive rate also, because false positive lead to expenses in unnecessary maintenance. The model selection performance metric will the $F-\beta$ score, using a value of $\beta = 1.5$, to favor the recall score in the harmonic mean. **[IMPORTANT]**: the choice of $\beta$ parameter, **in this case** is purely arbitrary, but in real cases, the costs of each type of model's errors must be taken into account for selecting the parameter that represents more reliably the reality of the problem