# Insurance_data
predict the propensity to pay renewal premium and build an incentive plan for its agents to maximise the net revenue 

### Exploratory Data Analysis
The EDA.ipynb file has basic exploratory data analysis along with some feature engineering. 

###Predictive Modelling
The propensity has been calculated as the probabilty of receiving a premium on a policy. XGBoost model has been used for predictive analysis. Hyperparamter tuning  done by Random GridSearch. 

###Optimising Net Revenue
The net revenue across all policies are calculated in the following manner:
![img]( http://latex.codecogs.com/svg.latex?Total\,Revenue=\sum_{across\,all\,policies}(p_{b}+\Delta p)*premium\,on\,policy - Incentives\,on\,policy )

where, 
* pb is the renewal probability predicted using a benchmark model by the insurance company
* ∆p (% Improvement in renewal probability*p benchmark ) is the improvement in renewal probability calculated from the agent efforts in hours
* ‘Premium on policy’ is the premium paid by the policy holder for the policy in consideration
* Incentive on policy’ is the incentive given to the agent for increasing the chance of renewal (estimated by the participant) on each policy


**Assumptions _effort-incentives_ and _improvement in renewal prob vs effort_ relationships**

Equation for the effort-incentives curve: Y = 10*(1-exp(-X/400))

Equation for the % improvement in renewal prob vs effort curve:
Y = 20*(1-exp(-X/5))
