# Insurance-data project 
Predict the propensity to pay renewal premium and build an incentive plan for its agents to maximise the net revenue 



### Exploratory Data Analysis
The EDA.ipynb file has basic exploratory data analysis along with some feature engineering. 

### Predictive Modelling
The propensity has been calculated as the probabilty of receiving a premium on a policy. XGBoost model has been used for predictive analysis. Hyperparamter tuning  done by Random GridSearch. 

### Optimising Net Revenue
The net or total revenue across all policies are calculated in the following manner:

![img](http://latex.codecogs.com/svg.latex?Total%5C%2CRevenue%3D%5Csum_%7Bacross%5C%2Call%5C%2Cpolicies%7D%28p_%7Bb%7D%2B%5CDelta%5C%2Cp%29%2Apremium%5C%2Con%5C%2Cpolicy-Incentives%5C%2Con%5C%2Cpolicy%29)

where, 
* $$p_b$$ is the renewal probability predicted using a benchmark model by the insurance company
* ∆p (% Improvement in renewal probability*$$p_b$$ ) is the improvement in renewal probability calculated from the agent efforts in hours
* ‘_Premium on policy_’ is the premium paid by the policy holder for the policy in consideration
* ‘_Incentive on policy_’ is the incentive given to the agent for increasing the chance of renewal (estimated by assumption curves mentioned below) on each policy

Optimisation of ‘_Total Revenue_’ performed using the `SciPy` optimize libray


**Assumptions for _effort vs incentives_ and _improvement in renewal prob vs effort_ relationships**

* Equation for the effort-incentives curve: $$Y = 10*(1-\exp(-X/400))$$

* Equation for the % improvement in renewal prob vs effort curve:
$$Y = 20*(1-\exp(-X/5))$$
