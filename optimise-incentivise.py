import numpy as np
import pandas as pd
import gc 
#import time

from xgboost import XGBClassifier
from sklearn.externals import joblib

from scipy.optimize import newton
from scipy.optimize import minimize


#import test data
test_data = pd.read_csv('data/X_test.csv')
test_labels = pd.read_csv('data/test_labels.csv')


#import model
XGB = joblib.load('data/XGB_model.pkl')


#predict probability renewal
prob_rnl =XGB.predict_proba(test_data)


#probability of sucess
prob_rnl = prob_rnl[:,1]


inc_test = pd.DataFrame(columns=['incentive'])
incentive_max = pd.DataFrame(columns=['incentive_max'])

#optimisation routine
def f(x):
    return np.exp(-x/400)-(x/400)-2.1512+ln_Pr


for i in range(len(test_data)):
    Pr = test_data.premium.iloc[i]*prob_rnl[i]
    ln_Pr = np.log(Pr)
    root = newton(f, (Pr/ln_Pr), maxiter=75, fprime=lambda x: (-1*np.exp(-x/400)/400 - (1/400)), fprime2=lambda x: (np.exp(-x/400))/160000)
    inc_test = np.append(inc_test, root)


def inc_opt(x):
    return x-(1+20*(1-np.exp(-2*(1-np.exp((-1*x)/400))))*P)

def inc_opt_der(x):
    return (1-((P/10)*np.exp(2*np.exp((-1*x)/400)-(x/400)-2)))

def hess(x):
    h=(P/4000)*(1+2*np.exp(2*np.exp((-1*x)/400)-((-1*x)/400)))
    h=h.reshape((1,1))
    return h


for i in range(len(test_data)):
    P = test_data.premium.iloc[i]*prob_rnl[i]
    maxim = minimize(inc_opt, [inc_test[i]], method='trust-krylov', jac=inc_opt_der, hess=hess)
    incentive_max = np.append(incentive_max, maxim.x)
    if i%2500==0:
        print(i,"th  step and time :",time.time()-ti)

#export renewal probability + incentives
Final_incentive_opt = pd.DataFrame({'renewal':prob_rnl, 'incentives':incentive_max})
Final_incentive_opt.to_csv('data/Final_incentive_opt.csv', index=False)
