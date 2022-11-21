#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.graphics.gofplots import ProbPlot

'''
Modified code from Kyle Caron: 
https://github.com/kylejcaron/Mentoring/blob/master/linear_regression/Linear%20Regression.ipynb

to create the common useful residual plots built into the R regression module, i.e. plot(model).


To use, import the module after placing the file in your active directory as follows: 

from residualplots import diagnostic_plots

and then build the model in Statsmodels and run the diagnostic plots:

model = sm.OLS.from_formula("speed ~ year", data=da)
result = model.fit()
diagnostic_plots(result, y = da.speed)


'''

def resid_v_fitted(fitted_y, y, ax):
    '''Plots the residuals vs. fitted yvalues'''
    sns.residplot(x = fitted_y, y = y,
          lowess=True, scatter_kws={'alpha': 0.5},
          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},ax=ax)
                 
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')

def normal_QQ_plot(norm_residuals, ax):
    QQ = ProbPlot(norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')
    abs_norm_resid =norm_residuals.sort_values(ascending=False)
     
    
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3.index):
        ax.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                        norm_residuals[i]));

def scale_loc_plot(fitted_y, norm_residuals, norm_residuals_abs_sqrt, ax):
    ax.scatter(fitted_y, norm_residuals_abs_sqrt, alpha=0.5);
    sns.regplot(x = fitted_y, y = norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
    ax.set_title('Scale-Location')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$\sqrt{|Standardized Residuals|}$');

    # annotations
    abs_norm_resid = norm_residuals.sort_values(ascending=False)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    abs_sq_norm_resid = norm_residuals_abs_sqrt.sort_values(ascending=False)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
 
    for i in abs_norm_resid_top_3.index:
        ax.annotate(i, xy=(fitted_y[i],
                norm_residuals_abs_sqrt[i]))
        
def residuals_v_leverage_plot(norm_residuals, model_leverage,ax):
    ax.scatter(model_leverage, norm_residuals, alpha=0.5);
    sns.regplot(x = model_leverage, y = norm_residuals, scatter=False,
              ci=False, lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
    
    ax.set_xlim(0, max(model_leverage)+0.01)
    ax.set_ylim(-3, 5)
    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    
        
def diagnostic_plots(model, y, return_high_leverage=False):
    fitted_y = model.fittedvalues
    residuals = model.resid
    norm_residuals = pd.Series(model.get_influence().resid_studentized_internal, index=fitted_y.index)
    norm_residuals_abs_sqrt = pd.Series(np.sqrt(np.abs(norm_residuals)), index=fitted_y.index)
    model_abs_resid = pd.Series(np.abs(residuals), index=fitted_y.index)
    model_leverage = pd.Series(model.get_influence().hat_matrix_diag, index=fitted_y.index)
    avg_leverage = model_leverage.mean()
    model_cooks = pd.Series(model.get_influence().cooks_distance[0], index=fitted_y.index)
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    # Residuals vs. Fitted
    resid_v_fitted(fitted_y, y, ax=ax[0,0])
    # Normal QQ Plot
    normal_QQ_plot(norm_residuals, ax=ax[0,1])
    # Scale-Location
    scale_loc_plot(fitted_y, norm_residuals, norm_residuals_abs_sqrt, ax=ax[1,0])
    # Residuals vs. Leverage
    residuals_v_leverage_plot(norm_residuals, model_leverage,ax=ax[1,1])
    if return_high_leverage:
        return model_leverage[model_leverage > avg_leverage*2]

