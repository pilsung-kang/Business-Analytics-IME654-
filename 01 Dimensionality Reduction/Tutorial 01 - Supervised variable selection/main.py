import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from BA_tutorial.variable_selection import Variable_selection
from BA_tutorial.Genetic_Algorithm import Genetic_algorithm

####################### Prepare dataset ############################
# import data
from sklearn.datasets import load_boston
boston = load_boston()

# divide to input and target data
# remove categorical variable part
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
input_data = np.array(data)

# target : price
target_data = np.array(boston.target)

# regression model
reg = lm.LinearRegression()

#################################### Forward, Backward, Stepwise #######################################
selection_tech = Variable_selection(model=reg,input_data=input_data,target_data=target_data)

# forward selection
var_fw = selection_tech.forward_selection(alpha=0.1)
var_fw

names = boston.feature_names
selected_names = [names[i] for i in var_fw]
selected_names

Rsq_fw,adj_Rsq_fw = selection_tech.R_sq(model=reg,X=np.take(input_data,var_fw,axis=1),Y=target_data)

# backward elimination
var_bw = selection_tech.backward_elimination(alpha=0.1)
var_bw

names = boston.feature_names
selected_names = [names[i] for i in var_bw]
selected_names

Rsq_bw,adj_Rsq_bw = selection_tech.R_sq(model=reg,X=np.take(input_data,var_bw,axis=1),Y=target_data)

# stepwise selection
var_st = selection_tech.stepwise_selection(alpha=0.1)
var_st

names = boston.feature_names
selected_names = [names[i] for i in var_st]
selected_names

Rsq_st,adj_Rsq_st = selection_tech.R_sq(model=reg,X=np.take(input_data,var_st,axis=1),Y=target_data)

###################################### Genetic Algorithm #####################################
# make class
Genetic_al = Genetic_algorithm(model=reg,X=input_data,Y=target_data,chrom_num=10,eval_metric='adj_Rsq',chrom_ratio=0.5)
var, eval = Genetic_al.Do_GA(max_len=100)
