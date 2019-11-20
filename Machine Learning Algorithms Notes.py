#!/usr/bin/env python
# coding: utf-8

# ### Machine Learning Algorithms 

# Context

# In[ ]:


TREE


# In[ ]:


KNN


# In[ ]:


REGRESSION


# In[5]:


import pandas as pd
movie_df=pd.read_csv('/Users/wty24/Desktop/2019SpringTerm/758T Data Mining and Predictive Analytics/Assignment/movies_data.csv')
movie_df.head()


# In[6]:


#process with categorical variables
X=movie_df.iloc[:,2:-1]
y=movie_df.iloc[:,-1]
for col in ['genre','united_states','english','title_change']:
    dummy=pd.get_dummies(X[col],drop_first=True,prefix=col)
    X=X.drop(col,axis=1)
    X=pd.merge(X,dummy,right_index=True,left_index=True)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[7]:


X_train.head()


# ### Tree

# In[52]:


#CART classification and regression tree
def impurity(x):
    a=x**2+(1-x)**2
    return 1-a


# In[7]:


#GINI INDEX
#range~(0,0.5)
#before split
impurity(37/71)


# In[9]:


#after split
0.5*(impurity(1/7)+impurity(7/36))


# In[8]:


from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
#通过GridSearchCV/RandomizedSearchCV, find the best hyperparameter combination, ex: {'max_depth':6,'max_features':5...}

# Fit it to the data
tree_cv.fit(X_train,y_train) #generate the model

tree_cv.best_estimator_.predict(X_test)

#get accuracy
tree_cv.score(X_test,y_test)


# training data
# validation data(prune) avoid overfitting pick k(terminal nodes)
# testing data

# Pro: easy to explain
# Con: computationally expensive
#     large datasets

# In[16]:


#Regression Tree
#impurity use SSR(to make a split)!!!
#parent node: SSR0
#children node: W1*SSR1+W2*SSR2 小，继续split


# In[ ]:


SSR=sum((Xi-avg(X))**2)
SSE=sum((Xi-Xp)**2)


# In[9]:


# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor

# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
dt.fit(X_train, y_train)


# ### KNN k-nearest-neighbors

# training data (labels)
# valid--pick K(K=5)
# 
# predictive model:
# training data: total_train=train+valid
# K=5

# In[9]:


#how to decide new label?
1/3*(0+0+1)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
#model fitting
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) #generate the model
#model predicting
y_pred=knn.predict(X_test)
knn.score(X_test,y_test)


# ### Split Traing and Testing data
# 

# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3,random_state=21)


# ### Regression

# In[30]:


import sklearn.datasets as ds
import pandas as pd


# In[34]:


#linear regression
#OLS: sum squre of residual 
    
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)#generate the model
#y_pred = reg.predict(X_pred)

# Print R^2 
print(reg.score(X_test,y_test))

# Plot regression line
import matplotlib.pyplot as plt
plt.plot(X_test, y_test, color='black', linewidth=3)
plt.show()


# In[ ]:


#R square
R**2=SSR/(SSR+SSE)
if R square==1:
    SSE=0# R squrare can only be compared when the number of variables are same

#Adj R square
adj R**2=1-(SSE/(n-k-1))/(SST/(n-1))
#if add another variable, both (n-k-1) and SSE go down, if this variable is not so related, then
#(n-k-1) goes down more, adj R square goes down


# ### Cross Validation

# In[ ]:


#validation used for pick parameter within the model(knn/tree)
#有了在指定参数下的model（knn中选定k=3），再测在validation上的average accuracy
#!!!最后选的是model的结构:
    linear regression(how many x involved)
    tree(how many terminal nodes)...

cv=5(number of folds)
k=3:(acc1+acc2+...+acc5)/5
k=4:(acc1+acc2+...+acc5)/5
Pro: avoid relying on train/test split
Con: too much computation


# In[ ]:


#code
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

reg = LinearRegression()

cv_scores = cross_val_score(reg,X,y,cv=5)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# ### Ridge/Lasso Regression

# In[35]:


import numpy as np


# In[ ]:


#avoid parameter too big
Ridge: Loss Function = OLS Loss Function + alpha*sum((ai**2)) #each row!

#Lasso 可以把不重要的变量，coefficient变为0，作为high dimension中选取关键变量
Lasso: Loss Function = OLS Loss Function + alpha*sum(np.abs(ai))  #feature selection, as unimportant ai can turn to 0


# In[36]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1,normalize=True)
lasso.fit(X_train,y_train)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)


# In[25]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)   #不同alpha值
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge,X,y,cv=10) #同一个alpha下，10个值
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_scores))


# In[32]:


#Pros:
    #feature selection
#Cons:
    #bias and variance trade off(We lose some kind of bias but most of time we get more convincible range.)
from IPython.display import Image
PATH = "/Users/wty24/Desktop/2020 Fulltime Data Scientist!/MLscreenShot/"
Image(filename = PATH + "/Bias and Variance Trade Off.png", width=300, height=200)


# ### Logistic Regression

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)#generate the model

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

n = 1000
ls = np.linspace(0, 2*np.pi, n)
df1Clean = pd.DataFrame(np.sin(ls))
df2Clean = pd.DataFrame(2*np.sin(ls+1))
dfClean = pd.concat([df1Clean, df2Clean], axis=1)
### dfDirty = dfClean+0.00001*np.random.rand(n, 2)

grangercausalitytests(dfClean, maxlag=20, verbose=False)    # Raises LinAlgError
grangercausalitytests(dfDirty, maxlag=20, verbose=False)    # Runs fine


# In[ ]:


### stats package
import statsmodels.formula.api as sm
 
model = sm.Logit(y, X)
result = model.fit()
result.summary()


# In[ ]:


#sort by p-values!
table=result.summary2().tables[1]
table.sort_values(by='P>|z|')


# In[34]:


#maths
#odds=p/(1-p)=e**(bo+b1*x1+b2*x2+...+bn*xn)


# ### Model Performance 

# In[39]:


#except accuracy, to solve imbalance data
#TPR=tp/(tp+fn) (actual y=1)
#实际垃圾邮件20封/9980/true cancer patients be predicted
#（y=1) tpr=19/20


# In[1]:


#tpr--tnr trade off
#while cut-off goes down,tpr up,tnr down(0,1)


# ### Other Metrics(Precision/Recall/AUROC)

# In[ ]:


#precision/recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print('precision score is: '+str(precision_score(y_test,y_pred)))
print('recall score is: '+str(recall_score(y_test,y_pred)))


# In[ ]:


#auroc
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

ns_probs = [1 if np.average(y_test)>=0.5 else 0 for _ in range(len(y_test))]#common class

ns_auc = roc_auc_score(y_test,ns_probs)
lr_auc = roc_auc_score(y_test,y_pred)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test,ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test,y_pred)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# Used as Model Selection for classification problem
# Compare AUC among different models(regardless the cut-off/threhold)
# I think in this way, we can focus on the model itself.

# In[23]:


from IPython.display import Image
PATH = "/Users/wty24/Desktop/2020 Fulltime Data Scientist!/MLscreenShot/"
Image(filename = PATH + "ROCAUC.png", width=400, height=200)

#TPR/FPR
#TN FP --FP/(TN+FP)
#FN TP --TP/(TP+FN)

#when threhold goes down--> move the top right corner(predict all the record to one)
#top left corner is the perfect classifier


# In[24]:


#AUC focus on the algorithm itself instead of the threhold.
#Model Selection
#AUC is classification-threshold-invariant. 
#It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

#reference:https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc


# ### Get Dummies

# In[ ]:


df_region = pd.get_dummies(df,drop_first=True)
#include either numeric/categorical in the df


# ### Data Preprocessing

# In[25]:


#Missing values(drop/impute average)
#Scaling(normalizing)--knn


# ### PipeLine

# In[39]:


# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline,parameters,cv=3)

# Fit to the training set
gm_cv.fit(X_train,y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)


# ### Ensemble Methods

# In[ ]:


D Tree/L Regression/kNN/ Others
每种model一个说法，meta model综合，给出结果


# In[ ]:


Bootstapping:
    1   3    2    1
    1   3    3    2
    3   3    3    2
Bagging:
    Use bootstrap sample(100)--run 100 tree--combine 100 results
Random Forest:(m.n=1000)
    Each split only consider m variables(p所有变量**0.5) 
    #if there is a good predictor, force to use other variables
    #in this way, all these trees looks different, DO REAL ENSEMBLE!!!


# In[141]:


# Set seed for reproducibility
SEED=1
# Instantiate each model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

from sklearn.ensemble import VotingClassifier
# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     
vc.fit(X_train, y_train)   #每个model都已经各自train好
y_pred = vc.predict(X_test)

# Calculate accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# ### Bagging

# In[42]:


#bagging and oob code
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=500,oob_score=True,random_state=1)
#oob_score=True Onaverage,每一次boostramp只会用到63% total data，将剩余37%unseen作为testing，only test at 'that' tree,
#get accuarcy of all the trees
bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

# Evaluate accuracy
acc_test = accuracy_score(y_pred, y_test)
acc_oob = bc.oob_score_
acc_test


# ### Random Forest

# In[168]:


#different from Bagging because force the structure of tree to be different
#rf code
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=25,random_state=2) 
rf.fit(X_train, y_train) 
y_pred=[1 if y>0.5 else 0 for y in rf.predict(X_test)]
accuracy_score(y_pred,y_test)


# In[170]:


#Tree 可以tell feature的重要性，哪些factor重要 by how much the impurity decrease
# Draw a horizontal barplot of importances_sorted
importances = pd.Series(data=rf.feature_importances_,index= X_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# ### Boosting

# In[47]:


#https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/


# ### Ada Boosting

# In[ ]:


#It's sequential!连续的几棵树
only put more focus on previous predictor wrong instance, increase the weight of wrong instance
weights增加，改变的是loss function


# In[177]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=3, random_state=1)
# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=1000, random_state=1)
ada.fit(X_train,y_train)

y_pred=ada.predict(X_test)
accuracy_score(y_pred,y_test)


# ### Gradient Boosting

# ·Gradient Boosting
# ·SGD
# ·XGB

# In[ ]:


#y_pred=y1+nr1+nr2+nr3+...
#keep making up the previous residual 不断补上参差，来弥补，每次都预测y，r1，r2, r3...--> aggregate
Con: feature may simliar in each tree


# In[ ]:


#over overfitting:
# constraints on tree size
#add learning rate to residue
#SGD/subsample(take only part of the data)
#


# In[184]:


from sklearn.ensemble import GradientBoostingClassifier

# Instantiate gb
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,random_state=2)
gb.fit(X_train,y_train)

y_pred = gb.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:


SGD
Stochastic Gradient Boosting#区别于GD，只选部分training records和features(without replacement，一次性取出)
1.X_train (part of the whole rows) #without replacement
2.feature(part of all features) #without replacement


# In[193]:


from IPython.display import Image
PATH = "/Users/wty24/Desktop/2020 Fulltime Data Scientist!/MLscreenShot/"
Image(filename = PATH + "SGD.png", width=800, height=500)
#material source: DataCamp


# In[196]:


#sgd code
from sklearn.ensemble import GradientBoostingClassifier
# Instantiate sgbr
sgbr = GradientBoostingClassifier(max_depth=4, subsample=0.9,max_features=0.75,n_estimators=200,random_state=2)
sgbr.fit(X_train,y_train)
y_pred=sgbr.predict(X_test)

acc=accuracy_score(y_pred,y_test)
acc


# In[46]:


from IPython.display import Image
PATH = "/Users/wty24/Desktop/2020 Fulltime Data Scientist!/MLscreenShot/"
Image(filename = PATH + "Bias and Variance.png", width=800, height=500)
#material source: DataCamp


# In[13]:


#XGB
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, seed=123)#reg:squarederror

xg_cl.fit(X_train,y_train)

preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# In[14]:


from IPython.display import Image
PATH = "/Users/wty24/Desktop/2020 Fulltime Data Scientist!/MLscreenShot/"
Image(filename = PATH + "XGB.png", width=800, height=500)
#material source: DataCamp


# ### Tune Hypeparameter

# In[ ]:


#GridSearchCV
from sklearn.model_selection import train_test_split
X_train,X_test,y_two_train,y_two_test=train_test_split(X,y_two,test_size = 0.25, random_state=42)

from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_dist = {"max_depth": [3, 20],
              "max_features": np.arange(1,20),
              "min_samples_leaf": np.arange(1,20),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X_train,y_two_train) 
y_pred=tree_cv.best_estimator_.predict(X_test)

#tree_cv.score(X_test,y_two_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_two_test,y_pred)


# In[203]:


#tuning hyperparameter 就是调model自己structure:max_depth,n_estimator,max_leaf_nodes...
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
params_rf = {
    'n_estimators':[100,350,500,1000],
    'max_features':['log2','auto','sqrt'],
    'min_samples_leaf':[2,5,10,30]} #set the parameter you would like to tune

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,param_grid=params_rf,scoring='accuracy',cv=3,verbose=1,n_jobs=-1)
grid_rf.fit(X_train,y_train)  #得出each grid metric，做比较
best_model=grid_rf.best_estimator_ #get the best hyperparameter
y_pred=best_model.predict(X_test)
accuracy_score(y_test,y_pred)


# ### Deep Learning Notes

# In[ ]:


w1x1+w2x2+...+wnxn+b(bias)
activation function:
    sigmoid:y=1/(1+e**(-x))
#One hidden layer network can represent any continuous function
#how to update parameter?
Gradient Descent:w1=w1-lr*l/w


# In[ ]:




