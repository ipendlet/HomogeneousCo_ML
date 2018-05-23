'''
Created on Oct 12, 2017
'''

import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import main
import scatterPlot
from sklearn import preprocessing
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import scale
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from scipy import stats


if __name__ == '__main__':
    pass
#    dataFile = Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/AllData_CH3CN.csv')
    dataFile = Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/AllData_CH3CN_Alkyl_NLMO.csv')


########### Features ###############
    name = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(0))
#Cobalt Charges
    NBO = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(29,30,31))
    lc = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(16,17,18))
    lcH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(78,79,80))
    mcH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(81,82,83))
    mc = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(13,14,15))
#Phosphorous Charges
    lcPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(56,57,58,59))
    lcPH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(60,61,62,63))
    lcPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(64,65,66,67))
    mcPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(44,45,46,47))
    mcPH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(48,49,50,51))
    mcPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(52,53,54,55))
    NBOPH2 = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(32,33,34,35))
    NBOPH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(36,37,38,39))
    NBOPnoH = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(40,41,42,43))

#Steric Properties
   #Buriedvolume
    bv = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(23,26))
   #tau
    tau = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(11,12))
   #surface area :q
    sa = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(19,21))
    sv = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(20,22))
   #biteangle
    ba = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(84,85,86,87,88,89))
    baa = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(102))
   #Ligand Torsion
    lt = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(90,91,94,95,98,99))
   #Ligand Bond Length -not really steric, but,... whatever
    lb = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(92,93,96,97,100,101))

#Key Thermodynamic Properties
    therm = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(8,9,10))
#PKA stuffs
    CoHOMO = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(68,70,72))
    CoLUMO = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(69,71,73))
    CoH2Len = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(74,75,76,77)) ##76 is CoH and 77 is angles


#Main Figures if wanting to reproduce
#kitchen Sink
    predictors = np.column_stack((lc[:,0],lc[:,1],lc[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))



# CoH2 and CoH
#    predictors = np.column_stack((therm[:,0],NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
# CoH2 and CoH
# CoH2 and CoH
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],tau[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],baa))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],CoHOMO[:,0],CoHOMO[:,1]))
#    predictors = np.column_stack((CoHOMO[:,0],CoHOMO[:,1]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],tau[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],baa))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],CoHOMO[:,0],CoHOMO[:,1]))
#    predictors = np.column_stack((CoHOMO[:,0],CoHOMO[:,1]))


#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2]))
#    predictors = np.column_stack((NBO[:,1],lcH[:,1],bv[:,0],bv[:,1],CoHOMO[:,1],CoLUMO[:,1],CoH2Len[:,2],ba[:,2],ba[:,3]))


### Wide spread (pka range of >10 units)
#Curated - For all tests see the list of predictors saved at the bottom of this file along with ppt notes referenced
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],ba[:,0],ba[:,1],lt[:,0]))

##final iterations of pKa plots (constrained variation of pka)
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],ba[:,0],ba[:,1],lt[:,0]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2]))

    hydricities = (therm[:,1])
#    predictors = np.column_stack((baa,lb[:,0],lb[:,1],sa[:,0],sa[:,1],sv[:,0],sv[:,1]))
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
# CoH2 and CoH


#Steric Relationships
#    predictors = np.column_stack((NBO[:,0],NBO[:,1],NBO[:,2],lcH[:,0],lcH[:,1],lcH[:,2],tau[:,0],tau[:,1],sa[:,0],sa[:,1],bv[:,0],bv[:,1],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],CoLUMO[:,2],CoH2Len[:,0],CoH2Len[:,1],CoH2Len[:,2],CoH2Len[:,3],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,3],lt[:,4],lt[:,5],lb[:,0],lb[:,1],lb[:,2],lb[:,3],lb[:,4],lb[:,5]))
#    predictors = np.column_stack((baa,sa[:,0],sa[:,1],bv[:,0],bv[:,1],lt[:,0],lt[:,1],lb[:,0],lb[:,1]))
#    predictors = np.column_stack((lb[:,0],lb[:,1]))

#selecting based on best performance
#    predictors = np.column_stack((NBO[:,0],sa[:,0],sa[:,1],bv[:,0],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],ba[:,0],ba[:,1],ba[:,2],ba[:,3],ba[:,4],ba[:,5],lt[:,0],lt[:,1],lt[:,2],lt[:,4],lt[:,5]))

#constraining to just a single complex
#    predictors = np.column_stack((NBO[:,0],sa[:,0],sa[:,1],bv[:,0],CoHOMO[:,0],CoHOMO[:,1],CoHOMO[:,2],CoLUMO[:,0],CoLUMO[:,1],ba[:,0],ba[:,1],lt[:,0],lt[:,1]))
#    predictors = np.column_stack((CoHOMO[:,1],CoHOMO[:,1]))
#    predictors = np.column_stack((therm[:,1],therm[:,1]))







###Paper walkthrough ###





#######Training targets  ###
#    hydricities = CoHOMO[:,1]
#    hyduns = np.column_stack((therm[:,1])).reshape((-1,1))
#    hyduns = np.column_stack((therm[:,1])).reshape((-1,1))
#    scaler = StandardScaler()
#    hydricities2 = hydricities1.reshape((-1,1))
#    hydricities=scale(hydricities2)
#    print(hyd1)


    # compound features
    polyFeatures = PolynomialFeatures(degree=1,interaction_only=True)
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=60000, cv=KFold(n_splits=5, shuffle=True)))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(max_iter=60000))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Ridge())
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Lasso(alpha=0.05, max_iter=70000))#, fit_intercept=True))
#    regressor = make_pipeline(polyFeatures, StandardScaler(), Lasso(alpha=0, max_iter=70000))#, fit_intercept=True))
    regressor = make_pipeline(polyFeatures, StandardScaler(), LinearRegression())
#    regressor = RandomForestRegressor(oob_score=True,n_estimators=2000)
    





####Make the output, print statements with detailed analysis of each step ##### 
    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
 #   print('Intercept:',regressor.steps[2][1].intercept_)
#    print((regressor.steps[2][1].coef_).reshape((-1,1)))


### Print Functions
#    cv = ShuffleSplit(n_splits=9, test_size=0.1, random_state=None) 
#    scores=cross_val_score(regressor,predictors,hydricities,cv=cv)
#    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
#    print(scores)
    count=0
#    count2=0
#    for item in regressor.predict(predictors):
#        print(item, name[count2])
#        count2+=1
#    print('Intercept:',regressor.steps[2][1].intercept_)
#    print('LassoOptIter:',regressor.steps[2][1].n_iter_)
    fn=np.asarray(polyFeatures.get_feature_names())
#    print(regressor.steps[2][1].sparse_coef_)
#    scaler=regressor.steps[1][1].scale_
    for item in regressor.steps[2][1].coef_:
#        print(fn[count], item, scaler[count])#,polyFeatures.get_feature_names())
        print(fn[count], item)#,polyFeatures.get_feature_names())
#        print(item)#,polyFeatures.get_feature_names())
        count+=1
#        count+=1

#    print(regressor.steps[2][1])
#    print(cross_val_score(predictors, hydricities))
#    predictions2 = predictions3
#    print(len(predictions2), len(hydricities))

#    print('Alpha',regressor.steps[2][1].alpha_)
    print(regressor.steps[2][1].intercept_)
 #   print('Intercept:',regressor.steps[2][1].intercept_)

 #   predictions3 =(regressor.predict(predictors))
    print(regressor.score(predictors, hydricities))
 #   predictions2 = predictions3.reshape((-1,1)) 



#    regressor2 = LinearRegression()
#    regressor2.fit(predictors, hydricities)
#    predictions2 = predictions.reshape((-1,1))
#    predictions = regressor2.predict(predictors)
#    print('R^2(2): ', regressor2.score(predictions2, hydricities))
#    print("scalar=",regressor[2][1].coef_, "intercept=", regressor[2][1].intercept_)
#    predictions = regressor2.predict(predictions2)
 


#    print('R^2: ', regressor.score(predictors, hydricities))
#    print(regressor2.predict(predictors))
#    print(predictions)

#    slope, intercept, r_value, p_value, std_err = stats.linregress(hydricities,predictions)
#    print("linear Regression", r_value**2)


#### Graphing the model versus prediction section ####
    if hydricities[0]==therm[0,0]:
        scatterPlot.pka(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/PkaPrediction'))
        print('pKa')
    if hydricities[0]==therm[0,1]: 
        scatterPlot.Hyd(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/HydricityPrediction'))
        print('Hydricity')
    if hydricities[0]==therm[0,2]:
        scatterPlot.h2binding(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2Binding'))
        print('H2Binding')
    else:
        scatterPlot.pka(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/OtherPrediction'))
        print('plotted with pka stuffs')

#    print(Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/ianPredictions')) 
#    regressor = make_pipeline(polyFeatures, StandardScaler(), LassoCV(eps=1e-3, max_iter=60000, cv=KFold(n_splits=3, shuffle=True),tol=1e-10,selection='random'))