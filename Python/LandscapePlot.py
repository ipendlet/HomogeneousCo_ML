import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    pass
    dataFile = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/AllData_CH3CN.csv')
#    dataFile2 = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/THFSolventEnergiesPCM.csv')
#    dataFile3 = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/ModelFigure.csv')
#    outFileName = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2vsHyd')
#    outFileName2 = Path('D:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/Hydvspka')
    pkalabel = '$\mathregular{pK_{a}}$'
    hydricitylabel = 'Hydricity (kcal/mol)'

#    CH3CN = np.loadtxt(dataFile1,delimiter=',',skiprows=0,usecols=(0,1,2))
    CH3CN = np.loadtxt(dataFile,delimiter=',',usecols=(0,1,2))
    THF = np.loadtxt(dataFile,delimiter=',',usecols=(0,1,2))
#    Lig = np.loadtxt(dataFile3, delimiter=',',usecols=(0,1))
#    THF = np.loadtxt(dataFile,delimiter=',',skiprows=1,usecols=(4,5,6))

    pkavalue= CH3CN[:,0]
    hydvalue= CH3CN[:,1]
    h2value= CH3CN[:,2]

    pkavalue1= THF[:,0]
    hydvalue1= THF[:,1]
    h2value1= THF[:,2]

    predictors = np.column_stack((nlmo)).reshape((-1,1))
    hydricities = (therm[:,1])
    regressor.fit(predictors, hydricities)
    predictions = regressor.predict(predictors)
    print(regressor.score(predictors, hydricities))
    print(regressor.steps[2][1].intercept_)

#    if hydricities[0]==therm[0,0]:
#        scatterPlot.pka(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/PkaPrediction'))
#        print('pKa')
#    if hydricities[0]==therm[0,1]: 
#        scatterPlot.Hyd(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/HydricityPrediction'))
#        print('Hydricity')
#    if hydricities[0]==therm[0,2]:
#        scatterPlot.h2binding(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/H2Binding'))
#        print('H2Binding')
#    else:
#        scatterPlot.other(hydricities, predictions, Path('E:/' '/GoogleDrive' '/Desktop' '/ML_Figures' '/OtherPrediction'), OtherPlotVar)
#        print('plotted with pka stuffs,',OtherPlotVar)


## for h2binding versus hdricity plot
def h2binding(outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(hydvalue1, h2value1, s=20, color='#000000')
    plt.xlim(-20,60)
    plt.ylim(0,100)
#    plt.show()
    plt.scatter(hydvalue, h2value, c=pkavalue, cmap='jet_r', s=20, vmin=-10, vmax=50)
    plt.xlim(0,100)
    plt.ylim(-50,30)
    clb = plt.colorbar()
    axes = plt.gca()
    plt.xlabel(hydricitylabel, fontsize=14)
    plt.ylabel('$\mathregular{H_{2}}$ Binding Energy', fontsize=14)
    clb.set_label(pkalabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

##for hydricity vs. pka plot
def hydricity(outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(pkavalue1, hydvalue1, s=20, color='#000000')
    plt.xlim(-20,60)
    plt.ylim(0,100)
#    plt.show()
    plt.scatter(pkavalue, hydvalue, c=h2value, cmap='jet_r', s=20, vmin=-30, vmax=20)
    plt.xlim(-20,60)
    plt.ylim(0,100)
    clb = plt.colorbar()
    axes = plt.gca()
    plt.xlabel(pkalabel, fontsize=14)
    plt.ylabel(hydricitylabel, fontsize=14)
    clb.set_label('$\mathregular{H_{2}}$ Binding Energy', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

##def h2binding(outFileName):
#    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
#    plt.scatter(hydvalue, h2value, s=20)
#    plt.xlim(-15,5)
##    plt.ylim(-50,30)
#    clb = plt.colorbar()
#    axes = plt.gca()
#    plt.xlabel(hydricitylabel, fontsize=14)
#    plt.ylabel('$\mathregular{H_{2}}$ Binding Energy', fontsize=14)
#    clb.set_label(pkalabel, fontsize=14)
#    plt.tight_layout()
#    plt.savefig(str(outFileName) + '.png')
#    plt.clf()

#h2binding(outFileName)
hydricity(outFileName2)
