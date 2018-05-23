import matplotlib.pyplot as plt
from sklearn.metrics.regression import r2_score
from sklearn.metrics import r2_score
import numpy as np


def Hyd(actual, predicted, nlmos, slope, yint, outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    xmin=-0.17
    xmax=-0.38
    ymin=[xmin*slope+yint]
    ymax=[xmax*slope+yint]
    print(ymin, ymax)

    plt.scatter(nlmos, actual, s=7, color='#4b9da6')
    axes = plt.gca()
    # make plot square with equal x and y axes
#    bounds = [min(list(actual) + list(predicted) + [0])-1, max(list(actual) + list(predicted))+1]
    boundsx = [min(list(nlmos)), max(list(nlmos))]
    boundsy = [max(list(actual)), min(list(actual))]
    print(boundsx[0],boundsx[1])
#    plt.axis(boundsy * 2)
    axes.set_aspect(aspect=1/(280), adjustable='box')
#    plt.axis('tight')
    # plot the identity for visual reference (10% darker than data)
    plt.plot([xmin, xmax], [ymin, ymax], color='#d95d41')
    rSquared = r2_score(actual, predicted)
#    print(rSquared)
    plt.figtext(0.4,0.15,'$R^2 = $'+format(rSquared,'.3f'), fontsize=12)
    plt.ylabel('QM Calculated $\mathregular{pK_{a}}$', fontsize=14)
    plt.xlabel('Co-H NLMO Energy (eV)', fontsize=14)
    plt.title('QM Calculated $\mathregular{pK_{a}}$ vs. Co-H NLMO Energy', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

def h2binding(actual, predicted, outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(actual, predicted, s=7, color='#4b9da6')
    axes = plt.gca()

    # make plot square with equal x and y axes
    bounds = [min(list(actual) + list(predicted) + [0])-1, max(list(actual) + list(predicted))+1]
    plt.axis(bounds * 2)
    axes.set_aspect('equal', adjustable='box')
    # plot the identity for visual reference (10% darker than data)
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], color='#d95d41')
    rSquared = r2_score(actual, predicted)
    plt.figtext(0.6,0.15,'$R^2 = $'+format(rSquared,'.3f'), fontsize=12)
    plt.xlabel('QM Calculated $\mathregular{H_{2}}$ Binding Energy', fontsize=14)
    plt.ylabel('Model Predicted $\mathregular{H_{2}}$ Binding Energy', fontsize=14)
    plt.title('Model Predicted vs. QM Calculated $\mathregular{H_{2}}$ Binding', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()


def pka(actual, predicted, yadjust, outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(actual, predicted, s=7, color='#4b9da6')
    axes = plt.gca()
    # make plot square with equal x and y axes
    bounds = [min(list(actual) + list(predicted))-1, max(list(actual) + list(predicted))+1]
    plt.axis(bounds * 2)
    axes.set_aspect('equal', adjustable='box')

    # plot the identity for visual reference (10% darker than data)
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], color='#d95d41')

    rSquared = r2_score(actual, predicted)
#    print(rSquared)
    plt.figtext(0.6,0.15,'$R^2 = $'+format(rSquared,'.3f'), fontsize=12)
    plt.xlabel('QM Calculated $\mathregular{pK_{a}}$', fontsize=14)
    plt.ylabel('Model Predicted $\mathregular{pK_{a}}$', fontsize=14)
    plt.title('Model Predicted vs. QM Calculated $\mathregular{pK_{a}}$', fontsize=14)
#    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

def other(actual, predicted, outFileName, thing):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(actual, predicted, s=7, color='#4b9da6')
    axes = plt.gca()

    # make plot square with equal x and y axes
    bounds = [min(list(actual) + list(predicted))-1, max(list(actual) + list(predicted))+1]
    plt.axis(bounds * 2)
    axes.set_aspect('equal', adjustable='box')

    # plot the identity for visual reference (10% darker than data)
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], color='#d95d41')

    rSquared = r2_score(actual, predicted)
#    print(rSquared)
    plt.figtext(0.6,0.15,'$R^2 = $'+format(rSquared,'.3f'), fontsize=12)
    plt.xlabel('QM Calculated %s' %thing, fontsize=14)
    plt.ylabel('Model Predicted %s' %thing, fontsize=14)
    plt.title('Model Predicted vs. %s' %thing, fontsize=14)
#    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()

