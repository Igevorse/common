import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from common_plus.supervised_projections import get_supervised_directions, get_projections
from common_plus.outliers import get_outlier_sels, get_outlier_sels_within_classes
from common.visualize.colors import COLORS
from pylab import *


    
    
def pca_report(X,Y=None):
    '''Find principal components for the data, plot the data in first 2 compoinents and show descriptive power of principal components.'''
    D=X.shape[1]
    pca =PCA()
    X_pca = pca.fit_transform(X)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
    
    figure()
    if Y==None:
        scatter(X_pca[:,0], X_pca[:,1])
    else:
        scatter(X_pca[:,0], X_pca[:,1], c=[COLORS[y] for y in Y])
    xlabel('principal component 1')
    ylabel('principal component 2')
    title('First 2 components explain %.3f variance'%cum_explained_variance[1])
    
    figure()
    plot(range(1,D+1), cum_explained_variance)
    xticks( range(1,D+1) )
    xlabel('# components')
    ylabel('explained variance fraction')
    for threshold in [0.9, 0.95, 0.99]:
        ind = find(cum_explained_variance>threshold)[0]
        print('%.2f of variance is explained by %d components'% (threshold,ind+1))
    show()
        
        

def plot_corr(df,size=10,show_colorbar=True,show_grid=True):
    '''Function plots a graphical correlation matrix for earch pair of columns in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
        

    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.1),
                       (1.0, 1.0, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
            }

    from matplotlib.colors import LinearSegmentedColormap
    blue_red_cmap = LinearSegmentedColormap('BlueRed1', cdict)


    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    plt.set_cmap(blue_red_cmap)
    m = ax.matshow(corr,interpolation='none', vmin=-1,vmax=1) #
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    if show_colorbar is True:
        plt.colorbar(m)
    if show_grid is True:
        plt.grid(color=[0.5,0.5,0.5], linestyle=':', linewidth=1)



        
def show_importances(features, importances, max_features=20, min_importance=0, figsize=None):
    '''Shows graphically a bar plot of feature importances ordered by decreasing 
    importance labelled by names of features. Features with importances<=0 are not displayed.
    Feature importances can be calculated as clf.feature_importances_ for clf being a
    decision tree, random forest or tree based boosting.
    
    
    Author: v.v.kitov(at domain)yandex.ru'''
    
    inds = argsort(importances)
    inds = array([ind for ind in inds if importances[ind]>min_importance])[-max_features:]

    features = [features[ind] for ind in inds]
    importances = [importances[ind] for ind in inds]

    if figsize!=None:
        figure(figsize=figsize)
        
    yy = arange(len(importances))
    barh(yy, importances)
    yticks(yy+0.5, features);
    title('Feature importances')
    grid()