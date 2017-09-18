import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from common_plus.outliers import get_outlier_sels, get_outlier_sels_within_classes
from common.visualize.colors import COLORS
from pylab import *


def pca_explained(X):
    '''Shows how much variance is explained by each number of principal components.
    Features are automatically standardized to zero mean and unit variance. Constant features are removed.
    
    X: matrix [n_objects x n_features]'''
    
    const_features_sels = (X.std(0)==0)
    X=X[:,~const_features_sels]  # remove constant features
    
    X = (X-X.mean(0))/X.std(0)  # data standardization
    
    D=X.shape[1]
    pca = PCA()
    X_pca = pca.fit_transform(X)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
	    
    figure()
    plot(range(1,D+1), cum_explained_variance)
    xticks( range(1,D+1) )
    xlabel('# components')
    ylabel('explained variance fraction')
    for threshold in [0.9, 0.95, 0.99]:
        ind = find(cum_explained_variance>threshold)[0]
        print('%.2f of variance is explained by %d components'% (threshold,ind+1))
    show()
        
        
        
    
def pca_2D(X, Y=None, task=None, cm=None, point_size=10, figsize=None):
    '''Display data in the space of first two principal components.
    Data is internally standardized to zero mean and unit variance. Constant features are removed.
    
    X: matrix [n_objects x n_features]
    Y: vector of outputs [n_objects]
    task: None or 'classification' or 'regression'
    cm: matplotlib colormap for color display for regression
    '''
    
    
    D=X.shape[1]
    pca = PCA()
    
    const_features_sels = (X.std(0)==0)
    X=X[:,~const_features_sels]  # remove constant features
    
    X = (X-X.mean(0))/X.std(0)  # data standardization
    
    X_pca = pca.fit_transform(X)
    cum_explained_variance = cumsum(pca.explained_variance_ratio_)
	
    if figsize:
        figure(figsize=figsize)
        
    if (Y is None) or (task is None):
        scatter(X_pca[:,0], X_pca[:,1])
    else:
        if task=='classification':
            scatter(X_pca[:,0], X_pca[:,1], c=[COLORS[y] for y in Y], s=point_size)
        elif task=='regression':
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
			
            if cmap==None:
                cmap = cm.ocean
            norm = Normalize(vmin=min(Y), vmax=max(Y)) 		
            scatter(X_pca[:,0], X_pca[:,1], c=[cmap(norm(y)) for y in Y], s=point_size)
        else:
            raise Exception('task should be either "regression" or "classification"!')
			
    xlabel('principal component 1')
    ylabel('principal component 2')
    title('First 2 components explain %.3f variance'%cum_explained_variance[1])
        
        

def plot_corr(df,size=10,show_colorbar=True,show_grid=True):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
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
    blue_red_cmap = LinearSegmentedColormap('hot', cdict)


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



        
def show_feature_importances(features, importances, max_features=20, min_importance=0, figsize=None):
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