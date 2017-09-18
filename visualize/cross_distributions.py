from pylab import *
from common.visualize.colors import COLORS


def cross_distributions(X, feature_names=None, bins=30, point_size=5, figsize=None):
    """Generate scatter plots of all pairs of variables. Variables are columns of matrix X.
    
    Input:
        X: matrix n_objects x n_features
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot
        figsize: tuple (X_size, Y_size)
    """

    nVariables = X.shape[1]
    assert nVariables<50, 'nVariables should be less than 50.'
    
    if feature_names is None:
        feature_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(feature_names)==nVariables,'len(feature_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(feature_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # vertical variable names
                ax.set_ylabel(feature_names[i])            
            
            if i == j:
                ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c='b', lw=0, s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])
			
			
			

def cross_distributions_classification(X, Y, var_names=None, bins=30, point_size=5, figsize=None):
    """Generate scatter plots of all pairs of features, coloring objects with their class. 
    Features are columns of matrix X. Objects are rows. Y stores classes of objects.
    
    Input:
        X: matrix n_objects x n_features
        Y: stores classes of objects.
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot        
        figsize: tuple (X_size, Y_size)
    """
    
    classes = unique(Y)
    assert len(classes)<=len(COLORS),'Classes count should be <=%s'%len(COLORS)
    
    nVariables = X.shape[1]
    assert nVariables<40, 'nVariables should be less than 40.'
    
    if var_names is None:
        var_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(var_names)==nVariables,'len(var_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables+1, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(var_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # horizontal variable names
                ax.set_ylabel(var_names[i])            
            
            if i == j:
                ax.hist([X[Y==y,i] for y in classes], bins=bins, stacked=True, color=[COLORS[i] for i in range(len(classes))])
                #ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c=[COLORS[find(classes==y)[0]] for y in Y], lw=0, s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])
    
    ax = fig.add_subplot(nVariables+1, nVariables, nSub+1)
    import matplotlib.patches as mpatches
    recs = []
    for y in classes:
        recs.append(mpatches.Rectangle((0,0),1,1,fc=COLORS[find(classes==y)[0]]))
    plt.legend(recs,classes,loc='center')
    axis('off');

			
			
def cross_distributions_regression(X, Y, var_names=None, bins=30, point_size=5, cmap=None, figsize=None):
    """Generate scatter plots of all pairs of features, coloring objects with their output regression value. 
    Features are columns of matrix X. Objects are rows of X. Y stores output values for each object.
    
    Input:
        X: matrix n_objects x n_features
        Y: stores classes of objects.
        feature_names: list of feature names
        bins: number of bins for each histogram on the diagonal
        point_size: size of each point on the scatterplot        
        figsize: tuple (X_size, Y_size)
    """

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    if cmap==None:
        cmap = cm.hot #cm.ocean
    norm = Normalize(vmin=min(Y), vmax=max(Y)) 
    
    nVariables = X.shape[1]
    assert nVariables<50, 'nVariables should be less than 50.'
    
    if var_names is None:
        var_names = ['x%d'%i for i in range(nVariables)]
    else:
        assert len(var_names)==nVariables,'len(var_names) should be equal to variables count.'

    if figsize==None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    
    for i in range(nVariables):
        for j in range(nVariables):
            nSub = i * nVariables + j + 1
            ax = fig.add_subplot(nVariables, nVariables, nSub)
            
            if i == 0:  # horizontal variable names
                ax.set_xlabel(var_names[j]) 
                ax.xaxis.set_label_position('top')

            if j == 0:  # horizontal variable names
                ax.set_ylabel(var_names[i])            
            
            if i == j:
                ax.hist(X[:,i],bins=bins)
            else:
                ax.scatter(X[:,j], X[:,i], c=[cmap(norm(y)) for y in Y],lw=0,s=point_size)
            ax.set_xticks([])
            ax.set_yticks([])