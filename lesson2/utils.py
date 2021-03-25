import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_contour(X_train, y_train, pipe, granularity=0.1, grid_side=0.5):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    cm = plt.cm.PiYG# let's change the colormap.

    # first, we get the min-max range over which we want to plot
    # this is the area for which we want to know the behavior of the model
    # we add some extra space with grid_side to the feature space.
    x0_min, x0_max = X_train[:,0].min() -grid_side, X_train[:,0].max() +grid_side
    x1_min, x1_max = X_train[:,1].min() -grid_side, X_train[:,1].max() +grid_side
    # we make a grid of coordinates
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, granularity),
                         np.arange(x1_min, x1_max, granularity))
    # and combine the grid into a new dataset.
    # this new dataset covers (with some granularity) every point of the original dataset
    # this newx is equal to the featurespace we want to examine.
    newx = np.c_[xx.ravel(), yy.ravel()]

    # we make a prediction with the new dataset. This will show us predictions over the complete featurespace.
    yhat = pipe.predict(newx)

    # and reshape the prediction, so that it will match our gridsize
    z = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=cm, alpha = 0.5)
    data = pd.DataFrame({'x1':X_train[:,0], 'x2':X_train[:,1]})
    sns.scatterplot(data=data,x= 'x1', y='x2', c=y_train, cmap=cm)
    
def gridsearch_heatmap(gridresults, param_grid, vmin = None, vmax = None, figsize=(10,10)):
    idx, col = ['param_' + [*param_grid.keys()][i] for i in range(2)]
    pivoted = pd.pivot_table(pd.DataFrame(gridresults.cv_results_),
                            values = 'mean_test_score',
                            index = idx,
                            columns = col)
    pivoted.index = ["{:.4f}".format(x) for x in pivoted.index]
    pivoted.columns = ["{:.4f}".format(x) for x in pivoted.columns]
    plt.figure(figsize=figsize)
    sns.heatmap(pivoted, vmin = vmin, vmax = vmax, annot = True)
    
    
def compare_results(results, ylim=None):
    data = pd.DataFrame(results, index = ['train', 'test']).reset_index()
    data = data.melt(id_vars='index')
    sns.barplot(x='index', y = 'value', hue='variable', data=data)
    plt.ylim(ylim, 1)