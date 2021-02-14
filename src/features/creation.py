# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FeatureCreator(BaseEstimator, TransformerMixin):
    """ A transformer that constructs and appends new features to the dataframe.
    Parameters
    ----------
    None
    
    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    
    data_feature_names_ : list of strings
        Names of features from the fitted data.
    
    new_feature_names_ : list of strings
        Names of the constructed features.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        _ = check_array(X, dtype=None, accept_sparse=True)

        self.n_features_ = X.shape[1]
        self.data_feature_names_ = list(X.columns)
        self.new_feature_names_ = ['sepal area', 
                                   'petal area', 
                                   'sepal/petal area ratio']
        
        # Return the transformer
        return self
    
    def transform(self, X):
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation and copying
        # check_array changes input type, so we don't use its output
        _ = check_array(X, dtype=None, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        X_new = X.copy()
        
        # Add new features
        X_new['sepal area'] = X_new['sepal length (cm)'] * X_new['sepal width (cm)']
        X_new['petal area'] = X_new['petal length (cm)'] * X_new['petal width (cm)']
        X_new['sepal/petal area ratio'] = X_new['sepal area'] / X_new['petal area']
        
        return X_new
    
    def get_feature_names(self):
        return self.data_feature_names_ + self.new_feature_names_