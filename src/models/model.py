# -*- coding: utf-8 -*-
from src.features.creation import FeatureCreator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from eli5 import transform_feature_names


def make_model():
    feature_creator = FeatureCreator()
    
    feature_transformer = ColumnTransformer([
        ('StandardScaler', StandardScaler(), make_column_selector('.*'))
    ])
    
    featurizer = Pipeline([
        ('creator', feature_creator), 
        ('transformer', feature_transformer)
    ])
    
    model = Pipeline([
        ('featurizer', featurizer), 
        ('predictor', LogisticRegression())
    ])
    
    return model


# allow ColumnTransformer to show feature names with eli5
# from https://stackoverflow.com/questions/60949339/how-to-get-feature-names-from-eli5-when-transformer-includes-an-embedded-pipelin
@transform_feature_names.register(ColumnTransformer)
def _col_tfm_names(transformer, in_names=None):
    if in_names is None:
        from eli5.sklearn.utils import get_feature_names
        # generate default feature names
        in_names = get_feature_names(transformer, num_features=transformer._n_features)
    # return a list of strings derived from in_names
    feature_names = []
    for name, trans, column, _ in transformer._iter(fitted=True):
        if hasattr(transformer, '_df_columns'):
            if ((not isinstance(column, slice))
                    and all(isinstance(col, str) for col in column)):
                names = column
            else:
                names = transformer._df_columns[column]
        else:
            indices = np.arange(transformer._n_features)
            names = ['x%d' % i for i in indices[column]]
        # erm, want to be able to override with in_names maybe???

        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            continue
        if trans == 'passthrough':
            feature_names.extend(names)
            continue
        feature_names.extend([name + "__" + f for f in
                              transform_feature_names(trans, in_names=names)])
    return feature_names

@transform_feature_names.register(OneHotEncoder)
def _ohe_names(est, in_names=None):
    return est.get_feature_names(input_features=in_names)