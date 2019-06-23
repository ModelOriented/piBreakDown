import pandas as pd
import numpy as np
import math
from piBreakDown.piBreakDownResults import piBreakDownResults
from piBreakDown.ExplainHelpers import *

class Interactions:
    def __init__(self, model, data, target_label, interaction_preference = 1):
        """
        Parameters
        ----------
        model: scikit-learn model
            a model to be explained, with `fit` and `predict` functions
        data: pandas.DataFrame
            data that was used to train model
        target_label: str
            label of target variable
        """
        self._model = model
        self._data = data
        self._target_label = target_label
        self._interaction_preference = interaction_preference
        
    def local_interactions(self, new_observation, keep_distributions = False, classes_names = None, order = None): 
        
        target_yhat = self._model.predict_proba(new_observation.loc[self._data.columns != self._target_label].values.reshape(1,-1))[0]
        
        if classes_names is None:
            if isinstance(target_yhat, float):
                classes_names = [1]
            else:
                classes_names = list(range(0,len(target_yhat)))

        baseline_yhat = self._model.predict_proba(self._data.loc[:,self._data.columns != self._target_label]).mean(axis = 0)
        average_yhats = calculate_1d_changes(self._model, self._data.loc[:, self._data.columns != self._target_label], 
                                                    new_observation.loc[self._data.columns != self._target_label], classes_names)
        
        diffs_1d = average_yhats - baseline_yhat
        
        changes = calculate_2d_changes(self._model, self._data.loc[:, self._data.columns != self._target_label], 
                                                    new_observation.loc[self._data.columns != self._target_label], classes_names,
                                                    diffs_1d)
        
        changes['average_yhats'] = changes['average_yhats'] - baseline_yhat
        changes['average_yhats_norm'] = changes['average_yhats_norm'] - baseline_yhat

        feature_path = create_ordered_path_2d(diffs_1d, changes, order, self._interaction_preference)
        return self._calculate_contributions_along_path(self._data.loc[:,self._data.columns != self._target_label],
                                                      new_observation, feature_path, keep_distributions, self._target_label,
                                                      baseline_yhat, target_yhat, classes_names)
        
    def _calculate_contributions_along_path(self, data, new_observation, feature_path, keep_distributions, 
                                           label, baseline_yhat, target_yhat, classes_names):
        
        open_variables = data.columns
        current_data = data.copy()
        yhats = None
        yhats_mean = pd.DataFrame(columns=classes_names, index=feature_path.index)
        selected_rows = []
        yhats = {}
        
        for index, row in feature_path.iterrows():
            candidates = [row['var1']]
            if row['var2'] is not None:
                candidates.append(row['var2'])
                
            if all([x in open_variables for x in candidates]):
                current_data.loc[:,candidates] = new_observation[candidates].tolist()
                yhats_pred = self._model.predict_proba(current_data)
                
                if keep_distributions:
                    return
                
                yhats_mean.loc[index,:] = yhats_pred.mean(axis = 0)
                selected_rows.append(index)
                open_variables = set(open_variables) - set(candidates)
                
        selected = feature_path.loc[selected_rows,:]

        selected_values = []
        var_names = []
        for index, row in selected.iterrows():
            selected_values.append(nice_pair(new_observation, row['var1'], row['var2']))
            var_names.append(str(row['var1']) + (',' + str(row['var2']) if row['var2'] is not None else ''))
            
        variable_name = ['intercept'] + var_names + [""]
        variable_value = ['1'] + selected_values + ['']
        variable = ['intercept'] + [x + ' = ' + y for x,y in zip(var_names,selected_values)] + ['prediction']

        cummulative = pd.DataFrame(columns=classes_names)
        cummulative.loc['baseline_yhat',:] = baseline_yhat
        cummulative = cummulative.append(yhats_mean.loc[selected.index.values,:])
        cummulative.loc['target_yhat',:] = target_yhat
        
        contribution = cummulative.diff(axis = 0)
        contribution.loc['baseline_yhat',:] = cummulative.loc['baseline_yhat',:]
        contribution.loc['target_yhat',:] = cummulative.loc['target_yhat',:]
        
        return piBreakDownResults(variable_name, variable_value, variable, cummulative, contribution, yhats)