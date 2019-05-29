import pandas as pd
import numpy as np
from piBreakDown.piBreakDownResults import piBreakDownResults

class Attributions:
    """
    Python version of iBreakDown package in R (https://github.com/ModelOriented/iBreakDown)
    """
    def __init__(self, model, data, target_label):
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
        
    def local_attributions(self, new_observation, keep_distributions = False, classes_names = None, order = None): 
        """
        Parameters
        ----------
        new_observation: pandas.Series
            a new observation with columns that correspond to variables used in the model
        keep_distributions: boolean
            if `True`, then distribution of partial predictions is stored
        classes_names: list
            names of the classes to be predicted, if `None` then it will be number from 0 to len(predicted values)
        order: list
            if not `None`, then it will be a fixed order of variables. It can be a numeric vector or vector 
            with names of variables
        """
        
        #not used due to "feature" cols_to_use = set(self._data.columns[self._data.columns != self._target_label]).intersection(set(new_observation.index))
        target_yhat = self._model.predict_proba(new_observation.loc[self._data.columns != self._target_label].values.reshape(1,-1))[0]
        if classes_names is None:
            classes_names = list(range(0,len(target_yhat)))
           
        yhatpred = self._model.predict_proba(self._data.loc[:,self._data.columns != self._target_label])
        baseline_yhat = yhatpred.mean(axis = 0)
        average_yhats = self._calculated_1d_changes(self._data.loc[:, self._data.columns != self._target_label], 
                                                    new_observation.loc[self._data.columns != self._target_label], classes_names)
        diffs_1d = (average_yhats.subtract(baseline_yhat)**2).mean(axis = 1)
        feature_path = self._create_ordered_path(diffs_1d, order)
        tmp = self._calculate_contributions_along_path(self._data.loc[:,self._data.columns != self._target_label],
                                                      new_observation, feature_path, keep_distributions, self._target_label,
                                                      baseline_yhat, target_yhat, classes_names)
        return tmp
    
    def _calculated_1d_changes(self, data, new_observation, classes_names):
        average_predictions_df = pd.DataFrame(columns=classes_names, index = data.columns)
        for col in average_predictions_df.index:
            data_tmp = data.copy()
            data_tmp.loc[:,col] = new_observation.loc[col]
            average_predictions_df.loc[col,:] =  self._model.predict_proba(data_tmp).mean(axis = 0)
            
        return average_predictions_df
    
    def _create_ordered_path(self, diffs_1d, order):
        feature_path = pd.DataFrame({'diffs': diffs_1d})
        if order is None:
            feature_path = feature_path.sort_values(by = 'diffs', ascending = False)
        else:
            feature_path = feature_path.loc[order]
            
        return feature_path
    
    def _calculate_contributions_along_path(self, data, new_observation, feature_path, keep_distributions, label, baseline_yhat, target_yhat, classes_names):
        open_variables = data.columns
        current_data = data.copy()
        step = 0
        yhats = None
        yhats_mean = pd.DataFrame(columns=classes_names, index=feature_path.index)
        selected_rows = []
        yhats = {}
        
        for i in feature_path.index:
            candidates = [i]
            if all([x in open_variables for x in candidates]):
                current_data.loc[:,candidates] = new_observation[candidates].tolist()
                step += 1
                yhats_pred = self._model.predict_proba(current_data)
                
                if(keep_distributions):
                    distribution_for_batch = pd.DataFrame()
                    
                    for j in range(yhats_pred.shape[1]):
                        lab = (label + '.' + classes_names[j]) if yhats_pred.shape[1] > 1 else label
                        distribution_for_batch['lab'] = yhats_pred[:,j]
                        
                    distribution_for_batch['variable_name'] = i
                    distribution_for_batch['variable'] = str(i) + ' = ' + self._nice_pair(new_observation, candidates[0], None)
                    yhats[i] = distribution_for_batch
                    
                yhats_mean.loc[i,:] = yhats_pred.mean(axis = 0)
                selected_rows.append(i)
                open_variables = set(open_variables) - set(candidates)
        selected = feature_path.loc[selected_rows,:]
        selected_values = []
        for i in selected.index:
            selected_values.append(self._nice_pair(new_observation, i, None))
            
        variable_name = ['intercept'] + feature_path.index
        variable_value = ['1'] + selected_values
        variable = ['intercept'] + [x + ' = ' + y for x,y in zip(variable_name,selected_values)] + ['prediction']
        cummulative = pd.DataFrame(columns=classes_names)
        cummulative.loc['baseline_yhat',:] = baseline_yhat

        cummulative = cummulative.append(yhats_mean)
        cummulative.loc['target_yhat',:] = target_yhat
        
        contribution = cummulative.diff(axis = 0)
        contribution.loc['baseline_yhat',:] = cummulative.loc['baseline_yhat',:]
        contribution.loc['target_yhat',:] = cummulative.loc['target_yhat',:]
        
        return piBreakDownResults(variable_name, variable_value, variable, cummulative, contribution, yhats)
    
    def _nice_pair(self, x, ind1, ind2):
        if(ind2 is None):
            return self._nice_format(x[ind1])
        return self._nice_format(x[ind1]) + ':' + self._nice_format(x[ind2])
        
    def _nice_format(self, x):
        if type(x) in [int, float]:
            return str(round(x,2))
        return str(x)