def calculate_1d_changes(model, data, new_observation, classes_names):
    average_predictions_df = pd.DataFrame(columns=classes_names, index = data.columns)
    for col in average_predictions_df.index:
        data_tmp = data.copy()
        data_tmp.loc[:,col] = new_observation.loc[col]
        average_predictions_df.loc[col,:] =  model.predict_proba(data_tmp).mean(axis = 0)

    return average_predictions_df

def create_ordered_path(diffs_1d, order):
    feature_path = pd.DataFrame({'diffs': diffs_1d})
    if order is None:
        feature_path = feature_path.sort_values(by = 'diffs', ascending = False)
    else:
        feature_path = feature_path.loc[order]

    return feature_path

def create_ordered_path_2d(diffs_1d, changes, order, interaction_preference):
    feature_path = pd.DataFrame({'val': changes['average_yhats_norm'].abs().mean(axis = 1)*interaction_preference})
    feature_path['var1'] = feature_path.index.get_level_values(0)
    feature_path['var2'] = feature_path.index.get_level_values(1)
    
    diffs = pd.DataFrame({'val': diffs_1d.abs().mean(axis = 1)})
    diffs['var1'] = diffs.index.values
    diffs['var2'] = None
    
    feature_path = pd.concat([feature_path, diffs], ignore_index = True)
    
    if order is None:
        return feature_path.sort_values(by = 'val', ascending = False)
    else:
        return feature_path.loc[order]

def calculate_2d_changes(model, data, new_observation, classes_names, diffs_1d):
    index = pd.MultiIndex.from_tuples((), names=['var1', 'var2'])
    average_predictions_df = pd.DataFrame(columns=classes_names, index = index)
    average_predictions_norm_df = pd.DataFrame(columns=classes_names, index = index)
    
    for i in range(data.shape[1]):
        for j in range((i+1),data.shape[1]):
            data_tmp = data.copy()
            data_tmp.loc[:,data_tmp.columns[i]] = new_observation[data_tmp.columns[i]]
            data_tmp.loc[:,data_tmp.columns[j]] = new_observation[data_tmp.columns[j]]
            yhats_mean = model.predict_proba(data_tmp).mean(axis = 0)
            
            average_predictions_df.loc[(data_tmp.columns[i],data_tmp.columns[j]),:] = yhats_mean
            average_predictions_norm_df.loc[(data_tmp.columns[i],data_tmp.columns[j]),:] = (yhats_mean - 
                                    diffs_1d.loc[data_tmp.columns[i],:].values - diffs_1d.loc[data_tmp.columns[j],:].values)
    
    return {'average_yhats': average_predictions_df,
            'average_yhats_norm': average_predictions_norm_df}

def nice_pair(x, ind1, ind2):
    if(ind2 is None):
        return nice_format(x[ind1])
    return nice_format(x[ind1]) + ':' + nice_format(x[ind2])

def nice_format(x):
    if type(x) in [int, float]:
        return str(round(x,2))
    return str(x)