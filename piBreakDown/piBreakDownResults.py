class piBreakDownResults:
    """
    Class that represents object returned from piBreakDown
    
    Available fields
    ----------------
    variable_name: list
        List containing names of the variables used in model
    variable_value: list
        List containing values of the variable that used in explanation
    variable: list
        List containing contatenated values of variable_name and variable_values
    cummulative: pandas.DataFrame
        DataFrame containing cummulative values of variables for observation used in explaining
    contribution: pandas.DataFrame
        DataFrame containing values of contribution of each variable for observation used in explaining
    yhats: pandas.DataFrame
        DataFrame with distributions of variables
    """
    
    def __init__(self, variable_name, variable_value, variable, cummulative, contribution, yhats):
        self.variable_name = variable_name
        self.variable_value = variable_value
        self.variable = variable
        self.cummulative = cummulative
        self.contribution = contribution
        self.yhats = yhats