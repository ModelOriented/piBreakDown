import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class PlotUtils:
    """
    Class containing static methods for plotting piBreakDown results
    """
    
    @staticmethod
    def plot_contribution(results, plot_class = 1):
        """
        Method for plotting contribution plot.
        
        Parameters
        ----------
        results: piBreakDownResults
            piBreakDownResults returned from piBreakDown methods
        plot_class: str or numeric
            Class for which the plot will be displayed
        """
        
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        
        results_data = results.cummulative.copy()
        
        results_y = results_data.loc[:,plot_class].values
        x_vals = results.variable
        y_vals = [0] * len(x_vals)
        white_bar_vals = [0] * len(x_vals)
        white_bar_vals[0] = results_y[0]
        base_line = mlines.Line2D([results_y[0],results_y[0]], [-1,len(x_vals) + 1],
                          linestyle = '--', linewidth = 1, color = 'black', alpha = 0.5)
        ax.add_line(mlines.Line2D([results_y[0],results_y[0]], [-0.5, 1.5],
                                          linewidth = 1, color = 'black'))
        red_bar_x = []
        red_bar_y = []
        
        y_vals[0] = results_y[0]
        y_vals[len(results_y)-1] = results_y[len(results_y)-1]
        for i in range(1, len(x_vals) - 1):
            if results_y[i] >= results_y[i - 1]:
                white_bar_vals[i] = results_y[i - 1]
                y_vals[i] = results_y[i]
                ax.add_line(mlines.Line2D([results_y[i],results_y[i]], [i-0.5,i + 1.5],
                                          linewidth = 1, color = 'black'))
            else:
                white_bar_vals[i] = results_y[i]
                ax.add_line(mlines.Line2D([results_y[i],results_y[i]], [i-0.5,i + 1.5],
                                          linewidth = 1, color = 'black'))
                y_vals[i] = results_y[i - 1]
                red_bar_x.append(x_vals[i])
                red_bar_y.append(y_vals[i])

        white_bar_vals[len(y_vals) - 1] = white_bar_vals[0]

        val_bars = ax.barh(x_vals, y_vals, align='center', color = 'green')
        labels = [''] * len(x_vals)
        
        labels[0] = str(round(results.contribution.loc[results_data.index[0],plot_class],3))
        for i in range(1,len(x_vals)-1):
            val = round(results.contribution.loc[results_data.index[i],plot_class],3)
            labels[i] = ('-' if val < 0 else '+') + str(val)
        labels[len(x_vals)-1] = str(round(results.contribution.loc[results_data.index[len(x_vals)-1],plot_class],3))
            
        for rect, label in zip(val_bars,labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() + 0.015, rect.get_y() + height/2, label,
                    ha='left', va='center')

        red_bars = ax.barh(red_bar_x, red_bar_y, align='center', color = 'red')    
        white_bars = ax.barh(x_vals, white_bar_vals, align='center', color = 'white')
        plt.gca().invert_yaxis()
        plt.grid(alpha = 0.2)
        ax.add_line(base_line)
        
        max_val = max(results.cummulative[plot_class].values)
        min_val = min(results.cummulative[plot_class].values)
        xlim = ax.set_xlim(min_val * 0.8, max_val * 1.2)