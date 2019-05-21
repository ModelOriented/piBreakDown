import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class PlotUtils:
    """
    Class containing static methods for plotting piBreakDown results
    """
    
    @staticmethod
    def plot_contribution(results):
        """
        Method for plotting contribution plot.
        
        Parameters
        ----------
        results: dict
            Dictionary returned from piBreakDown methods, must containg 'contribution' key
        """
        
        if 'contribution' not in results:
            print('\'contribution\' not missing in results, make sure that you are using result form piBreakDown methods')
            return
        
        fig, ax = plt.subplots()

        results_data = results['cummulative'].copy()
        y_vals = results_data.loc[:,1].values
        x_vals = results_data.index
        white_bar_vals = [0] * len(y_vals)
        white_bar_vals[0] = y_vals[0]
        base_line = mlines.Line2D([y_vals[0],y_vals[0]], [-1,len(x_vals) + 1],
                          linestyle = '--', linewidth = 1, color = 'black', alpha = 0.5)
        ax.add_line(mlines.Line2D([y_vals[0],y_vals[0]], [-0.5, 1.5],
                                          linewidth = 1, color = 'black'))
        red_bar_x = []
        red_bar_y = []

        for i in range(1, len(x_vals) - 1):
            if y_vals[i] >= y_vals[i - 1]:
                white_bar_vals[i] = y_vals[i - 1]
                ax.add_line(mlines.Line2D([y_vals[i],y_vals[i]], [i-0.5,i + 1.5],
                                          linewidth = 1, color = 'black'))
            else:
                white_bar_vals[i] = y_vals[i]
                ax.add_line(mlines.Line2D([y_vals[i],y_vals[i]], [i-0.5,i + 1.5],
                                          linewidth = 1, color = 'black'))
                y_vals[i] = y_vals[i - 1]
                red_bar_x.append(x_vals[i])
                red_bar_y.append(y_vals[i])

        white_bar_vals[len(y_vals) - 1] = white_bar_vals[0]

        val_bars = ax.barh(x_vals, y_vals, align='center', color = 'green')
        labels = [''] * len(x_vals)
        for i in range(len(x_vals)):
            labels[i] = str(round(results['contribution'].loc[results_data.index[i],:].values[0],3))

        for rect, label in zip(val_bars,labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() + 0.025, rect.get_y() + height/2, label,
                    ha='left', va='center')

        red_bars = ax.barh(red_bar_x, red_bar_y, align='center', color = 'red')    
        white_bars = ax.barh(x_vals, white_bar_vals, align='center', color = 'white')
        plt.gca().invert_yaxis()
        ax.add_line(base_line)
        xlim = ax.set_xlim(0,1)