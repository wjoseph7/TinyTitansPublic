import matplotlib.pyplot as plt
import pandas as pd

def plot_growth(plot_label: str,
                df: pd.DataFrame,
                savedir: str='results/',
                initial_capital: float=1.0) -> None:
    """
    Summary:
        This creates and saves an appreciation plot for our strategy and 
        benchmark funds
    Args:
        plot_label (str): the name to save our png to
        df (pd.DataFrame): df with our monthly appreciation data
        save_dir (str): directory we're saving to
        initial_capital (float): initial capital for plot
    Returns:
        None
    """
    x = list(range(0, len(df.index)+1))

    y_dict = {}

    for col in df.columns:

        if 'monthly_roi' in col:
            y = [initial_capital]
            for n in df.index:
                monthly_roi = df.loc[n, col]
                float_roi = 1 + monthly_roi / 100
                appreciation = y[-1] * float_roi
                y.append(appreciation)

            y_dict[col] = y

    for label in y_dict.keys():
        y = y_dict[label]
        plt.semilogy(x, y, label=label)

    plt.xlabel('Months of Investment')
    plt.ylabel(f'Growth of ${initial_capital}')
    plt.legend()
    plt.savefig(savedir + plot_label)
    
