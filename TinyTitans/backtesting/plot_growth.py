import matplotlib.pyplot as plt
import pandas as pd

def plot_growth(df: pd.DataFrame, savedir: str = 'results/', initial_capital: float = 1.0) -> None:

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
    plt.savefig(savedir + '10_stocks_non_conservative_missing_2022-12-08.png')#'10_stocks_inv_TT_ps_ratio_0_missing_approx_historical_returns_over_time_2022-12-03.png')
    
