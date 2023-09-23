from typing import Tuple
from TinyTitans.backtesting.backtest import BackTest
from datetime import datetime
import pickle
from TinyTitans.backtesting.polygon_api.ticker_api_calls import get_adjusted_close


class NanCorrector:

    def correct_nans(self,
                     date_list: List,
                     df: pd.DataFrame,
                     offset: int,
                     column_name: str,
                     num_stocks: int,
                     missing_approximation: float=None,
                     debug: bool=False) -> Tuple[pd.DataFrame, int]:
        """
        Summary:
            This function corrects a df for nans by using the nearest adjusted
            close or a missing approximation penalty.

            Called by BackTest's appreciation_operation method.
        Args:
            date_list (List): list of monthly dates from BackTest in temporal 
                order
            df (pd.DataFrame): df containing TT appreciations for particular 
                time period
            offset (int): number a months we're looking ahead from a particular
                date
            column_name (str): appreciation column we're modifying
            num_stocks (int): number of stocks in TT backtest
            missing_approximation (float): what to approximate appreciation 
                with if nan is encountered
            debug (bool): prints out useful info for debugging if true
        Returns:
            Tuple[pd.DataFrame, int]:
                - pd.DataFrame: df with new nan adjusted column
                - int: number of nans we encountered
        """
        nans_encountered = 0

        df[column_name + '_adjusted'] = df[column_name]

        df_length = len(df.index)

        print(f"only {df_length} / {num_stocks} stocks found")
        
        for n in range(min(num_stocks, df_length)):

            appreciation = df.loc[n, column_name]
            ticker = df.loc[n, 'ticker']
            adjusted_close = df.loc[n, 'adjusted_close']
            date = df.loc[n, 'date']
            date_index = date_list.index(date)
            offset_date = date_list[date_index+offset]

            if np.isnan(appreciation):
                nans_encountered += 1
                
                if missing_approximation is not None:
                    appreciation = missing_approximation
                else:
                    nearest_adj_close = self.get_nearest_adjusted_close(
                        ticker,
                        date,
                        offset_date
                    )
                    appreciation = BackTest.get_offset_appreciation(
                        offset,
                        adjusted_close,
                        nearest_adj_close
                    )

                    if debug:
                        print(ticker)
                        print(appreciation)
                    
                df.loc[n, column_name + '_adjusted'] = appreciation

        return df, nans_encountered

    def get_nearest_adjusted_close(self,
                                   ticker: str,
                                   begin_date: str,
                                   missing_date: str,
                                   cache=True) -> float:
        """
        Summary:
            This corrects a nan by finding the adjusted close nearest to the 
            missing date which is not a nan. We use a cache so we don't need to
            recompute the data each time.
        Args:
            ticker (str): ticker we're correcting nan for
            begin_date (str): begging date of the appreciation computation
            missing_date (str): missing date with nan to replace
            cache (bool): whether to save data to cache
        Returns:
            float: adjusted close to replace nan with
        """

        assert missing_date > begin_date

        str_begin_date = begin_date
        cache_fp = f'nan_cache/{begin_date}_{ticker}.pkl'

        # load nearest closes from cache if they exist
        if os.path.exists(cache_fp):
            nan_dict = pickle.load(open(cache_fp, 'rb'))
            last_adjusted_close = nan_dict['last_adjusted_close']
            return last_adjusted_close

        date = missing_date
        date = datetime.strptime(date,  '%Y-%m-%d') # convert to datetime

        #convert to datetime
        begin_date = datetime.strptime(begin_date,  '%Y-%m-%d') 

        while date >= begin_date:

            date -= timedelta(days=1)

            if is_trading_day(date):
                str_date = date.strftime('%Y-%m-%d')
                adjusted_close = get_adjusted_close(ticker, str_date)
                if not np.isnan(adjusted_close):

                    if cache:
                        nan_dict = {'ticker' : ticker,
                                    'purchase_date' : str_begin_date,
                                    'missing_date' : missing_date,
                                    'last_close_date' : str_date,
                                    'last_adjusted_close' : adjusted_close}

                        with open(cache_fp, 'wb') as f:
                            pickle.dump(nan_dict, f)
                            f.close()
                    
                    return adjusted_close

        raise ValueError
