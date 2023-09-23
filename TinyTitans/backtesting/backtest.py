import os
import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Union, Dict, Tuple
from TinyTitans.backtesting.polygon_api.utils import *
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
from TinyTitans.backtesting.missing_data_corrections.process_corporate_actions import CA_Parser
from TinyTitans.backtesting.missing_data_corrections.correct_nans import NanCorrector
from TinyTitans.backtesting.missing_data_corrections.adjust_market_cap import MarketCapAdjuster
from TinyTitans.backtesting.analyze_backtest import BackTestAnalyzer

num_stocks_default = 10
market_min_default = int(25*10**6)
market_max_default = int(250*10**6)
ps_ratio_default = 1.0

class BackTest:

    def __init__(self, polygon_data_fp: str, save_adjusted: bool=True) -> None:
        """
        Summary:
            This is the constructor for the BackTest class. We do the 
            following:

                - Instantiate the corporate action parser for later use in 
                    handle_symbol_change_and_bankruptcy
                - Load the polygon data into a df
                - Add date and ticker columns to the df
                - Compute and store a list of ordered sequential dates
                - Validate the list is indeed sequential
                - Adjust market cap for companies with multiple tickers
                - Add a column to estimate the price to sales ratio

        Args:
            polygon_date_fp (str): filepath to the polygon data
            save_adjusted (bool): whether we want to save the polygon data 
                after market cap adjustment for quicker backtesting later.
        """
        
        self.ca_parser = CA_Parser()
        self.nan_corrector = NanCorrector()
        self.backtest_analyzer = BackTestAnalyzer(self)
        self.polygon_data = self.read_polygon_data(polygon_data)
        self.polygon_data = self.compute_date_and_ticker_columns(
            self.polygon_data)

        self.date_list = self.compute_date_list(self.polygon_data)
        self.validate_date_list(self.date_list)

        self.polygon_data = \
            MarketCapAdjuster.adjust_market_cap_for_multiple_tickers_static(
                self.polygon_data,
                self.date_list,
                polygon_data_fp,
                save_adjusted)

        self.polygon_data = self.compute_PS_ratio_column(self.polygon_data)

    def read_polygon_data(self, fp: str) -> pd.DataFrame:
        """
        Summary:
            Loads polygon data which contains the following columns:

                'cik',
                'market_cap',
                'momentum',
                'shares_outstanding',
                'adjusted_close',
                'unadjusted_close',
                'trailing_12_month_revenue'

            Then sets the index to the tuple key column, which is just the
            ticker symbol and date in str format: "('ARNA', '2013-06-28')"
        Args:
            fp (str): Filepath to the csv containing the polygon data
        Returns:
            pd.DataFrame: dataframe ready for processing
        """

        df = pd.read_csv(fp)
        df = df.set_index('tuple_key')

        return df

    def compute_date_and_ticker_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary:
            Adds date and ticker columns to the df by extracting the info from
            the tuple key index
        Args:
            df (pd.DataFrame): The df we want to add the date and ticker 
            columns to
        Returns:
            pd.DataFrame: Same df but with date and ticker columns
        """

        ticker_list = []
        date_list = []
        
        for tuple_key in df.index:

            ticker = eval(tuple_key)[0]
            date = eval(tuple_key)[1]

            ticker_list.append(ticker)
            date_list.append(date)

        df['ticker'] = ticker_list
        df['date'] = date_list

        return df

    def compute_PS_ratio_column(self, 
                                df: pd.DataFrame,
                                market_cap_col: str='adjusted_market_cap') \
                                    -> pd.DataFrame:
        """
        Summary:
            This creates a new column of the dataframe intended to act as an
            estimate for the price to sales ratio. We just take the market cap
            and divide it by the trailing twelve month revenue.
        Args:
            df (pd.DataFrame):
            market_cap_col (str): The column name to use for market cap
        Returns:
            pd.DataFrame: with new PS_ratio column
        """
        df['PS_ratio'] = df[market_cap_col] / df['trailing_twelve_month_revenue']

        return df

    def compute_date_list(self, df: pd.DataFrame, verbose=False) -> List:
        """
        Summary:
            Creates an ordered list of unique dates extracted from the df
        Args:
            df (pd.DataFrame): df with date column whose dates we wish to 
                               extract
            verbose (bool): whether to print date range (useful for debugging)
        Returns:
            List: sorted list of unique dates in df
        """

        date_list = list(set(list(df['date'])))
        date_list = sorted(date_list)

        if verbose:
            print("Date Range:")
            pprint(date_list)
            print('\n\n')

        return date_list

    def validate_date_list(self, date_list: List) -> None:
        """
        Summary:
            Validates that year and month of dates and date list are in 
            sequential order. Day not used.

            Assertion thrown if this is not the case. Otherwise None is 
            returned.
        Args:
            date_list (List): The date list we wish to validate
        Returns:
            None
        """

        min_time = min(date_list)
        max_time = max(date_list)

        for n in range(len(date_list)-1):
            date1 = date_list[n]
            date2 = date_list[n+1]

            y1, m1, _ = date1.split('-')
            y2, m2, _ = date2.split('-')

            y1, m1 = int(y1), int(m1)
            y2, m2 = int(y2), int(m2)

            try:
                assert (m1 + 1) % 12 == m2 % 12

                if m1 == 12:
                    assert y1 == y2 - 1
                else:
                    assert y1 == y2

            except AssertionError as e:
                print(e)
                print(date1)
                print(date2)

        print(f"Date range validated. Data spans from {min_time} until " +\
            f"{max_time}.")

    def handle_appreciation_column_key_error(self,
                                             ticker: str,
                                             date: str,
                                             offset_date: str,
                                             label: str,
                                             data_df: pd.DataFrame) -> float:
        """
        Summary:
            This method handles the case where we get a KeyError in the 
            loop appreciation column method. Since there's a KeyError that 
            means the appreciation data point doesn't exist. We check for 
            bankruptcy or ticker symbol changes and adjust accordingly. 
            If no such corporate actions are found we return nan.
        Args:
            ticker (str): ticker we want aprpeciation of
            date (str): initial date we're querying
            offset_date (str): offset date
            label (str): label for new appreciation column
            data_df (pd.DataFrame): df we're pulling the data from
        Returns:
            float: the offset appreciation in the case of a KeyError
        """
        tuple_key, action = self.handle_symbol_and_bankruptcy(
            ticker,
            date,
            offset_date
        )
        if label != 'momentum':
            print("#####")
            print(tuple_key)
            print(action)

        if action is None:
            offset_appreciation = np.nan
        else:
            print(f"{ticker} {date} {offset_date} {action} {label}")

            if offset > 0 and action == 'bankruptcy':
                offset_appreciation = 0.0
            else:

                try:
                    offset_adj_close = data_df.loc[
                        tuple_key,
                        'adjusted_close'
                    ]
                    offset_appreciation = BackTest.get_offset_appreciation(
                        offset,
                        adjusted_close,
                        offset_adj_close
                    )
                except KeyError:
                    print("switched symbol missing")
                    offset_appreciation = np.nan
        
        return offset_appreciation

    def execute_compute_appreciation_column_row(self,
                                                i: int,
                                                offset: int,
                                                query_df: pd.DataFrame
                                                ) -> float:
        """
        Summary:
            Executes one loop for the compute_appreciation_column method.

            i corresponds to the row of the query_df we wish to use.
        Args:
            i (int): index of the query_df we wish to use
            offset (int): number of months between dates
            query_df (pd.DataFrame): df we're using to pull data from
        Returns:
            float: offset appreciation
        """
        adjusted_close = query_df.loc[i, 'adjusted_close']

        ticker = query_df.loc[i, 'ticker']
        date = query_df.loc[i, 'date']
        
        date_index = self.date_list.index(date)
        offset_date = self.date_list[date_index+offset]

        tuple_key = str((ticker, offset_date))

        try:
            offset_adj_close = data_df.loc[tuple_key, 'adjusted_close']
            offset_appreciation = BackTest.get_offset_appreciation(
                offset,
                adjusted_close,
                offset_adj_close
            )

        except KeyError:
            offset_appreciation = self.handle_appreciation_column_key_error(
                ticker,
                date,
                offset_date,
                label,
                data_df
            )
        
        return offset_appreciation


    def compute_appreciation_column(self,
                                    query_df: pd.DataFrame,
                                    offset: int,
                                    label: str,
                                    data_df: pd.DataFrame=None,
                                    debug=False) -> pd.DataFrame:
        """
        Summary:
            Computes appreciation of stocks in query_df using data in data_df.
        Args:
            query_df (pd.DataFrame): ususally a smaller snapshot of the data_df
                that we use for the query
            offset (int): the offset for appreciation we want to compute 
                (monthly 1, or annually for momentum over the past year -12)
            label (str): the label we want to save the appreciation under
            data_df (pd.DataFrame): the (usually) larger df we are pulling the 
                appreciation data from
            debug (bool): prints messages useful for debugging
        Returns:
            pd.DataFrame: query_df with the new appreciation column
        """

        if data_df is None:
            data_df = self.polygon_data
        
        if debug:
            print(query_df)

        appreciation_list = []

        # get rid of annoying pandas warning and make sure we're not modifying
        # polygon_data
        query_df = query_df.copy()
        
        for i in query_df.index:

            offset_appreciation = self.execute_compute_appreciation_column_row(
                i,
                offset,
                query_df
            )

            appreciation_list.append(offset_appreciation)

        query_df[label] = appreciation_list

        return query_df

    def handle_symbol_and_bankruptcy(self, 
                                     ticker: str,
                                     date: str,
                                     offset_date: str) -> Tuple[Tuple, str]:
        """
        Summary:
            Uses the CA_Parser to determine if there was a bankruptcy or symbol
            change between two given dates. 

            Returns the corporate action str and tuple key adjusted for 
            possible symbol changes.
        Args:
            ticker (str): the ticker we're interested in
            date (str): the initial date
            offset_date (str): the second date
        Returns:
            Tuple[Tuple, str]: 
                - Tuple: The tuple key with the new ticker if there was a 
                    symbol change
                - str: The corporate action str
        """
        backwards = False

        if date < offset_date:
            date_one = date
            date_two = offset_date
        else:
            date_one = offset_date
            date_two = date
            backwards = True
        
        action = self.ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)

        if action is None or 'symbol_change:' not in action:
            tuple_key = str((ticker, offset_date))

        else:
            tickers = action.split(':')[1]
            start_ticker, end_ticker = tickers.split('->')

            if not backwards:
                tuple_key = str((end_ticker, offset_date))
            else:
                tuple_key = str((start_ticker, offset_date))

        return tuple_key, action

    @staticmethod
    def get_offset_appreciation(self,
                                offset: int,
                                adjusted_close: float,
                                offset_adj_close: float,
                                eps: float=1e-3) -> float:
        """
        Summary:
            This function computes the appreciation between two values with
            a given monthly offset. Handles nan and dividing by near zero 
            quantities as well.

            Made a staticmethod since it's used by Nan_Corrector and I don't 
            want them both pointing to active instances of each other with 
            changing data.
        Args:
            offset (int): the number of months between adjusted_close and 
                offset_adj_close. Negative means we're offset_adj_close has an
                earlier date
            adjusted_close (float): adjusted close of current month
            offset_adj_close (float): adjusted close at offset
            eps (float): Used to ensure we don't divide by near zero values
        Returns:
            float: The appreciation
        """
        if offset < 0:
            if abs(offset_adj_close) > eps:
                offset_appreciation = adjusted_close / offset_adj_close
            else:
                offset_appreciation = np.nan
            
        elif offset > 0:

            if abs(adjusted_close) > eps:
                offset_appreciation = offset_adj_close / adjusted_close
            else:
                offset_appreciation = np.nan
                
        else:
            raise ValueError("offset must be nonzero")

        return offset_appreciation

    def all_but_momentum(self,
                         df: pd.DataFrame,
                         market_cap_min: int=market_min_default,
                         market_cap_max: int=market_max_default,
                         ps_ratio: float=ps_ratio_default,
                         market_cap_type: str='adjusted_market_cap'
                         ) -> pd.DataFrame:
        """
        Summary:
            Given a dataframe of ticker data over time, we compute a subset 
            based on 'all values but momentum' . So we eliminate all entries
            which violate our prices to sales ratio and market cap rules, for
            the tiny titans strategy. 
            
            Note that this function is very simple whereas the momentum computations
            get comparatively complex due to missing data.

        Args:
            df (pd.DataFrame): The df we wish to find a subset of.
            market_cap_min (int): The minimum market cap we consider
            market_cap_max (int): The maximum market cap we consider
            ps_ratio (float): The highest price to sales ratio we allow
            market_cap_type (str): The column name of the market cap type to
                use. We default to the market cap adjusted for multiple tickers
                with the same company.
        Returns:
            pd.DataFrame: with rows which violate the ps ratio and market cap
                thresholds removed.
        """

        df_ps = df[df['PS_ratio'] <= ps_ratio]
        df_ps = df_ps[df_ps['PS_ratio'] >= 0.0]

        df_min = df_ps[df_ps[market_cap_type] >= market_cap_min]

        df_max = df_min[df_min[market_cap_type] <= market_cap_max]

        return df_max

    def determine_momentum_type(self, debug: bool) -> int:
        """
        Summary:
            If there's already a momentum column in the dataframe that means
            when it was scraped the date 52 weeks prior was also scraped so 
            that momentum could be exactly computed. In this case that value
            is just used.

            If there is no momentum column the date 12 months ago is used as a
            proxy. 
        Args: 
            debug (bool): set to True to print momentum approximation
        Returns:
            int: The start index which is 0 if we're using the momentum 
                approximation and 0 otherwise
        """
        if 'momentum' not in list(self.polygon_data.columns):
            momentum_approximation = True
        else:
            momentum_approximation = False

        if debug:
            print(f"Using momentum approximation: {momentum_approximation}")

        if momentum_approximation:
            start_index = 12
            # need to look one year behind with close data
            assert len(self.date_list) > start_index 
        else:
            start_index = 0

        return start_index

    def compute_monthly_TT_df(self, 
                              n: int,
                              momentum_approximation: bool,
                              num_stocks: int) -> pd.DataFrame:
        """
        Summary:
            Computes the pandas df for a particular month of tiny titans
        Args:
            n (int): The index of the date list we're currently computing
            momentum_approximation (bool): whether we're approximating momentum
                using monthly data rather than 52 weeks exactly
            num_stocks (int): number of stocks we hold each month in backtest.
        Returns:
            pd.DataFrame: the df for that month of TT
        """
                            
        date = self.date_list[n]
        df_date = self.polygon_data[self.polygon_data['date'] == date]

        df_max = self.all_but_momentum(df_date)

        if momentum_approximation:
            print('\n\napproximating 52 week momentum by using price index '\
                  +'from 12 months prior\n\n')
            df = self.compute_appreciation_column(df_max, -12, 'momentum')
        else:
            df = df_max
            
        df = df.sort_values('momentum', ascending=True)#False)

        return df[:num_stocks].copy()

    def find_TT(self,
                num_stocks: int=num_stocks_default,
                market_cap_min: int=market_min_default,
                market_cap_max: int=market_max_default,
                ps_ratio: float=ps_ratio_default,
                debug: bool=True) -> Dict:
        """
        Summary:
            Computes the TT stocks for each month in our time frame.

            Stores the results in a which is then returned

        Args:
            num_stocks (int): number of stocks we hold each month in backtest.
            market_cap_min (int): minimum market cap of stocks we consider
            market_cap_max (int): maximum market cap of stocks we consider
            ps_ratio (float): price to sales ratio our stocks must be less than
            debug (bool): set to True to print momentum approximation
        Returns:
            Dict: of with values as dfs containing stats on TT stocks and keys
                are dates about one month apart
        """
        stocks_dict = {}

        start_index = self.determine_momentum_type(debug)

        beginning = self.date_list[start_index]
        n_months = len(self.date_list) - start_index
        end = self.date_list[-1]
        
        print(f"Benchmarking TT performance for {n_months} months from " \
              + f"{beginning} until {end}.\n\n")

        # can't do anything with last month because we need to compute one 
        # month appreciation
        for n in range(start_index, len(self.date_list)-1):
           stocks_dict[date] = self.compute_monthly_TT_df(
                n, 
                momentum_approximation,
                num_stocks
            )

        return stocks_dict

    def appreciation_operation(self, 
                               date: str,
                               stocks_dict: Dict,
                               df: pd.DataFrame,
                               num_stocks: int,
                               missing_appreciation_approximation: float,
                               appreciation_type: str) -> Tuple[Dict,float]:
        """
        Summary:
            This method executes the appreciation operations for run_backtest.

            First we compute the appreciation column.
            Then we correct for nans.
            Next we compute the portfolio appreciation.
            Finally we save the data in a dictionary.
            
        Args:
            date (str): particular date we wish to compute the appreciation for
            stocks_dict (Dict): dictionary containing dfs on TT stocks for each
                month
            num_stocks (int): number of stocks we hold each month in backtest.
            missing_approximation_approximation (float): if we encounter nans,
                how to fill in appreciation.
            appreciation_type (str): appreciation column to use
        Returns:
            Tuple[Dict,float]: 
                Dict - keys are monthly dates, values are df of TT portfolio 
                    on that day
                float - is investment appreciation for this month
        """
            
        df = stocks_dict[date]

        df = self.compute_appreciation_column(df, 
                                              1,
                                              'one_month_appreciation')
        
        df = df.reset_index()
        df, nans_encountered = self.nan_corrector.correct_nans(
            self.date_list,
            df,
            1,
            'one_month_appreciation',
            num_stocks,
            missing_appreciation_approximation
        )

        appreciation = sum(
            df.loc[:num_stocks, appreciation_type]) \
            / \
            min(num_stocks, len(df.index)
        )
        investment *= appreciation

        self.stats_dict[date] = {'monthly_roi' : appreciation - 1.0,
                                 'data' : df,
                                 'nans' : nans_encountered}

        return stats_dict, investment

    def run_backtest(self,
                     missing_appreciation_approximation: Union[float,None]=None,
                     num_stocks: int=num_stocks_default,
                     market_cap_min: int=market_min_default,
                     market_cap_max: int=market_max_default,
                     ps_ratio: float=ps_ratio_default,
                     appreciation_type: str='one_month_appreciation_adjusted'
                     ) -> Dict:
        """
        Summary:
            First we create the stocks_dict which contains a df of the TT stock
            for each month, the date of the snapshot corresponding the the key.

            Then we loop through all dates and compute the appreciation of TT
            over time. The results are saved in the stats dict which is then 
            used in analyze backtest to produce plots and statistics.
            
        Args:
            missing_approximation_penalty (float): if we encounter nans, how to
                fill in appreciation.
            num_stocks (int): number of stocks we hold each month in backtest.
            market_cap_min (int): minimum market cap of stocks we consider
            market_cap_max (int): maximum market cap of stocks we consider
            ps_ratio (float): price to sales ratio our stocks must be less than
            appreciation_type (str): appreciation column to use
        Returns:
            Dict: dictionary with query and result statistics and data
        """
        self.stats_dict = {}

        stocks_dict = self.find_TT(num_stocks,
                                   market_cap_min,
                                   market_cap_max,
                                   ps_ratio)

        if missing_appreciation_approximation is not None:
            print("If one month appreciation is missing we approximate by" \
                  + "assuming stock is sold for " \
                  + f"{missing_appreciation_approximation*100}% of its initial" \
                  + "value.\n\n")
        else:
            print("If one month appreciation is missing we use stock's last "\
                  + "close.\n\n")

        investment = 1.0
        for date in self.date_list:
            
            if date in stocks_dict.keys():
                self.stats_dict, investment = self.appreciation_operation(
                    date,
                    stocks_dict,
                    num_stocks,
                    missing_appreciation_approximation,
                    appreciation_type
                )

        stats_dict['growth_of_dollar'] = investment
        stats_dict = self.backtest_analyzer.analyze_backtest(stats_dict)

        return stats_dict

    @staticmethod
    def BackTestFactory(fp: str,
                        missing_approximation_penalty: float=1.0,
                        num_stocks: int=num_stocks_default,
                        market_cap_min: int=market_min_default,
                        market_cap_max: int=market_max_default,
                        ps_ratio: float= ps_ratio_default,
                        save_fp: Union[str,None]=None) -> Dict:
        """
        Summary:
            Factory method for run_backtest
        Args:
            fp (str): fp to polygon data
            missing_approximation_penalty (float): if we encounter nans, how to
                fill in appreciation.
            num_stocks (int): number of stocks we hold each month in backtest.
            market_cap_min (int): minimum market cap of stocks we consider
            market_cap_max (int): maximum market cap of stocks we consider
            ps_ratio (float): price to sales ratio our stocks must be less than
            save_fp (Union[str,None]): filepath to save our data to
        Returns:
            Dict: dictionary with query and result statistics and data
        """
        
        assert isinstance(missing_approximation_penalty, float)
        assert isinstance(num_stocks, int)
        assert isinstance(market_cap_min, int)
        assert isinstance(market_cap_max, int)
        assert isinstance(ps_ratio, float)

        back_test = BackTest(fp)
        stats_dict = {}

        stats_dict['query'] = {
            'num_stocks' : num_stocks,
            'market_cap_min' : market_cap_min,
            'market_cap_max' : market_cap_max,
            'ps_ratio' : ps_ratio,
            'missing_penalty' : missing_approximation_penalty}

        stats_dict['results'] = back_test.run_backtest(
            missing_approximation_penalty,
            num_stocks,
            market_cap_min,
            market_cap_max,
            ps_ratio)

        if save_fp is not None:
            with open('results/' + save_fp, 'wb') as f:
                pickle.dump(stats_dict, f)
                f.close()

        return stats_dict

if __name__ == '__main__':
    fp = '~/Desktop/table_b_test_long_adjusted_first_month_removed.csv'#
    #fp = '~/Desktop/table_b_test_long_adjusted.csv'

    stats_dict = BackTest.BackTestFactory(fp, None, save_fp = 'non_conservative_missing_tt_test.pickle')#'test_inv_TT_2022-12-03_0_missing_approx.pickle')

    #pprint(stats_dict)
