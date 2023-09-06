import os
import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Union
from TinyTitans.backtesting.polygon_api.getting_ticker_data import get_adjusted_close
from TinyTitans.backtesting.distributions import Distribution
from TinyTitans.backtesting.plot_growth import plot_growth
from TinyTitans.backtesting.polygon_api.utils import *
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
from TinyTitans.backtesting.missing_data_corrections.process_corporate_actions import CA_Parser

num_stocks_default = 10
market_min_default = int(25*10**6)
market_max_default = int(250*10**6)
ps_ratio_default = 1.0

class BackTest:

    def __init__(self, polygon_data: str, save_adjusted: bool=True) -> None:
        
        self.ca_parser = CA_Parser()
        self.polygon_data = self.read_polygon_data(polygon_data)
        self.compute_date_and_ticker_columns()
        self.compute_date_list()
        self.validate_date_list()
        self.adjust_market_cap_for_multiple_tickers(polygon_data, save_adjusted)
        self.compute_PS_ratio_column()

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

    @staticmethod
    def get_cik_list(df: pd.DataFrame) -> list:
        cik_list = list(set(list(df['cik'])))

        return cik_list

    @staticmethod
    def adjust_market_cap_for_multiple_tickers_static(df: pd.DataFrame, date_list: list) -> pd.DataFrame:

        if 'adjusted_market_cap' in list(df.columns):
            return df

        adjusted_dfs = []

        print("adjusting market cap for ciks that have multiple tickers")
        for date in tqdm(date_list):
            cik_mc_map = {}
            df_date = df[df['date'] == date].copy()
            cik_list = BackTest.get_cik_list(df_date)
            
            
            for tuple_key in df_date.index:
                cik = df_date.loc[tuple_key, 'cik']
                market_cap = df_date.loc[tuple_key, 'market_cap']
                if not np.isnan(cik):
                    if cik in cik_mc_map.keys():
                        cik_mc_map[cik]['tuple_key'].append(tuple_key)
                        cik_mc_map[cik]['market_cap'] += market_cap
                    else:
                        cik_mc_map[cik] = {}
                        cik_mc_map[cik]['tuple_key'] = [tuple_key]
                        cik_mc_map[cik]['market_cap'] = market_cap

            df_date.loc[:, 'adjusted_market_cap'] = df_date.loc[:, 'market_cap']

            for tuple_key in df_date.index:
                for cik in cik_mc_map.keys():
                    tuple_keys = cik_mc_map[cik]['tuple_key']
                    adjusted_market_cap = cik_mc_map[cik]['market_cap']
                    if tuple_key in tuple_keys:
                        df_date.loc[tuple_key, 'adjusted_market_cap'] = adjusted_market_cap
                    
            adjusted_dfs.append(df_date)

        df = pd.concat(adjusted_dfs)

        return df

                        
                        
        

    def adjust_market_cap_for_multiple_tickers(self, polygon_data_fp: str, save_adjusted: bool) -> None:
        self.polygon_data = BackTest.adjust_market_cap_for_multiple_tickers_static(self.polygon_data, self.date_list)

        if save_adjusted and 'adjusted.csv' not in polygon_data_fp:
            self.polygon_data.to_csv(polygon_data_fp.replace('.csv', '_adjusted.csv'))

    @staticmethod
    def compute_date_and_ticker_columns_static(df: pd.DataFrame) -> pd.DataFrame:

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

    def compute_date_and_ticker_columns(self) -> None:
        self.polygon_data = BackTest.compute_date_and_ticker_columns_static(self.polygon_data)

    @staticmethod
    def compute_PS_ratio_static(df: pd.DataFrame, market_cap='adjusted_market_cap') -> pd.DataFrame:
        df['PS_ratio'] = df[market_cap] / df['trailing_twelve_month_revenue']

        return df

    def compute_PS_ratio_column(self) -> None:
        self.polygon_data = BackTest.compute_PS_ratio_static(self.polygon_data)

    @staticmethod
    def compute_date_list_static(df: pd.DataFrame) -> list:
        date_list = list(set(list(df['date'])))
        date_list = sorted(date_list)

        return date_list

    def compute_date_list(self, verbose=False) -> None:

        self.date_list = BackTest.compute_date_list_static(self.polygon_data)

        if verbose:
            print("Date Range:")
            pprint(self.date_list)
            print('\n\n')

    def validate_date_list(self) -> None:

        min_time = min(self.date_list)
        max_time = max(self.date_list)

        for n in range(len(self.date_list)-1):
            date1 = self.date_list[n]
            date2 = self.date_list[n+1]

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


        print(f"Date range validated. Data spans from {min_time} until {max_time}.")

    def compute_appreciation_column(self, query_df: pd.DataFrame, offset: int, label: str, data_df: pd.DataFrame=None, debug=False) -> pd.DataFrame:

        if data_df is None:
            data_df = self.polygon_data
        
        if debug:
            print(query_df)

        appreciation_list = []

        # get rid of annoying pandas warning and make sure we're not modifying polygon_data
        query_df = query_df.copy()
        
        for i in query_df.index:

            adjusted_close = query_df.loc[i, 'adjusted_close']

            ticker = query_df.loc[i, 'ticker']
            date = query_df.loc[i, 'date']


            date_index = self.date_list.index(date)
            offset_date = self.date_list[date_index+offset]

            tuple_key = str((ticker, offset_date))

            try:

                
                offset_adj_close = data_df.loc[tuple_key, 'adjusted_close']
                offset_appreciation = BackTest.get_offset_appreciation(offset, adjusted_close, offset_adj_close)

            except KeyError:
                
                tuple_key, action = self.handle_symbol_and_bankruptcy(ticker, date, offset_date)
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
                            offset_adj_close = data_df.loc[tuple_key, 'adjusted_close']
                            offset_appreciation = BackTest.get_offset_appreciation(offset, adjusted_close, offset_adj_close)
                        except KeyError:
                            print("switched symbol missing")
                            offset_appreciation = np.nan
                        

            appreciation_list.append(offset_appreciation)

        query_df[label] = appreciation_list

        return query_df

    def handle_symbol_and_bankruptcy(self, ticker: str, date: str, offset_date: str) -> tuple:
        # adjust for corporate actions
        # symbol changes and bankruptcies

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
    def get_offset_appreciation(offset: int, adjusted_close: float, offset_adj_close: float, eps: float=1e-3) -> float:
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

    @staticmethod
    def all_but_momentum(df : pd.DataFrame,
                         market_cap_min: int = market_min_default,
                         market_cap_max: int = market_max_default,
                         ps_ratio: float = ps_ratio_default,
                         market_cap_type: str = 'adjusted_market_cap') -> pd.DataFrame:
        
        df_ps = df[df['PS_ratio'] <= ps_ratio]
        df_ps = df_ps[df_ps['PS_ratio'] >= 0.0]
        #df_ps = df

        df_min = df_ps[df_ps[market_cap_type] >= market_cap_min]

        df_max = df_min[df_min[market_cap_type] <= market_cap_max]

        return df_max


    def find_TT(self,
                num_stocks: int = num_stocks_default,
                market_cap_min: int = market_min_default,
                market_cap_max: int = market_max_default,
                ps_ratio: float = ps_ratio_default,
                debug: bool = True) -> dict:

        stocks_dict = {}

        if 'momentum' not in list(self.polygon_data.columns):
            momentum_approximation = True
        else:
            momentum_approximation = False

        if debug:
            print(f"Using momentum approximation: {momentum_approximation}")

        if momentum_approximation:

            start_index = 12
        
            assert len(self.date_list) > start_index # need to look one year behind with close data

        else:
            start_index = 0

        beginning = self.date_list[start_index]
        n_months = len(self.date_list) - start_index
        end = self.date_list[-1]
        
        print(f"Benchmarking TT performance for {n_months} months from {beginning} until {end}.\n\n")

        for n in range(start_index, len(self.date_list)-1): # can't do anything with last month because we need to compute one month appreciation

            date = self.date_list[n]
            df_date = self.polygon_data[self.polygon_data['date'] == date]

            df_max = BackTest.all_but_momentum(df_date)

            if momentum_approximation:
                print('\n\napproximating 52 week momentum by using price index from 12 months prior\n\n')
                df = self.compute_appreciation_column(df_max, -12, 'momentum')
            else:
                df = df_max

            #print(df['momentum'])
            #print(df)
                
            df = df.sort_values('momentum', ascending=True)#False)
            #print(df)
            #input()

            stocks_dict[date] = df[:num_stocks].copy()


        return stocks_dict

    def correct_nans(self, df: pd.DataFrame, offset: int, column_name: str, num_stocks: int, missing_approximation: float = None, debug: bool = False) -> (pd.DataFrame, int):
        
        nans_encountered = 0

        df[column_name + '_adjusted'] = df[column_name]

        df_length = len(df.index)

        print(f"only {df_length} / {num_stocks} stocks found")
        
        for n in range(min(num_stocks, df_length)):

            appreciation = df.loc[n, column_name]
            ticker = df.loc[n, 'ticker']
            adjusted_close = df.loc[n, 'adjusted_close']
            date = df.loc[n, 'date']
            date_index = self.date_list.index(date)
            offset_date = self.date_list[date_index+offset]

            if np.isnan(appreciation):
                nans_encountered += 1
                
                if missing_approximation is not None:
                    appreciation = missing_approximation
                else:
                    nearest_adj_close = BackTest.get_nearest_adjusted_close(ticker, date, offset_date)
                    appreciation = BackTest.get_offset_appreciation(offset, adjusted_close, nearest_adj_close)

                    if debug:
                        print(ticker)
                        print(appreciation)
                    
                df.loc[n, column_name + '_adjusted'] = appreciation

        return df, nans_encountered

    @staticmethod
    def get_nearest_adjusted_close(ticker: str, begin_date: str, missing_date: str, cache = True) -> float:

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

        begin_date = datetime.strptime(begin_date,  '%Y-%m-%d') #convert to datetime

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

    def run_backtest(self,
                     missing_appreciation_approximation: float = None,
                     num_stocks: int = num_stocks_default,
                     market_cap_min: int = market_min_default,
                     market_cap_max: int = market_max_default,
                     ps_ratio: float = ps_ratio_default,
                     appreciation_type: str = 'one_month_appreciation_adjusted') -> dict:


        stats_dict = {}

        stocks_dict = self.find_TT(num_stocks,
                                   market_cap_min,
                                   market_cap_max,
                                   ps_ratio)

        if missing_appreciation_approximation is not None:
            print("If one month appreciation is missing we approximate by assuming stock is sold for" \
                  + f" {missing_appreciation_approximation*100}% of its initial value.\n\n")
        else:
            print("If one month appreciation is missing we use stock's last close.\n\n")

        investment = 1.0
        for date in self.date_list:
            appreciation = 0
            nans_encountered = 0
            
            if date in stocks_dict.keys():
                df = stocks_dict[date]

                df = self.compute_appreciation_column(df, 1, 'one_month_appreciation')
                
                df = df.reset_index()
                df, nans_encountered = self.correct_nans(df, 1, 'one_month_appreciation', num_stocks, missing_appreciation_approximation)

                appreciation = sum(df.loc[:num_stocks, appreciation_type]) / min(num_stocks, len(df.index))
                investment *= appreciation

                stats_dict[date] = {'monthly_roi' : appreciation - 1.0, 'data' : df, 'nans' : nans_encountered}

        stats_dict['growth_of_dollar'] = investment

        self.analyze_backtest(stats_dict)

        return stats_dict

    @staticmethod
    def set_tuple_key(df: pd.DataFrame) -> pd.DataFrame:

        tuple_keys = []
        for i in df.index:
            ticker = df.loc[i, 'ticker']
            date = df.loc[i, 'date']
            tk = str((ticker, date))
            tuple_keys.append(tk)

        df['tuple_key'] = tuple_keys
        
        df = df.set_index('tuple_key')

        return df


    @staticmethod
    def rearrange_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        new_dates = []

        for i in df.index:
            date = df.loc[i, date_col]
            month, day, year = date.split('-')
            new_date = '-'.join([year, month, day])

            new_dates.append(new_date)

        df['date'] = new_dates

        return df
            
    
    @staticmethod
    def load_and_prep_AAII(cache_dir: str, dates: list) -> Union[pd.DataFrame, None]:

        first_date = dates[0]
        last_date = dates[-1]


        AAII = pd.read_excel(cache_dir + 'OShaughnessyTinyTitansScreen.xls')

        AAII = BackTest.rearrange_date(AAII, 'DATE')

        AAII = AAII[AAII['date'] >= first_date]
        AAII = AAII[AAII['date'] <= last_date]

        AAII['one_month_appreciation'] = 1 + AAII['MONTHLY PERFORMANCE']/100

        for i in AAII.index:
            date = AAII.loc[i, 'date']
            if date == last_date:
                break
            AAII.loc[i, 'one_month_appreciation'] = AAII.loc[i+1, 'one_month_appreciation']

        AAII = AAII[:-1]

            
        return AAII
        
        

    @staticmethod
    def check_and_load_cache(cache_dir: str, ticker: str, dates: list) -> Union[pd.DataFrame, None]:

        if ticker == 'AAII':
            ticker_df = BackTest.load_and_prep_AAII(cache_dir, dates)

        else:

            cached = os.listdir(cache_dir)

            first_date = dates[0]
            last_date = dates[-1]

            ticker_df = None

            for cache in cached:
                if ticker in cache and first_date in cache and last_date in cache:
                    ticker_df = pd.read_csv(cache_dir + cache)
                    break

        return ticker_df

    def create_and_save_benchmark_returns(self, cache_dir: str, ticker: str, dates: list) -> pd.DataFrame:
        adjusted_closes = []

        for date in dates:
            close = get_adjusted_close(ticker, date)
            adjusted_closes.append(close)

        ticker_df = {'adjusted_close' : adjusted_closes, 'ticker' : len(dates)*[ticker], 'date' : dates}
        ticker_df = pd.DataFrame.from_dict(ticker_df)

        ticker_df = BackTest.set_tuple_key(ticker_df)


        # have to end appreciation computation at last month
        ticker_df = self.compute_appreciation_column(ticker_df[:-1], 1, 'one_month_appreciation', data_df=ticker_df)

        first_date = dates[0]
        last_date = dates[-1]

        ticker_df.to_csv(cache_dir + ticker + '_' + first_date + '_' + last_date + '.csv')

        return ticker_df

        

    def compute_benchmark_returns(self, ticker: str, dates: list) -> list:
        cache_dir = 'benchmark_cache/'
        
        ticker_df = BackTest.check_and_load_cache(cache_dir, ticker, dates)

        if ticker_df is None:
            print(f"Benchmark {ticker} not in cache. Computing now.\n\n")
            ticker_df = self.create_and_save_benchmark_returns(cache_dir, ticker, dates)

        monthly_roi = []
        for i in ticker_df.index:
            roi = (ticker_df.loc[i, 'one_month_appreciation'] - 1.0)*100
            monthly_roi.append(roi)

        return monthly_roi


    def analyze_backtest(self, stats_dict: dict, benchmarks: List = ['VOO', 'VTWO', 'AAII']) -> dict:

        final_date = self.date_list[-1]
        
        analysis_dict = {'date' : [], 'TT_monthly_roi%' : [], 'nans' : []}

        target_dates = set(stats_dict.keys())
        target_dates = sorted(list(target_dates.intersection(set(self.date_list))))

        for date in self.date_list:
            if date in target_dates:
                analysis_dict['date'].append(date)
                analysis_dict['TT_monthly_roi%'].append(stats_dict[date]['monthly_roi']*100)
                analysis_dict['nans'].append(stats_dict[date]['nans'])

        target_dates.append(final_date) # need to add back on last one to compute appreciation
        for benchmark in benchmarks:
            analysis_dict[benchmark + '_monthly_roi%'] = self.compute_benchmark_returns(benchmark, target_dates)

        df = pd.DataFrame.from_dict(analysis_dict)

        plot_growth(df)

        print(df)

        average_nans = np.mean(df['nans'])
        print(f'\n\nAverage nans per month = {average_nans}')

        print(f"\n\nTT $1 becomes : {stats_dict['growth_of_dollar']} on {final_date}")

        for benchmark in benchmarks:
            one_dollar = 1.0
            for i in df.index:
                monthly_roi = df.loc[i, benchmark + '_monthly_roi%']
                monthly_roi = 1 + monthly_roi / 100
                one_dollar *= monthly_roi

            print(f"{benchmark} $1 becomes : {one_dollar}")

        stats = Distribution.DistributionFactory(analysis_dict)

    @staticmethod
    def BackTestFactory(fp: str,
                        missing_approximation_penalty: float = 1.0,
                        num_stocks : Union[int, List[int]] = num_stocks_default,
                        market_cap_min: Union[int, List[int]] = market_min_default,
                        market_cap_max : Union[int, List[int]] = market_max_default,
                        ps_ratio : Union[float, List[float]] = ps_ratio_default,
                        save_fp : str = None) -> dict:

        bt = BackTest(fp)

        stats_dict = {}
        
        if isinstance(num_stocks, int):
            assert isinstance(market_cap_min, int)
            assert isinstance(market_cap_max, int)
            assert isinstance(ps_ratio, float)

            stats_dict['query_0'] = {'num_stocks' : num_stocks,
                                     'market_cap_min' : market_cap_min,
                                     'market_cap_max' : market_cap_max,
                                     'ps_ratio' : ps_ratio,
                                     'missing_penalty' : missing_approximation_penalty}

            stats_dict['results_0']  = bt.run_backtest(missing_approximation_penalty,
                                                       num_stocks,
                                                       market_cap_min,
                                                       market_cap_max,
                                                       ps_ratio)

        elif isinstance(num_stocks, list):
            assert len(num_stocks) == len(market_cap_min)
            assert len(num_stocks) == len(makert_cap_max)
            assert len(num_stocks) == len(ps_ratio)

            for n in range(len(num_stocks)):

                ns = num_stocks[n]
                mc_min = market_cap_min[n] 
                mc_max = market_cap_max[n]
                ps = ps_ratio[n]

                stats_dict[f'query_{n}'] = {'num_stocks' : ns,
                                            'market_cap_min' : mc_min,
                                            'market_cap_max' : mc_max,
                                            'ps_ratio' : ps,
                                            'missing_penalty' : missing_approximation_penalty}
                
                stats_dict[f'results_{n}']  = bt.run_backtest(missing_approximation_penalty,
                                                              ns,
                                                              mc_min,
                                                              mc_max,
                                                              ps)

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
