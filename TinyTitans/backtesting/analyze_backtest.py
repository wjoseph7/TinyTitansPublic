from typing import List, Dict
from TinyTitans.backtesting.backtest import BackTest
from TinyTitans.backtesting.polygon_api.ticker_api_calls import get_adjusted_close
from TinyTitans.backtesting.distributions import Distribution
from TinyTitans.backtesting.plot_growth import plot_growth

class BackTestAnalyzer:

    def __init__(self, back_test: BackTest):
        self.back_test = back_test

    def set_tuple_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary:
            Creates a new index 'tuple_key' which is '(ticker,date)'
        Args:
            df (pd.DataFrame): df we want to change the index of
        Returns:
            pd.DataFrame: same df but with tuple_key as the index
        """
        tuple_keys = []
        for i in df.index:
            ticker = df.loc[i, 'ticker']
            date = df.loc[i, 'date']
            tk = str((ticker, date))
            tuple_keys.append(tk)

        df['tuple_key'] = tuple_keys
        
        df = df.set_index('tuple_key')

        return df

    def rearrange_date(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Summary:
            Creates a new column in the df called date in which the previous
            date column is rearranged from month-day-year to year-month-day
        Args:
            df (pd.DataFrame): df with different date format
            date_col (str): column name of current date format
        Returns:
            pd.DataFrame: same df with new year-month-day column
        """
        new_dates = []

        for i in df.index:
            date = df.loc[i, date_col]
            month, day, year = date.split('-')
            new_date = '-'.join([year, month, day])

            new_dates.append(new_date)

        df['date'] = new_dates

        return df
            
    def load_and_prep_AAII(self,
                           cache_dir: str,
                           dates: List) -> pd.DataFrame:
        """
        Summary:
            Loads AAII data from cache and computes one_month_appreciation
            column.
        Args:
            cache_dir (str): filepath to AAII TinyTitans data
            dates (List): list of dates for which we want AAII returns
        Returns:
            pd.DataFrame: df containing AAII over the given dates with a 
                one_month_appreciation column
        """
        first_date = dates[0]
        last_date = dates[-1]

        AAII = pd.read_excel(cache_dir + 'OShaughnessyTinyTitansScreen.xls')

        AAII = self.rearrange_date(AAII, 'DATE')

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
        
    def check_and_load_cache(self,
                             cache_dir: str,
                             ticker: str,
                             dates: List) -> Union[pd.DataFrame, None]:
        """
        Summary:
            Checks if a benchmark exists in the cache and returns a df with the
            data if so. Otherwise returns None.
        Args:
            cache_dir (str): filepath to cached benchmark data
            ticker (str): benchmark ETF we want to compare to or AAII
            dates (List): list of dates for which we want benchmark returns
        Returns:
            Union[pd.DataFrame, None]: returns 
                - pd.DataFrame with the cache data
                - None if benchmark not in cache
        """
        if ticker == 'AAII':
            ticker_df = self.load_and_prep_AAII(cache_dir, dates)

        else:
            cached = os.listdir(cache_dir)

            first_date = dates[0]
            last_date = dates[-1]

            ticker_df = None

            for cache in cached:
                if ticker in cache and \
                    first_date in cache and \
                    last_date in cache:
                    ticker_df = pd.read_csv(cache_dir + cache)
                    break

        return ticker_df

    def create_and_save_benchmark_returns(self,
                                          cache_dir: str,
                                          ticker: str,
                                          dates: List) -> pd.DataFrame:
        """
        Summary:
            Creates and saves df with benchmark returns.
        Args:
            cache_dir (str): filepath to cache dir we will save to
            ticker (str): benchmark ETF we want to compare to
            dates (List): list of dates for which we want benchmark returns
        Returns:
            pd.DataFrame: with benchmark data
        """
        adjusted_closes = []

        for date in dates:
            close = get_adjusted_close(ticker, date)
            adjusted_closes.append(close)

        ticker_df = {'adjusted_close' : adjusted_closes,
                     'ticker' : len(dates)*[ticker],
                     'date' : dates}
        ticker_df = pd.DataFrame.from_dict(ticker_df)

        ticker_df = self.set_tuple_key(ticker_df)

        # have to end appreciation computation at last month
        ticker_df = self.back_test.compute_appreciation_column(
            ticker_df[:-1],
            1,
            'one_month_appreciation',
            data_df=ticker_df
        )

        first_date = dates[0]
        last_date = dates[-1]

        ticker_df.to_csv(cache_dir + ticker + '_' + first_date + '_' +\
            last_date + '.csv')

        return ticker_df        

    def compute_benchmark_returns(self, ticker: str, dates: List) -> List:
        """
        Summary:
            Loads or scrapes list of ROI for given benchmarket ETF ticker and
            dates in date list
        Args:
            cache_dir (str): filepath to cache dir we will load / save to
            ticker (str): benchmark ETF we want to compare to or AAII
            dates (List): list of dates for which we want benchmark returns
        Returns:
            List: List with monthly ROI
        """
        cache_dir = 'benchmark_cache/'
        
        ticker_df = self.check_and_load_cache(cache_dir, ticker, dates)

        if ticker_df is None:
            print(f"Benchmark {ticker} not in cache. Computing now.\n\n")
            ticker_df = self.create_and_save_benchmark_returns(cache_dir, ticker, dates)

        monthly_roi = []
        for i in ticker_df.index:
            roi = (ticker_df.loc[i, 'one_month_appreciation'] - 1.0)*100
            monthly_roi.append(roi)

        return monthly_roi

    def make_analysis_df(self, 
                         final_date: str,
                         date_list: List,
                         stats_dict: Dict, 
                         benchmarks: List=['VOO','VTWO','AAII']) \
                         -> pd.DataFrame:
        """
        Summary:
            Adds benchmark data into the analysis dict and turns it into a df
            for further processing and plotting
        Args:
            stats_dict (Dict): dictionary with date keys and values are ticker
                data, monthly roi, and nan count
            benchmarks (List): List of benchmark tickers (or AAII) to compare
                strategy to
        Returns:
            pd.DataFrame: with  TT and benchmark dates roi and nan count
        """
        analysis_dict = {'date' : [], 'TT_monthly_roi%' : [], 'nans' : []}

        target_dates = set(stats_dict.keys())
        target_dates = sorted(list(target_dates.intersection(set(date_list))))

        for date in date_list:
            if date in target_dates:
                analysis_dict['date'].append(date)
                analysis_dict['TT_monthly_roi%'].append(
                    stats_dict[date]['monthly_roi']*100
                )
                analysis_dict['nans'].append(stats_dict[date]['nans'])

        # need to add back on last one to compute appreciation
        target_dates.append(final_date) 
        for benchmark in benchmarks:
            analysis_dict[benchmark + '_monthly_roi%'] = \
                self.compute_benchmark_returns(benchmark, target_dates)

        df = pd.DataFrame.from_dict(analysis_dict)

        return df


    def analyze_backtest(self,
                         plot_label: str,
                         stats_dict: Dict, 
                         benchmarks: List=['VOO','VTWO','AAII']) -> Dict:
        """
        Summary:
            Compares strategy against benchmarks in terms of roi, nan count,
            and distribution of returns. Plots returns over time and computes
            ditribution statistics.
        Args:
            plot_label (str): label to save plot with
            stats_dict (Dict): dictionary with date keys and values are ticker
                data, monthly roi, and nan count
            benchmarks (List): List of benchmark tickers (or AAII) to compare
                strategy to
        Returns:
            Dict: statistics dictionary with added monthly returns, benchmark 
                returns, nan counts, and distribution data
        """
        date_list = self.back_test.date_list
        final_date = date_list[-1]

        df = pd.DataFrame.from_dict(analysis_dict)

        df = self.make_analysis_df(final_date,
                                   date_list,
                                   stats_dict,
                                   benchmarks)

        plot_growth(plot_label, df)

        print(df)

        average_nans = np.mean(df['nans'])
        print(f'\n\nAverage nans per month = {average_nans}')

        print(f"\n\nTT $1 becomes : {stats_dict['growth_of_dollar']}" + \
            f" on {final_date}")

        for benchmark in benchmarks:
            one_dollar = 1.0
            for i in df.index:
                monthly_roi = df.loc[i, benchmark + '_monthly_roi%']
                monthly_roi = 1 + monthly_roi / 100
                one_dollar *= monthly_roi

            print(f"{benchmark} $1 becomes : {one_dollar}")

        stats = Distribution.DistributionFactory(analysis_dict)

        return stats