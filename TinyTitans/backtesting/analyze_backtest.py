from typing import List, Dict
from TinyTitans.backtesting.backtest import BackTest


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
            cache_dir (str): filepath to TinyTitans data
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
        Args:
        Returns:
            Union[pd.DataFrame, None]:
        """
        if ticker == 'AAII':
            ticker_df = self.load_and_prep_AAII(cache_dir, dates)

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

    def create_and_save_benchmark_returns(self,
                                          cache_dir: str,
                                          ticker: str,
                                          dates: List) -> pd.DataFrame:
        """
        Summary:
        Args:
        Returns:

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

        

    def compute_benchmark_returns(self, ticker: str, dates: list) -> list:
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


    def analyze_backtest(self,
                         stats_dict: Dict, 
                         benchmarks: List =['VOO','VTWO','AAII']) -> Dict:
        """
        Summary:
        Args:
        Returns:
            Dict?
        """
        date_list = self.back_test.date_list
        final_date = date_list[-1]
        
        analysis_dict = {'date' : [], 'TT_monthly_roi%' : [], 'nans' : []}

        target_dates = set(stats_dict.keys())
        target_dates = sorted(list(target_dates.intersection(set(date_list))))

        for date in date_list:
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