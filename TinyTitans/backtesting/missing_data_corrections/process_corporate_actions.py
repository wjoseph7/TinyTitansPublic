import pandas as pd
import os
from pprint import pprint
from typing import Union, Dict

month_map = {'Jan' : '01',
             'Feb' : '02',
             'Mar' : '03',
             'Apr' : '04',
             'May' : '05',
             'Jun' : '06',
             'Jul' : '07',
             'Aug' : '08',
             'Sep' : '09',
             'Oct' : '10',
             'Nov' : '11',
             'Dec' : '12'}

def zero_pad(day: str) -> str:
    """
    Summary:
        Just pads 0 to the beginning of the day if the number is under 10.
    Args:
        day (str): day as a numerical string
    Returns:
        str: 0 padded day string
    """

    assert isinstance(day, str)
    assert int(day) <= 31 and int(day) > 0

    if len(day) == 1:
        day = '0' + day

    return day

def convert_date(date: str) -> str:
    """
    Summary:
        Converts date str from 'month, D, YYYY' -> YYYY-MM-DD
    Args:
        date (str): Date str in month, D, YYYY format
    Returns:
        str: Date str in YYYY-MM-DD format
    """

    date = date.replace(',', '')
    month, day, year = date.split(' ')

    month = month_map[month]
    day = zero_pad(day)

    date = year + '-' + month + '-' + day

    return date

class CA_Parser:
    """
    Summary:
        The main purpose of this class is to 
    """

    def __init__(self):
        """
        Summary:
            Loads corporate actions and stores them in the ca_dict field
        """
        self.ca_dict = self.load_corporate_actions()

    def convert_df_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summary:
            Converts date into YYYY-MM-DD format for entire column of dataframe

        Args:
            df (pd.DataFrame): df with Date column
        Returns:
            pd.DataFrame: df with new 'date' column with YYYY-MM-DD formatting
        """
        new_dates = []

        for n in df.index:
            new_dates.append(convert_date(df.loc[n, 'Date']))

        df['date'] = new_dates

        return df

    def load_corporate_actions(self) -> Dict:
        """
        Summary:
            This method loads all corporate actions in the CA directory, for 
            each file it creates a dataframe which it stores by its year.

            It also creates the standardized YYYY-MM-DD date column to make 
            things easier.

        Args:
            None
        Returns:
            Dict: a dictionary storing the corporate action dfs by year
        """

        ca_dir = dirname = os.path.dirname(__file__) + '/' + 'corporate_actions/'

        ca_dict = {}
        
        for csv in os.listdir(ca_dir):

            year = csv.split('-')[1].replace('.csv', '')
            df = pd.read_csv(ca_dir + csv)

            # convert date to standard format
            df = self.convert_date_df(df)
            
            ca_dict[year] = df
            
        return ca_dict

    def process_symbol_changes_and_bankruptcies(self,
                                                ticker: str,
                                                date_one: str,
                                                date_two: str) \
                                                -> Union[None, str]:
        """
        Summary:
            This method returns a str indicating if the ticker underwent a 
            bankruptcy or changed symbols between date_one and date_two

        Args:
            ticker (str): the ticker we're interested in
            date_one (str): The lower bound for our query in YYYY-MM-DD format
            date_two (str): The upper bound for our query in YYYY-MM-DD format
        Returns:
            Union[None, str]: 
                - None is returned if there are no recorded bankruptcies or
                  symbol changes for the ticker otherwise
                - str containing bankruptcies or symbol changes 
                  between the two dates
        """

        df = self.filter_corporate_actions(ticker, date_one, date_two)
        action = self.return_action_str(df, ticker)

        return action


    def filter_corporate_actions_setup(self, date_one: str, date_two: str) \
        -> Tuple[str,str]:
        """
        Summary:
            This method extracts the first and second year for the query and
            ensures that year_one <= year_two as well as being contained within
            the range of our corporate action dataset

        Args:
            date_one (str): The lower bound for our query in YYYY-MM-DD format
            date_two (str): The upper bound for our query in YYYY-MM-DD format
        Returns:
            Tuple[str, str]: first year and second year for query
        """
        year_one = date_one.split('-')[0]
        year_two = date_two.split('-')[0]

        ca_years = list(self.ca_dict.keys())
        ca_years = sorted(ca_years)
        
        min_year = min(ca_years)
        max_year = max(ca_years)

        # double check our dates make sense
        assert year_one <= year_two
        assert year_one >= min_year and year_two <= max_year, \
            f"{year_one}, {min_year}, {year_two}, {max_year}"

        return year_one, year_two

    def filter_corporate_actions_inner_loop(self,
                                            year: str,
                                            ticker: str,
                                            date_one: str,
                                            date_two: str) -> pd.DataFrame:
        """
        Summary:
            This method executes the inner loop for the 
            filter_corporate_actions method. All that's done is to remove 
            irrelevant entries from the dataframe. So we only keep data within
            the selected time frame for the selected ticker.

        Args:
            year (str): The year in YYYY-MM-DD we are currently querying
                        This is the index of the outer loop
            ticker (str): the ticker we're interested in
            date_one (str): The lower bound for our query in YYYY-MM-DD format
            date_two (str): The upper bound for our query in YYYY-MM-DD format
        Returns:
            pd.DataFrame: df for the year with irrelevant entries removed
        """

        year_df = self.ca_dict[year]

        ticker_actions = year_df[year_df['date'] >= date_one].copy()
        ticker_actions = ticker_actions[ticker_actions['date'] <= date_two]

        for n in ticker_actions.index:
            symbol = ticker_actions.loc[n, 'Symbol']
            action = ticker_actions.loc[n, 'Action']

            if ticker == symbol or ticker in action.split(' '):
                is_relevant.append(True)
            else:
                is_relevant.append(False)

        ticker_actions['relevant'] = is_relevant
        df = ticker_actions[ticker_actions['relevant']]

        return df

    def filter_corporate_actions(self,
                                 ticker: str,
                                 date_one: str,
                                 date_two: str) -> Union[None, pd.DataFrame]:
        """
        Summary:
            We remove all entries which aren't for a given ticker with the 
            specified dates and return the info in a consolidated dataframe.

        Args:
            ticker (str): the ticker we're interested in
            date_one (str): The lower bound for our query in YYYY-MM-DD format
            date_two (str): The upper bound for our query in YYYY-MM-DD format
        Returns:
            Union[None, pd.DataFrame]: 
                None - if there are no relevant entries otherwise
                df - consolidating all years with irrelevant entries 
                     removed
        """

        data_dict = {}

        year_one, year_two = self.filter_corporate_actions_setup(date_one, date_two)

        ticker_action_dfs = []
        for year in ca_years:

            is_relevant = []
            if year >= year_one and year <= year_two:

                df = self.filter_corporate_actions_inner_loop(year, ticker, date_one, date_two)

                ticker_action_dfs.append(df)

        df = pd.concat(ticker_action_dfs)

        if df.empty:
            df = None

        return df

    def return_action_str(self,
                          df: pd.DataFrame,
                          ticker: str,
                          debug: bool=False) -> Union[None, str]:
        """
        Summary:
            This further filters down the df returned by 
            filter_corporate_actions to include only entries where the action 
            is bankruptcy or a symbol change. Then it returns a string which
            indicates whether there was a bankruptcy or symbol change

        Args:
            df (pd.DataFrame): The df we want to filter. Must have a Type 
                and Action columns
            ticker (str): the ticker we're interested in
            debug (bool): prints the df and ticker if debug is True
        Returns:
            Union[None, str]: 
                - None if there were no bankruptcies or symbol changes in the 
                  df
                - str which is 'bankruptcy' if there was a bankruptcy or 
                  SYM_OLD->SYM_NEW if there was a symbol change and no 
                  bankruptcy
        """

        if debug:
            print(df)
            print(ticker)
        
        if df is None:
            return None

        df_bankruptcy = df[df['Type'] == 'Bankruptcy']
        df_symbol_change = df[df['Type'] == 'Symbol Change']

        if not df_bankruptcy.empty:
            return 'bankruptcy'

        elif not df_symbol_change.empty:
            index_zero = df_symbol_change.index[0]
            most_recent_sc = df_symbol_change.loc[index_zero, 'Action']

            start_ticker = most_recent_sc.split(' ')[0]
            end_ticker = most_recent_sc.split(' ')[-1]

            return f'symbol_change:{start_ticker}->{end_ticker}'

        else:
            return None

        

            

if __name__ == '__main__':
    
    ca_parser = CA_Parser()

    #pprint(ca_parser.ca_dict)

    ticker = 'IKNX'

    date_one = '2021-11-01'
    date_two = '2021-12-16'
    action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
    assert action == 'symbol_change:IKNX->WULF'

    date_one = '2021-12-14'
    date_two = '2021-12-14'
    action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
    assert action == 'symbol_change:IKNX->WULF'

    date_one = '2012-11-01'
    date_two = '2021-12-16'
    action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
    assert action == 'symbol_change:IKNX->WULF'

    date_one = '2012-11-01'
    date_two = '2020-12-16'
    action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
    assert action is None

    ticker = 'HLCSQ'
    date_one = '2012-12-27'
    date_two = '2013-12-28'

    action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
    assert action == 'bankruptcy'

    print('Tests passed!!')
