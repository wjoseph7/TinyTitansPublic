import pandas as pd
from typing import List
from tqdm import tqdm
import numpy as np

class MarketCapAdjuster:
    """
    Summary:
        This class adjusts the market caps for companies with multiple tickers.
        This is done by just summing the market caps for tickers with the 
        same ciks.
    """

    def __init__(self, 
                 df: pd.DataFrame,
                 date_list: List,
                 polygon_data_fp: str,
                 save_adjusted: bool):
        """
        Summary:
            The constructor just saves the variables to self for reference 
            later.
        Args:
            df (pd.DataFrame): df with our polygon data
            date_list (List): Ordered list of dates in our polygon data
            polygon_data_fp (str): path to our polygon data
            save_adjusted (bool): whether to save the market cap adjusted data
        """
        self.df = df
        self.date_list = date_list
        self.polygon_data_fp = polygon_data_fp
        self.save_adjusted = save_adjusted

    def save_adjusted_df(self, polygon_data : pd.DataFrame) -> None:
        """
        Summary:
            This just saves the adjusted polygon data to the same fp but with 
            _adjusted.csv replacing the current suffix.

        Args:
            polygon_data (pd.DataFrame): df with our adjusted polygon data
        Returns:
            None
        """
        if self.save_adjusted and 'adjusted.csv' not in self.polygon_data_fp:
            polygon_data.to_csv(self.polygon_data_fp.
                replace('.csv', '_adjusted.csv'))

    def get_cik_list(self, df: pd.DataFrame) -> List:
        """
        Summary:
            Returns a list of unique ciks contained in the df.
        Args:
            df (pd.DataFrame): with cik column we wish to extract
        Returns:
            List: of unique ciks
        """
        cik_list = list(set(list(df['cik'])))

        return cik_list

    def adjust_market_cap_inner_loops(self) -> None:
        """
        Summary:
            This class encapsulates most of the mc adjustment logic. 

            The first loop updates the cik_mc_map with a list of ticker, date
            tuples which share the same cik and a running sum of their market 
            caps.

            The second loop updates the dataframe with a new
            adjustet_market_cap column.
        Args:
            None
        Returns:
            None
        """
        for tuple_key in self.df_date.index:
            cik = self.df_date.loc[tuple_key, 'cik']
            market_cap = self.df_date.loc[tuple_key, 'market_cap']
            if not np.isnan(cik):
                if cik in self.cik_mc_map.keys():
                    self.cik_mc_map[cik]['tuple_key'].append(tuple_key)
                    self.cik_mc_map[cik]['market_cap'] += market_cap
                else:
                    self.cik_mc_map[cik] = {}
                    self.cik_mc_map[cik]['tuple_key'] = [tuple_key]
                    self.cik_mc_map[cik]['market_cap'] = market_cap

        self.df_date.loc[:, 'adjusted_market_cap'] = \
            self.df_date.loc[:, 'market_cap']

        for tuple_key in self.df_date.index:
            for cik in self.cik_mc_map.keys():
                tuple_keys = self.cik_mc_map[cik]['tuple_key']
                adjusted_market_cap = self.cik_mc_map[cik]['market_cap']
                if tuple_key in tuple_keys:
                    self.df_date.loc[tuple_key, 'adjusted_market_cap'] = \
                        adjusted_market_cap


    def adjust_market_cap_for_multiple_tickers(self, 
                                               df: pd.DataFrame,
                                               date_list: List,
                                               polygon_data_fp: str,
                                               save_adjusted: bool) \
                                               -> pd.DataFrame:
        """
        Summary:
            This method adjusts the market cap for companies with multiple 
            tickers. This is done by just adding the market caps of tickers 
            that share the same cik.

            First we check whether the data has already been adjusted. Then
            if not we iterate over the dates and adjust the market cap using 
            the inner loop logic.

            Once that is done all the dataframes are concatenated and the 
            results are saved and returned.

        Args:
            df (pd.DataFrame): df with our polygon data
            date_list (List): Ordered list of dates in our polygon data
            polygon_data_fp (str): path to our polygon data
            save_adjusted (bool): whether to save the market cap adjusted data
        Returns:
            pd.DataFrame: waith adjusted market cap column
        """

        if 'adjusted_market_cap' in list(df.columns):
            return df

        adjusted_dfs = []

        print("adjusting market cap for ciks that have multiple tickers")
        for date in tqdm(date_list):
            self.cik_mc_map = {}
            self.df_date = df[df['date'] == date].copy()
            self.cik_list = self.get_cik_list(self.df_date)
            
            self.adjust_market_cap_inner_loops()
                    
            adjusted_dfs.append(self.df_date)

        polygon_data = pd.concat(adjusted_dfs)

        self.save_adjusted_df(polygon_data)

        return polygon_data

    @staticmethod
    def adjust_market_cap_for_multiple_tickers_static(df: pd.DataFrame,
                                                      date_list: List,
                                                      polygon_data_fp: str,
                                                      save_adjusted: bool) \
                                                      -> pd.DataFrame:
        """
        Summary:
            Static factory method to create market_cap_adjuster class and call
            the adjust_market_cap_for_multiple_tickers method, then returning
            the result.
        Args:
            df (pd.DataFrame): df with our polygon data
            date_list (List): Ordered list of dates in our polygon data
            polygon_data_fp (str): path to our polygon data
            save_adjusted (bool): whether to save the market cap adjusted data
        Returns:
            pd.DataFrame: with adjusted market cap column
        """
        mc_adjuster = MarketCapAdjuster(df,
                                        date_list,
                                        polygon_data_fp,
                                        save_adjusted)

        df = mc_adjuster.adjust_market_cap_for_multiple_tickers(df,
                                                                date_list,
                                                                polygon_data_fp,
                                                                save_adjusted)

        return df