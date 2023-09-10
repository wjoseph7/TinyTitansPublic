from TinyTitans.backtesting.missing_data_corrections.adjust_market_cap import MarketCapAdjuster
from unittest import TestCase
import unittest
import numpy as np
import pandas as pd

class TestMarketCapAdjuster(TestCase):
    """
    Summary:
        Tests MarketCapAdjuster

        # Todo: add further unit tests for component functions
    """

    def test_adjust_market_cap_for_multiple_tickers(self):
        """
        Summary:

            Creates a test df to see if market caps are adjusted correctly.

            Currently a large test for the entire functionality is implemented.

            We cover the following cases:

                - Market cap should be adjusted to add two tickers.
                -  Check the unadjusted market cap is untouched
                -  If there are two tickers to be added and one is nan, the 
                   adjusted is nan
                - Same cik or tickers and different days don't interfere
                - We don't use datapoints with nan ciks
                - singular ticker ciks should be left alone
            
        """
        dates = ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', 
                 '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-06',
                  '2023-01-06']
        tickers = ['GOOG', 'GOOGL', 'GOOG', 'GOOGL', 'AA', 'AAB', 'C', 'C',
                   'CA']
        market_cap = [1, 1, 1, np.nan, 1, 2, 3, 4, 2]
        cik = [0, 0, 0, 0, 1, 1, 2, 2, np.nan]
        tuple_keys = [str((ticker, date)) for date, ticker in zip(dates, tickers)]

        data = {'date' : dates, 'market_cap' : market_cap, 'cik' : cik, 'tuple_key' : tuple_keys}
        df = pd.DataFrame.from_dict(data)
        df = df.set_index('tuple_key')

        date_list = sorted(list(set(dates)))
        
        df = MarketCapAdjuster.adjust_market_cap_for_multiple_tickers_static(
            df,
            date_list,
            'test',
            False)

        # check to see if adjusted market caps are correct
        self.assertEqual(df.loc["('GOOG', '2023-01-01')", 'adjusted_market_cap'], 2)
        self.assertEqual(df.loc["('GOOGL', '2023-01-01')", 'adjusted_market_cap'], 2)

        # same example as above, check to see if unadjusted market caps are untouched
        self.assertEqual(df.loc["('GOOG', '2023-01-01')", 'market_cap'], 1)
        self.assertEqual(df.loc["('GOOGL', '2023-01-01')", 'market_cap'], 1)

        # both should be nan if one is nan
        self.assertTrue(np.isnan(df.loc["('GOOG', '2023-01-02')", 'adjusted_market_cap']))
        self.assertTrue(np.isnan(df.loc["('GOOGL', '2023-01-02')", 'adjusted_market_cap']))

        # different days shouldn't interfere, even with same cik
        self.assertEqual(df.loc["('AA', '2023-01-01')", 'adjusted_market_cap'], 1)
        self.assertEqual(df.loc["('AAB', '2023-01-02')", 'adjusted_market_cap'], 2)

        # if cik is nan we don't use that datapoint
        self.assertEqual(df.loc["('C', '2023-01-06')", 'adjusted_market_cap'], 4)
        self.assertEqual(df.loc["('CA', '2023-01-06')", 'adjusted_market_cap'], 2)

        # singular tickers should be left alone, adjusted and unadjusted market
        #  cap are the same
        self.assertEqual(df.loc["('C', '2023-01-02')", 'adjusted_market_cap'], 3)
        self.assertEqual(df.loc["('C', '2023-01-02')", 'market_cap'], 3)

if __name__ == '__main__':
    unittest.main()