from TinyTitans.backtesting.missing_data_corrections.process_corporate_actions import CA_Parser
from unittest import TestCase
import unittest

class TestCorporateActions(TestCase):
    """
    Summary:
        Tests CA_Parser
    """

    def test_process_symbol_changes_and_bankruptcies(self):
        """
        Summary:
            Tests the process_symbol_changes_and_bankruptcies function which is
            the function of CA_Parser which the backtester calls.

            There are several different test cases:
                - IKNX 2021-11-01 -> 2021-12-16
                    This should catch a symbol change to WULF
                - IKNX 2021-12-14 -> 2021-12-14
                    This should also catch a symbol change to WULF
                    Wanted to make sure it would work if we only give it the 
                        day of
                - IKNX 2012-11-01 -> 2021-12-16
                    This should also catch a symbol change to WULF
                    Wanted to make sure it would work if given many months of 
                        irrelevant data
                - HLCSQ 2012-12-27 -> 2013-12-28
                    This should catch a bankruptcy
        """
        ca_parser = CA_Parser()

        ticker = 'IKNX'

        date_one = '2021-11-01'
        date_two = '2021-12-16'
        action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
        self.assertEqual(action, 'symbol_change:IKNX->WULF')

        date_one = '2021-12-14'
        date_two = '2021-12-14'
        action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
        self.assertEqual(action, 'symbol_change:IKNX->WULF')

        date_one = '2012-11-01'
        date_two = '2021-12-16'
        action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
        self.assertEqual(action, 'symbol_change:IKNX->WULF')

        date_one = '2012-11-01'
        date_two = '2020-12-16'
        action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
        self.assertTrue(action is None)

        ticker = 'HLCSQ'
        date_one = '2012-12-27'
        date_two = '2013-12-28'

        action = ca_parser.process_symbol_changes_and_bankruptcies(ticker, date_one, date_two)
        self.assertEqual(action, 'bankruptcy')


if __name__ == '__main__':
    unittest.main()