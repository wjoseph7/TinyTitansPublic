from TinyTitans.util.util import *
from unittest import TestCase
import unittest

class TestUtil(TestCase):
    """
    Summary:
        Tests util functions
    """

    def test_get_TT_dir(self):
        """
        Summary:
            Checks to see if the TT home dir is returned correctly
        """
        TT_dir = get_TT_dir()
        self.assertEqual(TT_dir, 
            '/home/joe/TT_Projects/TinyTitansPublic/TinyTitans/')

    def test_get_CA_dir(self):
        """
        Summary:
            Checks to see if the corporate actions dir is returned correctly
        """
        CA_dir = get_CA_dir()
        self.assertEqual(CA_dir, 
            '/home/joe/TT_Projects/TinyTitansPublic/TinyTitans/data/' + \
            'corporate_actions/')


if __name__ == '__main__':
    unittest.main()