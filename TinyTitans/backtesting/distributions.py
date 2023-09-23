from scipy.stats import norm, rv_continuous
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


class Distribution:
    """
    Summary:
        This class collects statistics from a given dictionary of returns and 
        creates plots and computes the best fit of a given distribution.
    """

    def __init__(self, data: Dict, distribution: rv_continuous=norm):
        """
        Summary:
            Creates Distribution instance
        Args:
            data (Dict): dictionary containing return data
            distribution (rv_continuous): what distribution to use to model 
                return statistics
        """
        self.distribution = distribution
        self.data = data
        self.compute_MLE()
        self.compute_stats()

    def compute_MLE(self) -> None:
        """
        Summary:
            Fits data to MLE of given distribution.
            Fits different distribution for each key/index in data dictionary
        Args:
            None
        Returns:
            None
        """
        self.stats = {}
        
        for index in self.data.keys():
            index_data = self.data[index]
            min_, max_ = min(index_data), max(index_data)
            
            loc, scale = self.distribution.fit(index_data)

            self.stats[index] = {'loc' : loc,
                                 'scale' : scale,
                                 'min' : min_,
                                 'max' : max_}

    def compute_stats(self, CI: float=0.95) -> None:
        """
        Summary:
            Computes the following stats for the return distributions:
                - mean
                - median
                - std
                - CI
                - sharpe ratio
                - expected annual return / max monthly drawdown
        Args:
            CI (float): confidence intervals to compute
        Returns:
            None
        """
        for index in self.stats.keys():
            loc = self.stats[index]['loc']
            scale = self.stats[index]['scale']

            mean = self.distribution.mean(loc, scale)
            median = self.distribution.median(loc, scale)
            std = self.distribution.std(loc, scale)

            ci = self.distribution.interval(CI, loc, scale)

            self.stats[index]['mean'] = mean
            self.stats[index]['median'] = median
            self.stats[index]['std'] = std

            self.stats[index][f'CI_{str(CI)}'] = ci
            self.stats[index]['sharpe_ratio'] = mean/std

            expected_yearly_return = ((1.0 + mean/100)**12 - 1)*100
            max_drawdown = -self.stats[index]['min']

            self.stats[index]['expected_yearly_return/max_drawdown'] = \
                expected_yearly_return/max_drawdown

    def plot_distribution_and_histogram(self, savedir: str='results/') -> None:
        """
        Summary:
            Save plots of distributions and histograms for each key in stats
            dict
        Args:
            savedir (str): directory we're saving plots to
        Returns:
            None
        """
        indices = []
        xs = []
        ps = []
        
        for index in self.stats.keys():

            loc = self.stats[index]['loc']
            scale = self.stats[index]['scale']

            plt.hist(self.data[index], density=True, alpha=0.6, color='b')

            xmin, xmax = plt.xlim()

            x = np.linspace(xmin, xmax, 100)
            p = self.distribution.pdf(x, loc, scale)

            indices.append(index)
            xs.append(x)
            ps.append(p)

            plt.plot(x, p, 'k')
            
            plt.title(index)
            plt.savefig(savedir + index + '.png')
            plt.clf()

        
        for index, x, p in zip(indices, xs, ps):
            plt.plot(x, p, label=index)

        plt.legend()
        benchmarks = '_vs_'.join(indices)
        plt.savefig(savedir + benchmarks + '.png')

    def display_data(self) -> Dict:
        """
        Summary:
            Plots distribution and histogram data and prints and returns stats
            dict.
        Args:
            None
        Returns:
            Dict: stats Dict
        """
        self.plot_distribution_and_histogram()
        print("\n\n")
        pprint(self.stats)

        return self.stats

    @staticmethod
    def DistributionFactory(data: Dict) -> Dict:
        """
        Summary:
            Factory method which executes display_data and returns the stats
            dict
        Args:
            data (Dict): dictionary of returns data
        Returns:
            Dict: stats dict
        """
        keys = list(data.keys())

        for key in keys:
            if 'roi' not in key:
                del data[key]

        dist = Distribution(data)
        stats = dist.display_data()

        return stats
