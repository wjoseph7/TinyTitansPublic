from scipy.stats import norm, rv_continuous
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np


class Distribution:

    def __init__(self, data: dict, distribution: rv_continuous = norm) -> None:
        self.distribution = distribution
        self.data = data
        self.compute_MLE()
        self.compute_stats()

    def compute_MLE(self) -> None:
        self.stats = {}
        
        for index in self.data.keys():
            index_data = self.data[index]
            min_, max_ = min(index_data), max(index_data)
            
            loc, scale = self.distribution.fit(index_data)

            self.stats[index] = {'loc' : loc, 'scale' : scale, 'min' : min_, 'max' : max_}

    def compute_stats(self, CI: float = 0.95) -> None:

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

            self.stats[index]['expected_yearly_return/max_drawdown'] = expected_yearly_return/max_drawdown

    def plot_distribution_and_histogram(self, savedir: str = 'results/') -> None:

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

    def display_data(self) -> dict:

        self.plot_distribution_and_histogram()
        print("\n\n")
        pprint(self.stats)

        return self.stats

    @staticmethod
    def DistributionFactory(data: dict) -> dict:

        keys = list(data.keys())

        for key in keys:
            if 'roi' not in key:
                del data[key]

        dist = Distribution(data)
        stats = dist.display_data()

        return stats
