import numpy as np 
import matplotlib.pyplot as plt
from sklearn.mixture import *
from tqdm import tqdm 

class Fit_GMM(object):
    """
    Returns an object with fitted GMM 
    Inputs : 
    distances : (#t, #d) size numpy array.
    gmm_max_peaks : Maximum number of peaks to fit.
    gmm_replicates : No. of replicates for GMM. 
    
    """
    
    def __init__(self, distances, gmm_max_peaks=5, gmm_replicates=5):
        print("Fitting GMMs with max peaks %i and %i replicates."%(gmm_max_peaks, gmm_replicates))
        self.train_distances = distances
        self.gmms = []
        for i in tqdm(range(distances.shape[1])):
            self.gmms.append(self.fit(distances[:,i], maxpeaks=gmm_max_peaks, replicates=gmm_replicates))
        self.trainll = self.predict(distances)
        print('Done!')
    
    def fit(self, data, maxpeaks, replicates):
        data = np.atleast_2d(data).T
        aic = []
        gmms = []
        for peaks in range(1, maxpeaks+1):
            gmm = GaussianMixture(n_components=peaks, n_init=replicates)
            gmm.fit(data)
            numParameters = peaks-1+peaks+peaks
            aic.append(gmm.aic(data)+2*(numParameters*(numParameters+1))/(data.shape[0]-numParameters-1))
            gmms.append(gmm)
#         print('Replicate AICs :', aic)
        return gmms[np.argmin(aic)]
    
    def predict(self, data):
        """
        Calculate log-likelihoods for data using fitted GMM model. 
        
        data : (txdist) numpy array
        """
        ll = np.zeros(data.shape)
        assert data.shape[1] == len(self.gmms)
        print('Calculating Log-likelihoods...')
        for i in range(data.shape[1]):    
            data_ = data[:,i]
            ll[:,i] = self.gmms[i].score_samples(np.atleast_2d(data_).T)
        return ll
        
    def filter(self, data, ll_threshold = 0.05):
        """
        Filter the data based on fitted GMMs. Returns a bool array indicating good frames.
        
        data : (txdist) numpy array
        ll_threshold : between 0 and 1

        """
        ll = predict(data)
        thresh = np.percentile(np.sum(self.trainll, axis=1), ll_threshold*100)
        return (np.sum(ll, axis=1) > thresh)
                
    def plot_gmm(self):
        for i in range(self.train_distances.shape[1]):
            fig, ax = plt.subplots()
            data = np.sort(self.train_distances[:, i])
            sort_args = np.argsort(self.train_distances[:,i])
            ax.hist(data, bins=50, density=True, alpha=0.7)
            ax.plot(data, np.exp(self.trainll[sort_args,i]), linewidth=2)
            if self.gmms[i].means_.shape[0] > 1:
                for mu, sigma, p in zip(self.gmms[i].means_.squeeze(), np.sqrt(self.gmms[i].covariances_.squeeze()), 
                                        self.gmms[i].weights_):
                    ax.plot(data, self._calculate_gaussian(data, mu, sigma, p),'--', linewidth=2)
            else:
                mu, sigma, p  = self.gmms[i].means_[0,0], np.sqrt(self.gmms[i].covariances_[0,0,0]), self.gmms[i].weights_[0]
                ax.plot(data, self._calculate_gaussian(data, mu, sigma, p),'--', linewidth=2)
            ax.set_title('Joint Distance #' + str(i))
            ax.set_xlabel('Length (px)')

    def _calculate_gaussian(self, x, mu, sigma, p):    
        return p*np.exp(-0.5*np.square(x-mu)/sigma**2)/np.sqrt(2*np.pi*sigma*sigma)