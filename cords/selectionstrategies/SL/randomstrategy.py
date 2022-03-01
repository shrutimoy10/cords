# from torch.utils.data import Subset
from scipy.linalg import hadamard
import numpy as np
import torch


class RandomStrategy(object):
    """
    This is the Random Selection Strategy class where we select a set of random points as a datasubset
    and often acts as baselines to compare other selection strategies.

    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    """

    def __init__(self, trainloader, online=False):
        """
        Constructor method
        """

        self.trainloader = trainloader
        self.N_trn = len(trainloader.sampler.data_source)
        self.online = online
        self.indices = None
        self.gammas = None

        # print("Random strategy, init, 3")


    def select(self, budget):
        """
        Perform random sampling of indices of size budget.

        Parameters
        ----------
        budget: int
            The number of data points to be selected

        Returns
        ----------
        indices: ndarray
            Array of indices of size budget selected randomly
        gammas: Tensor
            Gradient weight values of selected indices
        """

        # dataiter = iter(self.trainloader)
        # data, labels = dataiter.next()

        # data = data.reshape(data.shape[0],-1)

        # print("Type : ", data.shape)

        # leverage_scores = compute_leverage_scores(data, budget)

        # print("Total points : ", self.N_trn)
        # print("Budget : ", budget)

        # trainset = self.trainloader.sampler.data_source

        

        # print("Trainset dict : ", trainset[10000][0].shape)




        if self.online or (self.indices is None):
            np.random.seed()
            self.indices = np.random.choice(self.N_trn, size=budget, replace=False)
            # random_indices = np.random.choice(self.N_trn, size=2*budget, replace=False)
            # self.indices = self.leverage_score_sampling(budget, random_indices)
            # print("Num indices : ", len(self.indices))

            # print("Data shape : ", data[self.indices,:].shape)

            self.gammas = torch.ones(budget)
            # self.gammas = self.compute_leverage_scores()
        return self.indices, self.gammas

    # Adding a function for computing the leverage scores
    """ 

    Given the data matrix and a budget, this function computes the leverage scores of the samples
    in the given batch. Its important that the batch size is in powers of 2, hence in 
    the config file, we have taken batch_size = 32.

    ALGO : (for creating a randomized Hadamard sampling matrix)
    1. Create a Hadamard matrix H of order of power of 2
        - scipy allows Hadamard matrices of order of power of 2
    2. Create a diagonal matrix D such that the entries are in {-1,+1} randomly
    3. Create a sampling matrix \Omega such that for each t \in [1,budget],
                        \Omega[t,*] = e_i \sqrt(n/budget) w.p. 1/n.
        Here, e_i is the ith row of the identity matrix, where i is chosen 
        uniformly at random.
    The randomized Hadamard matrix, \Omega, of size budget \times n, is computed as
                        P = \Omega . H . D

    ALGO : (for computing the Leverage scores)
    1. Pre-multiply A by the randomized Hadamard sampling matrix P to get PA
    2. Compute the QR decomposition of PA, 
                        PA = QR
    3. Then, AR^{-1} is a good approximation of an othogonal basis for A, i.e.,
                        AR^{-1} = U
        Compute the leverage scores from the rows of U.

    Computing leverage scores using this algorithm requires access to the full
    dataset, which is a technical problem for now.

    Hence, for now, I will be computing the leverage scores of the randomly 
    sampled datasets. They are weighted as 1 for now, instead of that, lets see 
    what happens when we send the leverage scores as weights.
    """
    def compute_leverage_scores(self):

        #extracting the rows from the trainloader
        trainset = self.trainloader.sampler.data_source

        #initializing size to be that of the first point
        sampled_points = torch.empty((len(trainset[0][0].reshape(-1)),))

        for idx in self.indices:
            sampled_points = torch.vstack((sampled_points, trainset[idx][0].reshape(-1)))
        
        #purging the empty row from the tensor
        sampled_points = sampled_points[1:,:]

        print("Shape of sampled points : ", sampled_points.shape)




        [U, S, V] = torch.linalg.svd(sampled_points)

        row_sum = torch.sum(torch.square(U[:,:50]),axis=1)

        beta = 1

        scores = beta * row_sum

        leverage_scores = scores / torch.sum(row_sum)

        print("Leverage scores length : ", len(leverage_scores))

        return leverage_scores

