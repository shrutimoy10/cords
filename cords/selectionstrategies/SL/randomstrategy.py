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
        print("Budget : ", budget)

        # trainset = self.trainloader.sampler.data_source


        # if self.indices is not None:
        #     print("Num indices : {}".format(len(self.indices)))
        
        self.indices = None

        if self.online or (self.indices is None):
            np.random.seed()
            # self.indices = np.random.choice(self.N_trn, size=budget, replace=False)

            # self.gammas = torch.ones(budget)
            # self.gammas = self.compute_leverage_scores()
            # self.indices, self.gammas = self.leverage_score_sampling(budget)
            self.indices, self.gammas = self.normal_random_sampling(budget)
        return self.indices, self.gammas

    # Adding a function for computing the leverage scores
    
    def compute_leverage_scores(self):

        #extracting the rows from the trainloader
        trainset = self.trainloader.sampler.data_source

        #initializing size to be that of the first point
        sampled_points = torch.empty((len(trainset[0][0].reshape(-1)),))

        for idx in self.indices:
            sampled_points = torch.vstack((sampled_points, trainset[idx][0].reshape(-1)))
        
        #purging the empty row from the tensor
        sampled_points = sampled_points[1:,:]
        [U, S, V] = torch.linalg.svd(sampled_points)

        """
        No. of singular vectors to keep
        """
        top_k = 100 

        row_sum = torch.sum(torch.square(U[:,:top_k]),axis=1)

        beta = 1

        scores = beta * row_sum

        leverage_scores = scores / torch.sum(row_sum)
        
        leverage_scores = torch.sqrt(torch.reciprocal(leverage_scores))

        print("Leverage scores length : ", len(leverage_scores))

        return leverage_scores

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
    dataset, which is a technical problem for now. -- SOLVED

    Hence, for now, I will be computing the leverage scores of the randomly 
    sampled datasets. They are weighted as 1 for now, instead of that, lets see 
    what happens when we send the leverage scores as weights.
    """
    def leverage_score_sampling(self, budget):
    
        N = 1024 # nearest power of 2 closest to 450
        H = hadamard(N)

        diag_entries = np.ones(N)
        num_neg_ones = np.random.randint(low = 1, high = N)
        # print(num_neg_ones)
        rand_neg_indices = np.random.randint(low = 0, high = N, size = (1,num_neg_ones))
        diag_entries[rand_neg_indices] *= -1

        D = np.diag(diag_entries)
        s = 512

        norm_factor = np.sqrt(N/s)

        P = np.empty((s,N))

        sampling_indices = []

        while len(sampling_indices) < s:
            index = np.random.randint(N)
            if index in sampling_indices:
                continue
            else:
                sampling_indices.append(index)
        
        for t in range(budget):
            sampling_vector = np.zeros(N)
            # print(sampling_indices[t])
            sampling_vector[sampling_indices[t]] = 1
            P[t,:] = norm_factor * sampling_vector

        H = np.asmatrix(H).reshape(N,N)
        D = np.asmatrix(D).reshape(N,N)
        P = np.asmatrix(P).reshape(s,N)


        sampling_matrix = np.dot(np.dot(P,H),D)

        #randomly select 1000 points
        random_indices = np.random.choice(self.N_trn, size=1024, replace=False)

        #extracting the rows from the trainloader
        trainset = self.trainloader.sampler.data_source
        
        data_matrix = np.empty((len(trainset[0][0].reshape(-1)),))

        for idx in random_indices:
            data_matrix = np.vstack((data_matrix, trainset[idx][0].reshape(-1)))
        
        data_matrix = data_matrix[1:,:]

        #extracting 512 random points from data_matrix of size 1024
        sampled_matrix = np.dot(sampling_matrix, data_matrix)
        print("Sampled Matrix shape : {}".format(sampled_matrix.shape))


        """
        #not working...what if we perform svd of sampled_matrix
        [q,r] = np.linalg.qr(sampled_matrix)
        
        #taking only the rank(sampled_matrix) number of columns
        #since s < N in this case, rank(sampled_matrix) = s
        r = r[:, :s] 


        # print("shape of q : {}".format(q.shape))
        # print("shape of r : {}".format(r.shape))

        U = np.dot(data_matrix[:,:s], np.linalg.inv(r))

        row_sum = np.sum(np.square(U), axis = 1)
        """

        beta = 1

        leverage_scores = beta * row_sum / np.sum(row_sum)
        leverage_scores = np.asarray(leverage_scores).reshape(s,)
        leverage_scores = np.sqrt(leverage_scores)
        print(leverage_scores.shape)

        #randomly sampling "budget" indices from 512 indices
        self.indices =np.random.choice(s, size = budget, replace = False, p = leverage_scores)

        self.gammas = np.sqrt(np.reciprocal(leverage_scores))

        return self.indices, self.gammas


    """
    In this function,we uniformly sample 5000 points from the data matrix. This gives us a 
    5000 x 3072 sample matrix. Multiply this matrix with a normal random matrix of size 
    3072 x 200. This will result in a 5000 x 200 matrix, ie, we are randomly picking 200 
    features from each row. select 450 points. 

    """
    def normal_random_sampling(self,budget):

        #randomly select 5000 points
        random_indices = np.random.choice(self.N_trn, size=5000, replace=False)

        #extracting the rows from the trainloader
        trainset = self.trainloader.sampler.data_source
        
        data_matrix = np.empty((len(trainset[0][0].reshape(-1)),))

        for idx in random_indices:
            data_matrix = np.vstack((data_matrix, trainset[idx][0].reshape(-1)))
        
        data_matrix = data_matrix[1:,:]

        normal_random_matrix = np.random.randn(data_matrix.shape[1], 200)


        #matrix of sampled features
        sampled_feature_matrix = np.dot(data_matrix, normal_random_matrix)
        print("Shape of sampled_feature_matrix : ",sampled_feature_matrix.shape)

        [U,S,V] = np.linalg.svd(sampled_feature_matrix, 'econ')


        top_k = 200 

        # row_sum = np.sum(np.square(U[:,:top_k]),axis=1)
        row_sum = np.sum(np.abs(U[:,:top_k]),axis=1)
        beta = 1

        scores = beta * row_sum

        leverage_scores = scores / np.sum(row_sum)
        
        # leverage_scores = np.sqrt(np.reciprocal(leverage_scores))
        # leverage_scores = np.sqrt(scores)

        print("Leverage scores length : ", len(leverage_scores))
        print("Leverage scores sum : ", np.sum(leverage_scores))

        sampled_indices = np.random.choice(random_indices, size = budget, replace = False,\
                                            p = leverage_scores)
        leverage_weights = np.asarray(np.reciprocal(leverage_scores), dtype = np.int)

        return sampled_indices, leverage_weights

