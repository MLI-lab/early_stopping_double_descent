import numpy as np



class linear_model():
    def __init__(self,d,sigma_noise=0,beta=None,sigmas=None,normalized=True,s_range=[1,10]):
        self.d = d
        if beta is None:
            self.beta = np.random.randn(self.d)
            #self.beta = np.ones(self.d)
        else:
            self.beta = beta
        
        self.sigma_noise = sigma_noise
        
        if isinstance(sigmas, int) or isinstance(sigmas, float):
            if normalized:
                self.sigmas = np.array([sigmas] * d) / np.sqrt(self.d)
            else:
                self.sigmas = np.array([sigmas] * d) 
        
        elif sigmas in ['geo', 'geometric']:
            if normalized:
                self.sigmas = np.geomspace(s_range[0], s_range[1], d) / np.sqrt(self.d)
            else:
                self.sigmas = np.geomspace(s_range[0], s_range[1], d)
        
        elif sigmas is None:
            if normalized:
                self.sigmas = (np.array([1 for i in range(int(np.floor(d/2)))] +
                                    [0.01 for i in range(int(np.ceil(d/2)))]) / np.sqrt(self.d))
            else:
                self.sigmas = np.array([1 for i in range(int(np.floor(d/2)))] +
                                    [0.01 for i in range(int(np.ceil(d/2)))])
        else:
            self.sigmas = sigmas
            
    def estimate_risk(self,estimator,avover=500):
        # estimator is an instance of a class with a predict function mapping x to a predicted y
        # function estimates the risk by averaging
        risk = 0
        for i in range(avover):
            x = np.random.randn(self.d) * self.sigmas 
            y = x @ self.beta + self.sigma_noise*np.random.randn(1)[0]
            risk += (y - estimator.predict(x))**2
        return risk/avover
    
    def compute_risk(self,hatbeta):
        # compute risk of a linear estimator based on formula
        return np.linalg.norm( self.beta - hatbeta )**2 + self.sigma_noise**2
    
    def sample(self,n):
        Xs = []
        ys = []
        for i in range(n):
            x = np.random.randn(self.d) * self.sigmas
            y = x @ self.beta + self.sigma_noise*np.random.randn(1)[0]
            Xs += [x]
            ys += [y]
        return np.array(Xs),np.array(ys)