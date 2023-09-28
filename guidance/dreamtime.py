import math

class Timestep():
    def __init__(self, num_of_timestep = 1000, num_of_iters = 10000):
        super().__init__()
        self.m1 = 800
        self.m2 = 500
        self.s1 = 300
        self.s2 = 100
        self.T = num_of_timestep # total timestep
        self.N = num_of_iters # nerf iteration
        self.weight_sum_val = None

        self.prior_dict = {}
        self.weight_dict = {}
        self.sigma_dict = {}

    def weight(self, t):
        if t not in self.weight_dict:
            if (t > self.m1) :
                self.weight_dict[t] = pow(math.e, -((t-self.m1) ** 2) / (2 * (self.s1**2))) 
            elif (t < self.m2):
                self.weight_dict[t] = pow(math.e, -((t-self.m2) ** 2) / (2 * (self.s2**2))) 
            else:
                self.weight_dict[t] = 1.0
        return self.weight_dict[t]

    def weight_sum(self):
        if self.weight_sum_val == None:
            self.weight_sum_val = sum(self.weight(_t) for _t in range(1, self.T))
        return self.weight_sum_val

    def prior(self, t):
        if t not in self.prior_dict:
            self.prior_dict[t] = self.weight(t) / self.weight_sum()
        return self.prior_dict[t]

    def timestep(self, i):
        def sigma(t_s):
            if t_s not in self.sigma_dict:
                self.sigma_dict[t_s] = sum(self.prior(t) for t in range(t_s, self.T))
            return self.sigma_dict[t_s]
        return max(min((abs(sigma(t_star) - i/self.N), t_star) for t_star in range(1, self.T))[1], self.T * 0.2)