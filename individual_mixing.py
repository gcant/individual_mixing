import numpy as np
import scipy.optimize, scipy.special


lng = scipy.special.gammaln
psi = scipy.special.psi
psip = lambda x: scipy.special.polygamma(1,x)

class DirichletModel:

    def __init__(self, edge_data, group_data, sigma=8.):
        self.sigma = sigma
        self.k = edge_data.copy().astype(float)
        self.K = np.sum(self.k,axis=0)
        self.m = np.sum(self.K)
        self.g = group_data.copy()
        self.n = self.k.shape[0]
        self.c = self.k.shape[1]
        self.p = np.array([ np.mean(self.g==r) for r in range(self.c) ])
        self.y = np.ones((5, self.c, self.c))*np.nan
        self.R = np.ones(self.c)*np.nan
        self.R2 = np.ones(self.c)*np.nan
        self.v = np.ones(self.c)*np.nan
        self.v2 = np.ones(self.c)*np.nan
        self.k_dict = {}
        for r in range(self.c):
            self.k_dict[r] = self.k[self.g==r]


    def log_likelihood_raw(self, a, group):
        alpha = np.exp(a)
        ans = np.sum(lng(alpha+self.k_dict[group])-lng(alpha)) + np.sum( lng(np.sum(alpha)) - lng(np.sum(alpha)+np.sum(self.k_dict[group],axis=1))) 
        ans += -np.sum((a**2)/(2*self.sigma**2))
        return ans
    
    def jacobian_raw(self, a, group):
        alpha = np.exp(a)
        ans = np.exp(a)*(np.sum(psi(alpha+self.k_dict[group])-psi(alpha),axis=0) + np.sum(psi(sum(alpha))-psi(sum(alpha)+np.sum(self.k_dict[group],axis=1))))
        ans += -a/(self.sigma**2)
        return ans
    
    def hessian_raw(self, a, group):
        alpha = np.exp(a)
        ans = np.zeros((self.c,self.c))
        di = (np.sum(psip(alpha+self.k_dict[group])-psip(alpha),axis=0) + np.sum(psip(sum(alpha))-psip(sum(alpha)+np.sum(self.k_dict[group],axis=1))))
        ans[np.diag_indices(self.c)] = di
        for r in range(self.c):
            for s in range(r+1,self.c):
                ans[r,s] = np.sum(psip(sum(alpha))-psip(sum(alpha)+np.sum(self.k_dict[group],axis=1)))
                ans[s,r] = ans[r,s]
        for r in range(self.c):
            for s in range(r,self.c):
                ans[r,s] = ans[r,s]*np.exp(a[r]+a[s])
                ans[s,r] = ans[r,s]
        temp = np.exp(a)*(np.sum(psi(alpha+self.k_dict[group])-psi(alpha),axis=0) + np.sum(psi(sum(alpha))-psi(sum(alpha)+np.sum(self.k_dict[group],axis=1)))) - 1./self.sigma**2
        for r in range(self.c):
            ans[r,r] += temp[r]
        return(ans)
    
    def log_likelihood(self, a, group, R_val=0, v_val=0):
        alpha = np.exp(a)
        return self.log_likelihood_raw(a, group) + R_val*(np.log(alpha[group]) - np.log(np.sum(alpha))) - v_val*np.log(1+np.sum(alpha))
    
    def jacobian(self, a, group, R_val=0, v_val=0):
        alpha = np.exp(a)
        ans = self.jacobian_raw(a, group) + R_val*( -alpha/np.sum(alpha)) - v_val*(alpha/(1+np.sum(alpha)))
        ans[group] += R_val
        return ans
    
    def hessian(self, a, group, R_val=0, v_val=0):
        alpha = np.exp(a)
        ans = self.hessian_raw(a,group)
        for rr in range(len(a)):
            temp = alpha[rr]/np.sum(alpha)
            ans[rr,rr] += R_val*(temp**2 - temp)
            temp = alpha[rr]/(1+np.sum(alpha))
            ans[rr,rr] += v_val*(temp**2 - temp)
            for ss in range(rr+1,len(a)):
                ans[rr,ss] += R_val*alpha[rr]*alpha[ss]/(np.sum(alpha)**2)
                ans[ss,rr] += R_val*alpha[rr]*alpha[ss]/(np.sum(alpha)**2)
                ans[rr,ss] += v_val*alpha[rr]*alpha[ss]/((1+np.sum(alpha))**2)
                ans[ss,rr] += v_val*alpha[rr]*alpha[ss]/((1+np.sum(alpha))**2)
        return ans

    def find_approx_fit_single_group(self, group, R_val, v_val):
        ind = (R_val==0)*v_val + (R_val!=0)*(R_val+2) 
        y_init = np.mean(self.k[self.g==group],axis=0) + 10**-4
        y_init = np.log( 2*self.c*y_init / sum(y_init) )
        F = lambda x: -self.log_likelihood(x, group, R_val=R_val, v_val=v_val)
        opt = scipy.optimize.minimize(F, y_init, method="Powell")
        if opt.success:
            self.y[ind][group] = opt.x
        else:
            print("Error fitting group "+str(group)+".")

    def find_approx_fit_all_groups(self):
        for r in range(self.c):
            self.find_approx_fit_single_group(r, R_val=0, v_val=0)
            self.find_approx_fit_single_group(r, R_val=0, v_val=1)
            self.find_approx_fit_single_group(r, R_val=0, v_val=2)
            self.find_approx_fit_single_group(r, R_val=1, v_val=0)
            self.find_approx_fit_single_group(r, R_val=2, v_val=0)

    def Newton_iterations(self, y_init, group, R_val=0, v_val=0, max_its=50, tol=10**-7):
        F = lambda x: -self.log_likelihood(x, group, R_val=R_val, v_val=v_val)
        J = lambda x: -self.jacobian(x, group, R_val=R_val, v_val=v_val)
        H = lambda x: -self.hessian(x, group, R_val=R_val, v_val=v_val)
        delta,i,y = 10.,0,y_init.copy()
        while delta>tol and i<max_its:
            delta = np.dot(np.linalg.inv(H(y)),J(y))
            y -= delta
            delta,i = np.sum(np.abs(delta)), i+1
        init_likelihood = self.log_likelihood(y_init, group, R_val=R_val, v_val=v_val)
        new_likelihood = self.log_likelihood(y, group, R_val=R_val, v_val=v_val)
        if new_likelihood>=init_likelihood:
            return y
        else:
            print('Group '+str(group)+' not improved by Newton iterations. Already at max?')
            return y_init

    def Newton_iterations_all_groups(self, y_init, R_val=0, v_val=0, max_its=50, tol=10**-7):
        ans = y_init.copy()
        for group in range(self.c):
            x = y_init[group].copy()
            x_new = self.Newton_iterations(x, group, R_val=R_val, v_val=v_val,  max_its=max_its, tol=tol)
            ans[group] = x_new.copy()
        return(ans)

    def Newton_iterations_all_groups_all_conditions(self, y_init, max_its=50, tol=10**-7):
        self.y[0] = self.Newton_iterations_all_groups(y_init[0], R_val=0, v_val=0, max_its=max_its, tol=tol)
        self.y[1] = self.Newton_iterations_all_groups(y_init[1], R_val=0, v_val=1, max_its=max_its, tol=tol)
        self.y[2] = self.Newton_iterations_all_groups(y_init[2], R_val=0, v_val=2, max_its=max_its, tol=tol)
        self.y[3] = self.Newton_iterations_all_groups(y_init[3], R_val=1, v_val=0, max_its=max_its, tol=tol)
        self.y[4] = self.Newton_iterations_all_groups(y_init[4], R_val=2, v_val=0, max_its=max_its, tol=tol)


    def update_R_posteriors(self):
        for r in range(self.c):
            F = self.log_likelihood(self.y[0][r], r)
            H = self.hessian(self.y[0][r], r)
            FR = self.log_likelihood(self.y[3][r], r, R_val=1)
            HR = self.hessian(self.y[3][r], r, R_val=1)
            FR2 = self.log_likelihood(self.y[4][r], r, R_val=2)
            HR2 = self.hessian(self.y[4][r], r, R_val=2)
            temp1 = np.exp(FR-F)*np.sqrt( np.linalg.det(-H) / np.linalg.det(-HR) )
            temp2 = np.exp(FR2-F)*np.sqrt( np.linalg.det(-H) / np.linalg.det(-HR2) )
            self.R[r] = (temp1-self.K[r]/self.m)/(1.-self.K[r]/self.m)
            self.R2[r] = (temp2-temp1**2)/((1-self.K[r]/self.m)**2)
        return( self.R,self.R2 )

    def update_v_posteriors(self):
        for r in range(self.c):
            F = self.log_likelihood(self.y[0][r], r)
            H = self.hessian(self.y[0][r], r)
            Fv = self.log_likelihood(self.y[1][r], r, v_val=1)
            Hv = self.hessian(self.y[1][r], r, v_val=1)
            Fv2 = self.log_likelihood(self.y[2][r], r, v_val=2)
            Hv2 = self.hessian(self.y[2][r], r, v_val=2)
            temp1 = np.exp(Fv-F)*np.sqrt( np.linalg.det(-H) / np.linalg.det(-Hv) )
            temp2 = np.exp(Fv2-F)*np.sqrt( np.linalg.det(-H) / np.linalg.det(-Hv2) )
            self.v[r] = temp1
            self.v2[r] = temp2-temp1**2
        return( self.v,self.v2 )

    def R_posterior(self):
        mu = np.sum(self.p*self.R)
        sigma_sq = np.sum(self.p*self.p*self.R2)
        return( mu, sigma_sq )

    def v_posterior(self):
        mu = np.sum(self.p*self.v)
        sigma_sq = np.sum(self.p*self.p*self.v2)
        return( mu, sigma_sq )

    def fit(self,tol=10**-12):
        print('Finding approx fits...')
        self.find_approx_fit_all_groups()
        print('Improving approx fits with Newton iterations...\n')
        self.Newton_iterations_all_groups_all_conditions(self.y,tol=tol)
        print('MAP alpha estimate:')
        print(np.exp(self.y[0]))
        print('')
        self.update_v_posteriors()
        self.update_R_posteriors()
        mu,sigsq = self.R_posterior()
        print("R posterior: mu ="+"{:10.5f}".format(mu)+",   sigma^2 ="+"{:10.7f}".format(sigsq) + '\n')
        mu,sigsq = self.v_posterior()
        print("v posterior: mu ="+"{:10.5f}".format(mu)+",   sigma^2 ="+"{:10.7f}".format(sigsq) + '\n')


