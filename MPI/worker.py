from __future__ import division

import numpy as np
import torch

class Worker(object):
    '''This class contains the instructions for the local computations on each worker.

    Args:
        X: Feature data.
        Y: Label data.
        config: A dictionary of all necessary parameters.
    '''
    def __init__(self, X, Y, config):
        self.X = X # [s,d]
        self.Y = Y
        self.s, self.d = X.shape
        self.config = config


    def autoencoder(self, w):
        '''Return an autoencoder with weights w.
        Args:
            w: The current point.
        '''
        al = self.config['autoencoder_layers']
        modules = []
        start_val = 0 # this identifies what weights we have already used

        for k in range(len(al)-1):
            module = torch.nn.Linear(al[k], al[k+1])

            my_weight_matrix = w[start_val : start_val + al[k]*al[k+1]].reshape(al[k+1], al[k])
            start_val += al[k]*al[k+1]

            my_bias_vector = w[start_val : start_val + al[k+1]].reshape(al[k+1])
            start_val += al[k+1]

            module.weight = torch.nn.Parameter(my_weight_matrix)
            module.bias = torch.nn.Parameter(my_bias_vector)

            modules.append(module)
            if k != len(al)-2:
                modules.append(torch.nn.ELU())
            else:
                modules.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*modules) # the Autoencoder function


    def loss(self, w):
        '''Return the loss at point w.
        Args:
            w: The current point.
        '''
        if self.config['lamda'] == 0:
            reg = 0
        else:
            reg = self.config['lamda']*0.5*torch.mm(w.transpose(0,1), w) # [1]

        if self.config['obj_fun'] == 'Autoencoder':
            autoencoder = self.autoencoder(w)
            criterion = torch.nn.MSELoss()
            X = autoencoder(self.X) # [s,d]
            loss = reg + (1.0/self.s)*(self.s*self.d)*criterion(X, self.X).data # []

        if self.config['obj_fun'] == 'GMM':
            t = w[0:1] # [1,1]
            alpha = (torch.tanh(t)+1)/2 # [1,1]
            w1 = w[1:self.d+1] # [d,1]
            w2 = w[self.d+1:]  # [d,1]
            C1 = self.config['GMM_C1'] # [d,d]
            C2 = self.config['GMM_C2'] # [d,d]
            c1 = torch.abs(self.config['GMM_C1_determinant']) # []
            c2 = torch.abs(self.config['GMM_C2_determinant']) # []
            W1 = torch.mm(C1, self.X.transpose(0,1)-w1) # [d,s]
            W2 = torch.mm(C2, self.X.transpose(0,1)-w2) # [d,s]
            a = -0.5*(W1*W1).sum(dim=0) # [s]
            b = -0.5*(W2*W2).sum(dim=0) # [s]
            if alpha == 1:
                loss = reg - (1.0/self.s)*torch.sum(torch.log(c1)+a)
            elif alpha == 0:
                loss = reg - (1.0/self.s)*torch.sum(torch.log(c2)+b)
            else:
                m = torch.max(a, b) # [s]
                loss = reg - (1.0/self.s)*torch.sum(m + torch.log(c1*alpha*torch.exp(a-m) + c2*(1-alpha)*torch.exp(b-m)))

        if self.config['obj_fun'] == 'softmax':
            C = int(len(w)/self.d)
            W = w.reshape(C, self.d).transpose(0,1) # [d,C]
            XW = torch.mm(self.X, W) # [s,C]
            large_vals = torch.max(XW, dim=1, keepdim=True)[0] # [s,1]
            large_vals = torch.max(large_vals, torch.Tensor([float(0)])) #M(x), [s,1]
            #XW - M(x)/<Xi,Wc> - M(x), [s x C]
            XW_trick = XW - large_vals.repeat(1, C) # [s,C]
            #sum over b to calc alphax, [s x total_C]
            XW_1_trick = torch.cat((-large_vals, XW_trick), dim=1) # [s,C+1]
            #alphax, [n, ]
            sum_exp_trick = torch.exp(XW_1_trick).sum(dim=1, keepdim=True) # [s,1]
            log_sum_exp_trick = large_vals + torch.log(sum_exp_trick) # [s,1]
            loss = reg + (1.0/self.s)*(torch.sum(log_sum_exp_trick) - torch.sum(XW*self.Y))

        return loss


    def grad(self, w):
        '''Return the gradient vector at point w.
        Args:
            w: The current point.
        '''
        if self.config['lamda'] == 0:
            reg_grad = 0
        else:
            reg_grad = self.config['lamda']*w # [d,1]

        if self.config['obj_fun'] == 'Autoencoder':
            autoencoder = self.autoencoder(w)
            criterion = torch.nn.MSELoss()
            X = autoencoder(self.X) # [s,d]
            loss = (self.s*self.d)*criterion(X, self.X) # []

            for p in autoencoder.parameters(): # zero the gradients
                if p.grad is not None:
                    p.grad.data.zero_()

            loss.backward()
            grad = reg_grad + (1.0/self.s)*torch.cat([T.grad.reshape(-1,1) for T in autoencoder.parameters()]).data

        if self.config['obj_fun'] == 'GMM':
            t = w[0:1] # [1,1]
            alpha = (torch.tanh(t)+1)/2 # [1,1]
            alpha_d = 1.0/(2*torch.cosh(t)**2) # [1,1]
            w1 = w[1:self.d+1] # [d,1]
            w2 = w[self.d+1:]  # [d,1]
            C1 = self.config['GMM_C1'] # [d,d]
            C2 = self.config['GMM_C2'] # [d,d]
            c1 = torch.abs(self.config['GMM_C1_determinant']) # []
            c2 = torch.abs(self.config['GMM_C2_determinant']) # []
            W1 = torch.mm(C1, self.X.transpose(0,1)-w1) # [d,s]
            W2 = torch.mm(C2, self.X.transpose(0,1)-w2) # [d,s]
            a = -0.5*(W1*W1).sum(dim=0) # [s]
            b = -0.5*(W2*W2).sum(dim=0) # [s]
            if alpha == 1:
                g0 = -alpha_d*(1-(c2/c1)*torch.exp(b-a)) # [1,s]
                g0 = g0.sum(dim=1, keepdim=True) # [1,1]
                g1 = -torch.mm(C1.transpose(0,1), W1) # [d,s]
                g1 = g1.sum(dim=1, keepdim=True) # [d,1]
                g2 = torch.zeros(self.d,1) # [d,1]
            elif alpha == 0:
                g0 = -alpha_d*(-1+(c1/c2)*torch.exp(a-b)) # [1,s]
                g0 = g0.sum(dim=1, keepdim=True) # [1,1]
                g1 = torch.zeros(self.d,1) # [d,1]
                g2 = -torch.mm(C2.transpose(0,1), W2) # [d,s]
                g2 = g2.sum(dim=1, keepdim=True) # [d,1]
            else:
                m = torch.max(a, b) # [s]
                ea = torch.exp(a-m) # [s]
                eb = torch.exp(b-m) # [s]
                denominator = c1*alpha*ea + c2*(1-alpha)*eb # [1,s]
                g0 = -(alpha_d*(c1*ea - c2*eb))/denominator  # [1,s]
                g0 = g0.sum(dim=1, keepdim=True) # [1,1]
                g1 = -torch.mm(C1.transpose(0,1), W1) * c1*alpha*ea / denominator # [d,s]
                g1 = g1.sum(dim=1, keepdim=True) # [d,1]
                g2 = -torch.mm(C2.transpose(0,1), W2) * c2*(1-alpha)*eb / denominator # [d,s]
                g2 = g2.sum(dim=1, keepdim=True) # [d,1]
            grad = reg_grad + (1.0/self.s)*torch.cat([g0, g1, g2]) # [2d+1,1]

        if self.config['obj_fun'] == 'softmax':
            C = int(len(w)/self.d)
            W = w.reshape(C, self.d).transpose(0,1) # [d,C]
            XW = torch.mm(self.X, W) # [s,C]
            large_vals = torch.max(XW, dim=1, keepdim=True)[0] # [s,1]
            large_vals = torch.max(large_vals, torch.Tensor([float(0)])) #M(x), [s,1]
            #XW - M(x)/<Xi,Wc> - M(x), [s x C]
            XW_trick = XW - large_vals.repeat(1, C) # [s,C]
            #sum over b to calc alphax, [s x total_C]
            XW_1_trick = torch.cat((-large_vals, XW_trick), dim=1) # [s,C+1]
            #alphax, [s, ]
            sum_exp_trick = torch.exp(XW_1_trick).sum(dim=1, keepdim=True) # [s,1]
            inv_sum_exp = 1.0/sum_exp_trick # [s,1]
            inv_sum_exp = inv_sum_exp.repeat(1, C) # [s,C]
            S = inv_sum_exp*torch.exp(XW_trick) # [s,C]
            g = torch.mm(self.X.transpose(0,1), S-self.Y) # [d,C]
            grad = reg_grad + (1.0/self.s)*g.transpose(0,1).reshape(self.d*C, 1) # [d*C,1]

        return grad


    def hess_vect(self, w, v):
        '''Return the Hessian, at point w, times v.
        Args:
            w: The current point.
            v: A vector.
        '''
        if self.config['lamda'] == 0:
            reg_hess_v = 0
        else:
            reg_hess_v = self.config['lamda']*v # [d,1]

        if self.config['obj_fun'] == 'Autoencoder':
            autoencoder = self.autoencoder(w)
            criterion = torch.nn.MSELoss()
            X = autoencoder(self.X) # [s,d]
            loss = (self.s*self.d)*criterion(X, self.X) # []

            for p in autoencoder.parameters(): # zero the gradients
                if p.grad is not None:
                    p.grad.data.zero_()

            # create_graph=True allows us to call .backward() later
            grads = torch.autograd.grad(loss, autoencoder.parameters(), create_graph=True)
            grad = torch.cat([g.reshape(-1,1) for g in grads])

            for p in autoencoder.parameters(): # zero the gradients
                if p.grad is not None:
                    p.grad.data.zero_()

            z = torch.mm(grad.transpose(0,1), v)
            z.backward()
            return reg_hess_v + (1.0/self.s)*torch.cat([T.grad.reshape(-1,1) for T in autoencoder.parameters()]).data

        if self.config['obj_fun'] == 'GMM':
            v0 = v[0:1] # [1,1]
            v1 = v[1:self.d+1] # [d,1]
            v2 = v[self.d+1:]  # [d,1]
            t = w[0:1] # [1,1]
            alpha = (torch.tanh(t)+1)/2 # [1,1]
            alpha_d = 1.0/(2*torch.cosh(t)**2) # [1,1]
            alpha_2d = -torch.tanh(t)/(torch.cosh(t)**2)
            w1 = w[1:self.d+1] # [d,1]
            w2 = w[self.d+1:]  # [d,1]
            C1 = self.config['GMM_C1'] # [d,d]
            C2 = self.config['GMM_C2'] # [d,d]
            c1 = torch.abs(self.config['GMM_C1_determinant']) # []
            c2 = torch.abs(self.config['GMM_C2_determinant']) # []
            W1 = torch.mm(C1, self.X.transpose(0,1)-w1) # [d,s]
            W2 = torch.mm(C2, self.X.transpose(0,1)-w2) # [d,s]
            a = -0.5*(W1*W1).sum(dim=0) # [s]
            b = -0.5*(W2*W2).sum(dim=0) # [s]
            C1TW1 = torch.mm(C1.transpose(0,1), W1) # [d,s]
            C2TW2 = torch.mm(C2.transpose(0,1), W2) # [d,s]
            if alpha == 1:
                H00 = alpha_d*alpha_d*(1-(c2/c1)*torch.exp(b-a))**2 - alpha_2d*(1-(c2/c1)*torch.exp(b-a)) # [1,s]
                H10 = -alpha_d*C1TW1*(c2/c1)*torch.exp(b-a) # [d,s]
                H20 = alpha_d*C2TW2*(c2/c1)*torch.exp(b-a)  # [d,s]
                H11_v1 = (torch.mm(C1.transpose(0,1), torch.mm(C1, v1)) \
                          - torch.mm(C1TW1, torch.mm(C1TW1.transpose(0,1), v1)) \
                          + torch.mm(C1TW1, torch.mm(C1TW1.transpose(0,1), v1))) # [d,1]
                H22_v2 = torch.zeros(self.d,1)
                H12_v2 = torch.zeros(self.d,1)
                H21_v1 = torch.zeros(self.d,1)
            elif alpha == 0:
                H00 = alpha_d*alpha_d*(-1+(c1/c2)*torch.exp(a-b))**2 - alpha_2d*(-1+(c1/c2)*torch.exp(a-b)) # [1,s]
                H10 = -alpha_d*C1TW1*(c1/c2)*torch.exp(a-b) # [d,s]
                H20 = alpha_d*C2TW2*(c1/c2)*torch.exp(a-b)  # [d,s]
                H11_v1 = torch.zeros(self.d,1)
                H22_v2 = (torch.mm(C2.transpose(0,1), torch.mm(C2, v2)) \
                          - torch.mm(C2TW2, torch.mm(C2TW2.transpose(0,1), v2)) \
                          + torch.mm(C2TW2, torch.mm(C2TW2.transpose(0,1), v2))) # [d,1]
                H12_v2 = torch.zeros(self.d,1)
                H21_v1 = torch.zeros(self.d,1)
            else:
                m = torch.max(a, b) # [s]
                ea = torch.exp(a-m) # [s]
                eb = torch.exp(b-m) # [s]
                denominator = c1*alpha*ea + c2*(1-alpha)*eb # [1,s]
                f1 = c1*ea / denominator # [1,s]
                f2 = c2*eb / denominator # [1,s]
                H00 = alpha_d*alpha_d*(f1-f2)**2 - alpha_2d*(f1-f2) # [1,s]
                H10 = -alpha_d*C1TW1*(f1*f2) # [d,s]
                H20 = alpha_d*C2TW2*(f1*f2)  # [d,s]
                H11_v1 = ((torch.mm(C1.transpose(0,1), torch.mm(C1, v1)) * (alpha*f1)).sum(dim=1, keepdim=True) \
                          - torch.mm(C1TW1, torch.mm(C1TW1.transpose(0,1), v1) * (alpha*f1).transpose(0,1)) \
                          + torch.mm(C1TW1, torch.mm(C1TW1.transpose(0,1), v1) * (alpha*f1).transpose(0,1).pow(2))) # [d,1]
                H22_v2 = ((torch.mm(C2.transpose(0,1), torch.mm(C2, v2)) * ((1-alpha)*f2)).sum(dim=1, keepdim=True) \
                          - torch.mm(C2TW2, torch.mm(C2TW2.transpose(0,1), v2) * ((1-alpha)*f2).transpose(0,1)) \
                          + torch.mm(C2TW2, torch.mm(C2TW2.transpose(0,1), v2) * ((1-alpha)*f2).transpose(0,1).pow(2))) # [d,1]
                H12_v2 = torch.mm(C1TW1, torch.mm(C2TW2.transpose(0,1), v2) * (alpha*(1-alpha)*f1*f2).transpose(0,1)) # [d,1]
                H21_v1 = torch.mm(C2TW2, torch.mm(C1TW1.transpose(0,1), v1) * (alpha*(1-alpha)*f1*f2).transpose(0,1)) # [d,1]

            H00 = H00.sum(dim=1, keepdim=True) # [1,1]
            H00_v0 = H00*v0 # [1,1]
            H10 = H10.sum(dim=1, keepdim=True) # [d,1]
            H10_v0 = H10*v0 # [d,1]
            H20 = H20.sum(dim=1, keepdim=True) # [d,1]
            H20_v0 = H20*v0 # [d,1]
            H01_v1 = torch.mm(H10.transpose(0,1), v1) # [1,1]
            H02_v2 = torch.mm(H20.transpose(0,1), v2) # [1,1]

            return reg_hess_v + (1.0/self.s)*torch.cat([H00_v0 + H01_v1 + H02_v2,
                                                      H10_v0 + H11_v1 + H12_v2,
                                                      H20_v0 + H21_v1 + H22_v2]) # [2d+1,1]

        if self.config['obj_fun'] == 'softmax':
            C = int(len(w)/self.d)
            W = w.reshape(C, self.d).transpose(0,1) # [d,C]
            XW = torch.mm(self.X, W) # [s,C]
            large_vals = torch.max(XW, dim=1, keepdim=True)[0] # [s,1]
            large_vals = torch.max(large_vals, torch.Tensor([float(0)])) #M(x), [s,1]
            #XW - M(x)/<Xi,Wc> - M(x), [s x C]
            XW_trick = XW - large_vals.repeat(1, C) # [s,C]
            #sum over b to calc alphax, [s x total_C]
            XW_1_trick = torch.cat((-large_vals, XW_trick), dim=1) # [s,C+1]
            #alphax, [s, ]
            sum_exp_trick = torch.exp(XW_1_trick).sum(dim=1, keepdim=True) # [s,1]
            inv_sum_exp = 1.0/sum_exp_trick # [s,1]
            inv_sum_exp = inv_sum_exp.repeat(1, C) # [s,C]
            S = inv_sum_exp*torch.exp(XW_trick) # [s,C]

            V = v.reshape(C, self.d).transpose(0,1) # [d,C]
            A = torch.mm(self.X, V) # [s,C]
            AS = torch.sum(A*S, dim=1, keepdim=True) # [s,1]
            rep = AS.repeat(1, C) #A.dot(B)*e*e.T # [s,C]
            XVd1W = A*S - S*rep # [s,C]
            Hv = torch.mm(self.X.transpose(0,1), XVd1W) # [d,C]
            Hv = Hv.transpose(0,1).reshape(self.d*C,1) # [d*C,1]
            return reg_hess_v + (1.0/self.s)*Hv


    ###############################################################################################
    #-------------------------------------------- CG ---------------------------------------------#

    def CG(self, A, b, rtol=None, maxit=None):
        '''Compute an approximate solution to A*x=b using CG.

        Args:
            A: Matrix-vector product function.
            b: A vector.
            rtol: Relative residual tolerance.
            maxit: Maximum iterations.
        '''
        if rtol is None:
            rtol = self.config.get('subproblem_tolerance', 1e-2)
        if maxit is None:
            maxit = self.config.get('subproblem_max_iterations', 100)

        rtol2 = rtol**2
        x = torch.zeros(b.shape) # [d,1]
        r = b - A(x) # [d,1]
        delta = torch.mm(r.transpose(0,1), r) # [1]
        bb = torch.mm(b.transpose(0,1), b) # [1]
        p = r
        iteration = 0
        best_rel_residual = float('inf')

        if torch.norm(r) == 0:
            rel_res = 0
            return x, rel_res, iteration

        x_best = torch.zeros(b.shape[0], 1) # [d,1]

        while delta > rtol2*bb and iteration < maxit and torch.norm(r)/torch.norm(b) > rtol:
            Ap = A(p) # [d,1]
            pAp = torch.mm(p.transpose(0,1), Ap) # [1]
            if pAp <= 0:
                return None
            alpha = delta/pAp # [1]
            x = x + alpha*p # [d,1]
            r = r - alpha*Ap # [d,1]
            rel_res_k = torch.norm(r)/torch.norm(b) # []
            if best_rel_residual > rel_res_k:
                x_best = x # [d,1]
                best_rel_residual = rel_res_k # []
            prev_delta = delta # [1]
            delta = torch.mm(r.transpose(0,1), r) # [1]
            p = r + (delta/prev_delta)*p # [d,1]
            iteration += 1

        x = x_best
        rel_res = best_rel_residual

        return x, rel_res, iteration


    ###############################################################################################
    #---------------------------------------- MINRES-QLP -----------------------------------------#

    def SymGivens(self, a, b):
        '''This is used by the function MinresQLP.'''
        if torch.is_tensor(a) == False:
            a = torch.tensor(float(a))
        if torch.is_tensor(b) == False:
            b = torch.tensor(float(b))

        if b == 0:
            if a == 0:
                c = 1
            else:
                c = torch.sign(a)
            s = 0
            r = abs(a)
        elif a == 0:
            c = 0
            s = torch.sign(b)
            r = abs(b)
        elif abs(b) > abs(a):
            t = a / b
            s = torch.sign(b) / torch.sqrt(1 + t ** 2)
            c = s * t
            r = b / s
        else:
            t = b / a
            c = torch.sign(a) / torch.sqrt(1 + t ** 2)
            s = c * t
            r = a / c
        return c, s, r


    def MinresQLP(self, A, b, rtol=None, maxit=None):
        '''Compute an approximate least-squares solution to A*x=b using MINRES-QLP.

        Args:
            A: Matrix-vector product function.
            b: A vector.
            rtol: Tolerance.
            maxit: Maximum iterations.
        '''
        if rtol is None:
            rtol = self.config.get('subproblem_tolerance', 1e-2)
        if maxit is None:
            maxit = self.config.get('subproblem_max_iterations', 100)

        shift = self.config.get('QLP_shift', 0)
        maxxnorm = self.config.get('QLP_maxxnorm', 1e7)
        Acondlim = self.config.get('QLP_Acondlim', 1e15)
        TranCond = self.config.get('QLP_TranCond', 1e7)

        M = 1 # idenetity matrix. We never use this
        Ab = A(b)
        n = len(b)
        x0 = torch.zeros((n,1))
        x = x0.clone().detach() # this copies the tensor
        xcg = x0.clone().detach()
        b = b.reshape(n,1)
        r2 = b - A(x0)
        r3 = r2
        beta1 = torch.norm(r2)

        ## Initialize
        flag0 = -2
        flag = -2
        iters = 0
        QLPiter = 0
        beta = 0
        tau = 0
        taul = 0
        phi = beta1
        betan = beta1
        gmin = 0
        cs = -1
        sn = 0
        cr1 = -1
        sr1 = 0
        cr2 = -1
        sr2 = 0
        dltan = 0
        eplnn = 0
        gama = 0
        gamal = 0
        gamal2 = 0
        eta = 0
        etal = 0
        etal2 = 0
        vepln = 0
        veplnl = 0
        veplnl2 = 0
        ul3 = 0
        ul2 = 0
        ul = 0
        u = 0
        rnorm = betan
        xnorm = 0
        xl2norm = 0
        Anorm = 0
        Acond = 1
        relres = rnorm / (beta1 + 1e-50)
        w = torch.zeros((n,1))
        wl = torch.zeros((n,1))

        #b = 0 --> x = 0 skip the main loop
        if beta1 == 0:
            flag = 0

        while flag == flag0 and iters < maxit:
            #lanczos
            iters += 1
            betal = beta
            beta = betan
            v = r3/beta
            r3 = A(v)
            if shift == 0:
                pass
            else:
                r3 = r3 - shift*v

            if iters > 1:
                r3 = r3 - r1*beta/betal

            alfa = torch.mm(r3.transpose(0,1), v)
            r3 = r3 - r2*alfa/beta
            r1 = r2
            r2 = r3

            betan = torch.norm(r3)
            if iters == 1:
                if betan == 0:
                    if alfa == 0:
                        flag = 0
                        print('WARNNING: flag = 0')
                        break
                    else:
                        flag = -1
                        print('WARNNING: flag = -1')
                        x = b/alfa
                        break

            pnorm = torch.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)

            #previous left rotation Q_{k-1}
            dbar = dltan
            dlta = cs*dbar + sn*alfa
            epln = eplnn
            gbar = sn*dbar - cs*alfa
            eplnn = sn*betan
            dltan = -cs*betan
            dlta_QLP = dlta
            #current left plane rotation Q_k
            gamal3 = gamal2
            gamal2 = gamal
            gamal = gama
            cs, sn, gama = self.SymGivens(gbar, betan)
            gama_tmp = gama
            taul2 = taul
            taul = tau
            tau = cs*phi
            phi = sn*phi
            #previous right plane rotation P_{k-2,k}
            if iters > 2:
                veplnl2 = veplnl
                etal2 = etal
                etal = eta
                dlta_tmp = sr2*vepln - cr2*dlta
                veplnl = cr2*vepln + sr2*dlta
                dlta = dlta_tmp
                eta = sr2*gama
                gama = -cr2 *gama
            #current right plane rotation P{k-1,k}
            if iters > 1:
                cr1, sr1, gamal = self.SymGivens(gamal, dlta)
                vepln = sr1*gama
                gama = -cr1*gama

            ul4 = ul3
            ul3 = ul2
            if iters > 2:
                ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
            if iters > 1:
                ul = (taul - etal*ul3 - veplnl *ul2)/gamal

            if torch.is_tensor(xl2norm**2 + ul2**2 + ul**2) == False:
                xnorm_tmp = torch.sqrt(torch.tensor(float(xl2norm**2 + ul2**2 + ul**2)))
            else:
                xnorm_tmp = torch.sqrt(xl2norm**2 + ul2**2 + ul**2)

            if abs(gama) > np.finfo(np.double).tiny and xnorm_tmp < maxxnorm:
                u = (tau - eta*ul2 - vepln*ul)/gama
                if torch.sqrt(xnorm_tmp**2 + u**2) > maxxnorm:
                    u = 0
                    flag = 3
                    print('WARNNING: flag = 3')
            else:
                u = 0
                flag = 6

            if torch.is_tensor(xl2norm**2 + ul2**2) == False:
                xl2norm = torch.sqrt(torch.tensor(float(xl2norm**2 + ul2**2)))
            else:
                xl2norm = torch.sqrt(xl2norm**2 + ul2**2)
            xnorm = torch.sqrt(xl2norm**2 + ul**2 + u**2)

            #update w&x
            #Minres
            if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
                wl2 = wl
                #print(wl2.shape)
                wl = w
                w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
                if xnorm < maxxnorm:
                    x += tau*w
                else:
                    flag = 3
                    print('WARNNING: flag = 3')
            #Minres-QLP
            else:
                QLPiter += 1
                if QLPiter == 1:
                    xl2 = torch.zeros((n,1))
                    if (iters > 1):  # construct w_{k-3}, w_{k-2}, w_{k-1}
                        if iters > 3:
                            wl2 = gamal3*wl2 + veplnl2*wl + etal*w
                        if iters > 2:
                            wl = gamal_QLP*wl + vepln_QLP*w
                        w = gama_QLP*w
                        xl2 = x - wl*ul_QLP - w*u_QLP

                if iters == 1:
                    wl2 = wl
                    wl = v*sr1
                    w = -v*cr1
                elif iters == 2:
                    wl2 = wl
                    wl = w*cr1 + v*sr1
                    w = w*sr1 - v*cr1
                else:
                    wl2 = wl
                    wl = w
                    w = wl2*sr2 - v*cr2
                    wl2 = wl2*cr2 +v*sr2
                    v = wl*cr1 + w*sr1
                    w = wl*sr1 - w*cr1
                    wl = v
                xl2 = xl2 + wl2*ul2
                x = xl2 + wl*ul + w*u

            #next right plane rotation P{k-1,k+1}
            gamal_tmp = gamal
            cr2, sr2, gamal = self.SymGivens(gamal, eplnn)
            #transfering from Minres to Minres-QLP
            gamal_QLP = gamal_tmp
            vepln_QLP = vepln
            gama_QLP = gama
            ul_QLP = ul
            u_QLP = u
            ## Estimate various norms
            abs_gama = abs(gama)
            Anorm = max([Anorm, pnorm, gamal, abs_gama])
            if iters == 1:
                gmin = gama
                gminl = gmin
            elif iters > 1:
                gminl2 = gminl
                gminl = gmin
                gmin = min([gminl2, gamal, abs_gama])
            Acondl = Acond
            Acond = Anorm / gmin
            rnorml = rnorm
            relresl = relres
            if flag != 9:
                rnorm = phi
            stopc = rnorm / beta1 # relative residual
            ## See if any of the stopping criteria are satisfied.
            epsx = Anorm * xnorm * torch.tensor(np.finfo(float).eps)
            if (flag == flag0) or (flag == 6):
                if iters >= maxit:
                    flag = 5 #exit before maxit
                if Acond >= Acondlim:
                    flag = 4 #Huge Acond
                    print('WARNNING: Acondlim exceeded!')
                if xnorm >= maxxnorm:
                    flag = 3 #xnorm exceeded
                    print('WARNNING: maxxnorm exceeded!')
                if epsx >= beta1:
                    flag = 2 #x = eigenvector
                if stopc <= rtol:
                    flag = 1 #Trustful Ax = b Solution
            if flag == 3 or flag == 4:
                print('WARNNING: possibly singular!')
                iters = iters - 1
                Acond = Acondl
                rnorm = rnorml
                relres = relresl

        return x,stopc,iters


    ###############################################################################################
    #------------------------------------------- LSMR --------------------------------------------#

    def symOrtho(self,a,b):
        '''This is used by the function LSMR.'''
        if torch.is_tensor(a) == False:
            a = torch.tensor(float(a))
        if torch.is_tensor(b) == False:
            b = torch.tensor(float(b))

        if b==0:
            return torch.sign(a), 0, abs(a)
        elif a==0:
            return 0, torch.sign(b), abs(b)
        elif abs(b)>abs(a):
            tau = a / b
            s = torch.sign(b) / torch.sqrt(1+tau*tau)
            c = s * tau
            r = b / s
        else:
            tau = b / a
            c = torch.sign(a) / torch.sqrt(1+tau*tau)
            s = c * tau
            r = a / c
        return c, s, r


    def LSMR(self, A, A_transpose, b, dimension, myRtol=None, maxit=None):
        '''Compute an approximate least-squares solution to A*x=b using LSMR.

        Args:
            A: Matrix-vector product function.
            A_transpose: Matrix-transpose vector product function.
            b: A vector.
            dimension: Dimension of solution vector.
            myRtol: Relative residual tolerance.
            maxit: Maximum iterations.
        '''
        if myRtol is None:
            myRtol = self.config.get('subproblem_tolerance', 1e-2)
        if maxit is None:
            maxit = self.config.get('subproblem_max_iterations', 100)

        damp = self.config.get('LSMR_damp', 0.0)
        atol = self.config.get('LSMR_atol', 0.0)
        btol = self.config.get('LSMR_btol', 0.0)
        conlim = self.config.get('LSMR_conlim', 0.0)

        self.resids = []             # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.norms = []              # Squared energy norm of iterates.
        self.dir_errors_window = []  # Direct error estimates.
        self.iterates = []

        # Initialize the Golub-Kahan bidiagonalization process.

        Mu = b.clone().detach() # this copies the tensor
        u = Mu
        beta = torch.norm(u)  # norm(u)

        v = torch.zeros(dimension,1)
        alpha = 0

        if beta > 0:
            u /= beta
            Nv = A_transpose(u)
            v = Nv
            alpha = torch.norm(v)  # norm(v)

        if alpha > 0:
            v /= alpha

        # Initialize variables for 1st iteration.

        itn      = 0
        zetabar  = alpha*beta
        alphabar = alpha
        rho      = 1
        rhobar   = 1
        cbar     = 1
        sbar     = 0

        h    = v.clone().detach() # this copies the tensor
        hbar = torch.zeros(dimension,1)
        x    = torch.zeros(dimension,1)

        # Initialize variables for estimation of ||r||.

        betadd      = beta
        betad       = 0
        rhodold     = 1
        tautildeold = 0
        thetatilde  = 0
        zeta        = 0
        d           = 0

        # Initialize variables for estimation of ||A|| and cond(A)

        normA2  = alpha*alpha
        maxrbar = 0
        minrbar = 1e+100
        if torch.is_tensor(normA2) == False:
            normA = torch.sqrt(torch.tensor(float(normA2)))
        else:
            normA = torch.sqrt(normA2)
        condA   = 1
        normx   = 0

        # Items for use in stopping rules.
        normb  = beta
        istop  = 0
        ctol   = 0
        if conlim > 0: ctol = 1.0/conlim
        normr  = beta

        # Reverse the order here from the original matlab code because
        # there was an error on return when arnorm==0
        normar = alpha * beta
        if normar == 0:
            return x, istop, itn, normr, normar, normA, condA, normx

        # Main iteration loop.
        while itn < maxit:
            itn = itn + 1
            Mu = A(v) - alpha * Mu
            u = Mu
            beta = torch.norm(u)  # norm(u)

            if beta > 0:
                u /= beta
                Nv = A_transpose(u) - beta * Nv
                v = Nv

                alpha  = torch.norm(v)  # norm(v)

                if alpha > 0:
                    v /= alpha

            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.

            chat, shat, alphahat = self.symOrtho(alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i

            rhoold   = rho
            c, s, rho = self.symOrtho(alphahat, beta)
            thetanew = s*alpha
            alphabar = c*alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

            rhobarold = rhobar
            zetaold   = zeta
            thetabar  = sbar*rho
            rhotemp   = cbar*rho
            cbar, sbar, rhobar = self.symOrtho(cbar*rho, thetanew)
            zeta      =   cbar*zetabar
            zetabar   = - sbar*zetabar

            # Update h, h_hat, x.

            hbar       = h - (thetabar*rho/(rhoold*rhobarold))*hbar
            x          = x + (zeta/(rho*rhobar))*hbar
            h          = v - (thetanew/rho)*h

            # Estimate of ||r||.

            # Apply rotation Qhat_{k,2k+1}.
            betaacute =   chat* betadd
            betacheck = - shat* betadd

            # Apply rotation Q_{k,k+1}.
            betahat   =   c*betaacute
            betadd    = - s*betaacute

            # Apply rotation Qtilde_{k-1}.
            # betad = betad_{k-1} here.

            thetatildeold = thetatilde
            ctildeold, stildeold, rhotildeold = self.symOrtho(rhodold, thetabar)
            thetatilde    = stildeold* rhobar
            rhodold       =   ctildeold* rhobar
            betad         = - stildeold*betad + ctildeold*betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            tautildeold   = (zetaold - thetatildeold*tautildeold)/rhotildeold
            taud          = (zeta - thetatilde*tautildeold)/rhodold
            d             = d + betacheck*betacheck
            if torch.is_tensor(d + (betad - taud)**2 + betadd*betadd) == False:
                normr = torch.sqrt(torch.tensor(float(d + (betad - taud)**2 + betadd*betadd)))
            else:
                normr = torch.sqrt(d + (betad - taud)**2 + betadd*betadd)

            # Estimate ||A||.
            normA2        = normA2 + beta*beta
            normA         = torch.sqrt(normA2)
            normA2        = normA2 + alpha*alpha

            # Test for convergence.

            # Compute norms for convergence testing.
            normar  = abs(zetabar)
            normx   = torch.norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1   = normr /normb
            if test1 <= myRtol:
                break
            test2   = normar/(normA*normr)
            t1      =  test1/(1 + normA*normx/normb)
            rtol    = btol + atol*normA*normx/normb

            if itn >= maxit:    istop = 7
            if 1 + test2  <= 1: istop = 5
            if 1 + t1     <= 1: istop = 4
            if  test2 <= atol:  istop = 2
            if  test1 <= rtol:  istop = 1

            if istop > 0: break

        return x, test1, itn


    ###############################################################################################
    #--------------------------------------- sub-problems ----------------------------------------#
    
    # in what follows, t indicates the current iteration and i indicates the current worker.

    def hess_inv_vect(self, w, b, method='MINRES-QLP', rtol=None, maxit=None):
        '''Compute an approximate solution to H_{t,i}*x=b using CG or MINRES-QLP.

        Args:
            w: The current point.
            b: A vector.
            method: Which method to use, CG or MINRES-QLP.
            rtol: Tolerance.
            maxit: Maximum iterations.
        '''
        H = lambda v : self.hess_vect(w, v)
        if method == 'CG':
            return self.CG(H, b, rtol, maxit)
        else:
            return self.MinresQLP(H, b, rtol, maxit)


    def hess_tilde_vect(self, w, v):
        '''Return \tilde{H}_{t,i}*v.

        Args:
            w: The current point.
            v: A vector.
        '''
        return torch.cat([self.hess_vect(w, v), self.config['DINGO_phi']*v]) # [2d,1]


    def hess_tilde_transpose_vect(self, w, v):
        '''Return \tilde{H}_{t,i}^{T}*v.

        Args:
            w: The current point.
            v: A vector.
        '''
        return self.hess_vect(w, v[0:len(v)//2]) + self.config['DINGO_phi']*v[len(v)//2:] # [d,1]


    def hess_tilde_inv_vect(self, w, b, rtol=None, maxit=None):
        '''Compute an approximate solution to \tilde{H}_{t,i}*x=b using LSMR.

        Args:
            w: The current point.
            b: A vector.
            rtol: Relative residual tolerance.
            maxit: Maximum iterations.
        '''
        H = lambda v : self.hess_tilde_vect(w, v) # [2d,1]
        H_transpose = lambda v : self.hess_tilde_transpose_vect(w, v) # [d,1]
        return self.LSMR(H, H_transpose, b, b.shape[0]//2, rtol, maxit)


    def hess_tilde_inv_grad_tilde(self, w, grad, rtol=None, maxit=None):
        '''Compute an approximate solution to \tilde{H}_{t,i}*x=\tilde{grad}_{t} using LSMR.

        Args:
            w: The current point.
            grad: The current gradient.
            rtol: Relative residual tolerance.
            maxit: Maximum iterations.
        '''
        grad_tilde = torch.cat([grad, torch.zeros(grad.shape)])
        return self.hess_tilde_inv_vect(w, grad_tilde, rtol, maxit)


    def hess_tilde_trans_hess_tilde_inv_vect(self, w, b, method='CG', rtol=None, maxit=None):
        '''Compute an approximate solution to (\tilde{H}_{t,i}^{T}*\tilde{H}_{t,i})*x=b using CG or MINRES-QLP.

        Args:
            w: The current point.
            b: A vector.
            method: Which method to use: CG or MINRES-QLP.
            rtol: Tolerance.
            maxit: Maximum iterations.
        '''
        A = lambda v : self.hess_tilde_vect(w, v) # [d,1] -> [2d,1]
        B = lambda v : self.hess_tilde_transpose_vect(w, v) # [2d,1] -> [d,1]
        C = lambda v : B(A(v)) # [d,1] -> [d,1]
        if method == 'CG':
            return self.CG(C, b, rtol, maxit) # (x, relres, iters)
        else:
            return self.MinresQLP(C, b, rtol, maxit) # (x, stopc, iters)


    def lagrangian_direction(self, weights, g, Hg, local_hess_tilde_inv_grad_tilde):
        '''Compute an approximate solution to p_{t,i} in Case 3.

        Args:
            weights: The current point.
            g: The current gradient.
            Hg: The current Hessian-gradient product.
            local_hess_tilde_inv_grad_tilde: An approximate solution to \tilde{H}_{t,i}*x=\tilde{grad}_{t}.
        '''
        q = local_hess_tilde_inv_grad_tilde
        qHg = torch.mm(q.transpose(0,1), Hg)
        g_norm_squared = torch.mm(g.transpose(0,1), g)
        if -qHg <= -self.config['DINGO_theta']*g_norm_squared:
            return -q, torch.zeros((1,1)) # lagrangian lambda will be zero
        
        r = self.hess_tilde_trans_hess_tilde_inv_vect(weights, Hg)
        
        if r == None: # CG fails. Use MINRES-QLP until it returns and appropriate solution
            print('MINRES-QLP')
            inner_solver_tol2 = self.config['subproblem_tolerance']
            inner_solver_max_iterations2 = self.config['subproblem_max_iterations']
            while True:
                r, res, iters = self.hess_tilde_trans_hess_tilde_inv_vect(weights, Hg,
                                                                          method='MINRES-QLP',
                                                                          rtol=inner_solver_tol2,
                                                                          maxit=inner_solver_max_iterations2)
                if torch.mm(Hg.transpose(0,1), r) > 0: # Theoretically this should always be true
                    break
                elif iters == inner_solver_max_iterations2:
                    inner_solver_max_iterations2 = inner_solver_max_iterations2*2
                else:
                    inner_solver_tol2 = inner_solver_tol2/2
        else:
            r = r[0]
        
        numerator = -qHg + self.config['DINGO_theta']*g_norm_squared
        denominator = torch.mm(Hg.transpose(0,1), r)
        assert(denominator > 0)
        lagrangian_lambda = numerator/denominator # > 0
        direction = - q - lagrangian_lambda * r
        return direction, lagrangian_lambda
