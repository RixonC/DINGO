import torch


def minresQLP(A, b, device, rtol=1e-4, maxit=100, M=None, shift=0, 
              maxxnorm=1e7, Acondlim=1e15, TranCond=1e7, show=False, 
              rnormvec=False):
    """Compute an approximation of A_pseudoinverse*b using MinresQLP.
    
    Arguments:
        A (function): Matrix-vector product function.
        b (torch.Tensor): A vector.
        device (torch.device): The device tensors will be allocated to.
        rtol (float, optional): The relative residual tolerance.
        maxit (int, optional): The maximum number of iterations.
    """
    if rnormvec:
        resvec = []
        Aresvec = []

    b = b.reshape(-1,1)
    n = b.size()[0]
    r2 = b
    r3 = r2
    beta1 = torch.norm(r2)

    if M is None:
        noprecon = True
        pass
    else:
        noprecon = False
        r3 = Precond(M, r2)
        beta1 = torch.mm(r3.transpose(0,1), r2)
        if beta1 < 0:
            print('Error: "M" is indefinite!')
        else:
            beta1 = torch.sqrt(beta1)

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
    Axnorm = 0
    Anorm = 0
    Acond = 1
    relres = rnorm / (beta1 + 1e-50)
    x  = torch.zeros(n,1,device=device)
    w  = torch.zeros(n,1,device=device)
    wl = torch.zeros(n,1,device=device)
    if rnormvec:
        resvec = np.append(resvec, beta1)

    msg = [' beta2 = 0.  b and x are eigenvectors                   ',  # -1
           ' beta1 = 0.  The exact solution is  x = 0               ',  # 0
           ' A solution to Ax = b found, given rtol                 ',  # 1
           ' Min-length solution for singular LS problem, given rtol',  # 2
           ' A solution to Ax = b found, given eps                  ',  # 3
           ' Min-length solution for singular LS problem, given eps ',  # 4
           ' x has converged to an eigenvector                      ',  # 5
           ' xnorm has exceeded maxxnorm                            ',  # 6
           ' Acond has exceeded Acondlim                            ',  # 7
           ' The iteration limit was reached                        ',  # 8
           ' Least-squares problem but no converged solution yet    ']  # 9

    if show:
        print(' ')
        print('Enter Minres-QLP: ')
        print('Min-length solution of symmetric(singular)', end=' ')
        print('(A-sI)x = b or min ||(A-sI)x - b||')
        #||Ax - b|| is ||(A-sI)x - b|| if shift != 0 here
        hstr1 = '    n = %8g    ||Ax - b|| = %8.2e     ' % (n, beta1)
        hstr2 = 'shift = %8.2e       rtol = %8g' % (shift, rtol)
        hstr3 = 'maxit = %8g      maxxnorm = %8.2e  ' % (maxit, maxxnorm)
        hstr4 = 'Acondlim = %8.2e   TranCond = %8g' % (Acondlim, TranCond)
        print(hstr1, hstr2)
        print(hstr3, hstr4)

    #b = 0 --> x = 0 skip the main loop
    if beta1 == 0:
        flag = 0

    while flag == flag0 and iters < maxit:
        #lanczos
        iters += 1
        betal = beta
        beta = betan
        v = r3/beta
        r3 = Ax(A, v)
        if shift == 0:
            pass
        else:
            r3 = r3 - shift*v

        if iters > 1:
            r3 = r3 - r1*beta/betal

        # alfa = torch.tensor(np.real(torch.mm(r3.transpose(0,1), v).numpy()))
        alfa = torch.mm(r3.transpose(0,1), v)
        r3 = r3 - r2*alfa/beta
        r1 = r2
        r2 = r3

        if noprecon:
            betan = torch.norm(r3)
            if iters == 1:
                if betan == 0:
                    if alfa == 0:
                        flag = 0
                        break
                    else:
                        flag = -1
                        x = b/alfa
                        break
        else:
            r3 = Precond(M, r2)
            betan = torch.mm(r2.transpose(0,1), r3)
            if betan > 0:
                betan = torch.sqrt(betan)
            else:
                print('Error: "M" is indefinite or singular!')
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
        cs, sn, gama = SymGivens(gbar, betan, device)
        gama_tmp = gama
        taul2 = taul
        taul = tau
        tau = cs*phi
        Axnorm = torch.sqrt(Axnorm ** 2 + tau ** 2)
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
            cr1, sr1, gamal = SymGivens(gamal, dlta, device)
            vepln = sr1*gama
            gama = -cr1*gama

        #update xnorm
        xnorml = xnorm
        ul4 = ul3
        ul3 = ul2
        if iters > 2:
            ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
        if iters > 1:
            ul = (taul - etal*ul3 - veplnl *ul2)/gamal
        if torch.is_tensor(xl2norm**2 + ul2**2 + ul**2):
            xnorm_tmp = torch.sqrt(xl2norm**2 + ul2**2 + ul**2)
        else:
            xnorm_tmp = torch.sqrt(torch.tensor(
                float(xl2norm**2 + ul2**2 + ul**2), device=device))
        if torch.abs(gama) > torch.finfo().tiny and xnorm_tmp < maxxnorm:
            u = (tau - eta*ul2 - vepln*ul)/gama
            if torch.sqrt(xnorm_tmp**2 + u**2) > maxxnorm:
                u = 0
                flag = 6
        else:
            u = 0
            flag = 9
        if torch.is_tensor(xl2norm**2 + ul2**2):
            xl2norm = torch.sqrt(xl2norm**2 + ul2**2)
        else:
            xl2norm = torch.sqrt(
                torch.tensor(float(xl2norm**2 + ul2**2), device=device))
        xnorm = torch.sqrt(xl2norm**2 + ul**2 + u**2)
        #update w&x
        #Minres
        if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
            if xnorm < maxxnorm:
                x += tau*w
            else:
                flag = 6
        #Minres-QLP
        else:
            QLPiter += 1
            if QLPiter == 1:
                xl2 = torch.zeros(n,1,device=device)
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
        cr2, sr2, gamal = SymGivens(gamal, eplnn, device)
        #transfering from Minres to Minres-QLP
        gamal_QLP = gamal_tmp
        #print('gamal_QLP=', gamal_QLP)
        vepln_QLP = vepln
        gama_QLP = gama
        ul_QLP = ul
        u_QLP = u
        ## Estimate various norms
        abs_gama = torch.abs(gama)
        Anorml = Anorm
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
        relres = rnorm / (Anorm * xnorm + beta1)
        rootl = torch.sqrt(gbar ** 2 + dltan ** 2)
        Arnorml = rnorml * rootl
        relAresl = rootl / Anorm
        ## See if any of the stopping criteria are satisfied.
        epsx = Anorm * xnorm * torch.finfo().eps
        if (flag == flag0) or (flag == 9):
            t1 = 1 + relres
            t2 = 1 + relAresl
            if iters >= maxit:
                flag = 8 #exit before maxit
            if Acond >= Acondlim:
                flag = 7 #Huge Acond
            if xnorm >= maxxnorm:
                flag = 6 #xnorm exceeded
            if epsx >= beta1:
                flag = 5 #x = eigenvector
            if t2 <= 1:
                flag = 4 #Accurate Least Square Solution
            if t1 <= 1:
                flag = 3 #Accurate Ax = b Solution
            if relAresl <= rtol:
                flag = 2 #Trustful Least Square Solution
            if relres <= rtol:
                flag = 1 #Trustful Ax = b Solution
        if flag == 2 or flag == 4 or flag == 6 or flag == 7:
            #possibly singular
            iters = iters - 1
            Acond = Acondl
            rnorm = rnorml
            relres = relresl
        else:
            if rnormvec:
                resvec = torch.stack(resvec, rnorm)
                Aresvec = torch.stack(Aresvec, Arnorml)

            if show:
                if iters%10 - 1 == 0:
                    lstr = ('        iter     rnorm    Arnorm    relres   ' +
                            'relAres    Anorm     Acond     xnorm')
                    print(' ')
                    print(lstr)
                if QLPiter == 1:
                    print('QLP', end='')
                else:
                    print('   ', end='')
                lstr1 = '%8g    %8.2e ' % (iters-1, rnorml)
                lstr2 = '%8.2e  %8.2e ' % (Arnorml, relresl)
                lstr3 = '%8.2e  %8.2e ' % (relAresl, Anorml)
                lstr4 = '%8.2e  %8.2e ' % (Acondl, xnorml)
                print(lstr1, lstr2, lstr3, lstr4)

    #exited the main loop
    if show:
        if QLPiter == 1:
            print('QLP', end = '')
        else:
            print('   ', end = '')
    Miter = iters - QLPiter

    #final quantities
    r1 = b - Ax(A,x) + shift*x
    rnorm = torch.norm(r1)
    Arnorm = torch.norm(Ax(A,r1) - shift*r1)
    xnorm = torch.norm(x)
    relres = rnorm/(Anorm*xnorm + beta1)
    relAres = 0
    if rnorm > torch.finfo().tiny:
        relAres = Arnorm/(Anorm*rnorm)

    if show:
        if rnorm > torch.finfo().tiny:
            lstr1 = '%8g    %8.2e ' % (iters, rnorm)
            lstr2 = '%8.2eD %8.2e ' % (Arnorm, relres)
            lstr3 = '%8.2eD %8.2e ' % (relAres, Anorm)
            lstr4 = '%8.2e  %8.2e ' % (Acond, xnorm)
            print(lstr1, lstr2, lstr3, lstr4)
        else:
            lstr1 = '%8g    %8.2e ' % (iters, rnorm)
            lstr2 = '%8.2eD %8.2e ' % (Arnorm, relres)
            lstr3 = '          %8.2e ' % (Anorm)
            lstr4 = '%8.2e  %8.2e ' % (Acond, xnorm)
            print(lstr1, lstr2, lstr3, lstr4)

        print(' ')
        print('Exit Minres-QLP: ')
        str1 = 'Flag = %8g    %8s' % (flag, msg[int(flag + 1)])
        str2 = 'Iter = %8g      ' % (iters)
        str3 = 'Minres = %8g       Minres-QLP = %8g' % (Miter, QLPiter)
        str4 = 'relres = %8.2e    relAres = %8.2e    ' % (relres, relAres)
        str5 = 'rnorm = %8.2e      Arnorm = %8.2e' % (rnorm, Arnorm)
        str6 = 'Anorm = %8.2e       Acond = %8.2e    ' % (Anorm, Acond)
        str7 = 'xnorm = %8.2e      Axnorm = %8.2e' % (xnorm, Axnorm)
        print(str1)
        print(str2, str3)
        print(str4, str5)
        print(str6, str7)

    if rnormvec:
        Aresvec = torch.stack(Aresvec, Arnorm)
        return (x,flag,iters,Miter,QLPiter,relres,relAres,Anorm,Acond,
                xnorm,Axnorm,resvec,Aresvec)

    return (x,flag,iters,Miter,QLPiter,relres,relAres,Anorm,Acond,xnorm,Axnorm)


def Ax(A, x):
    """Returns the Matrix-vector product Ax.
    
    Arguments:
        A (function or torch.Tensor): Matrix-vector product function or 
            explicit matrix.
        x (torch.Tensor): A vector.
    """
    if callable(A):
        Ax = A(x)
    else:
        Ax = torch.mm(A, x)
    return Ax


def Precond(M, r):
    """This can implement preconditioning into minresQLP."""
    # if callable(M):
    #     h = cg(M, r)
    # else:
    #     h = inv(M).dot(r)
    # return h
    return r


def SymGivens(a, b, device):
    """This is used by the function minresQLP.
    
    Arguments:
        a (float): A number.
        b (float): A number.
        device (torch.device): The device tensors will be allocated to.
    """
    if not torch.is_tensor(a):
        a = torch.tensor(float(a), device=device)
    if not torch.is_tensor(b):
        b = torch.tensor(float(b), device=device)
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = torch.sign(a)
        s = 0
        r = torch.abs(a)
    elif a == 0:
        c = 0
        s = torch.sign(b)
        r = torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
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


def main():
    """Run an example of minresQLP."""
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 1000
    A = torch.randn(n, n, device=device)
    b = torch.randn(n, device=device)
    x,flag,iters,Miter,QLPiter,relres,relAres,Anorm,Acond,xnorm,Axnorm \
        = minresQLP(A, b, device, maxit=100, show=True)


if __name__ == '__main__':
    main()
