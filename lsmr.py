from copy import deepcopy
import torch


def LSMR(A, A_transpose, b, dimension, device, myRtol=1e-4, maxit=100,
         damp=0.0, atol=0.0, btol=0.0, conlim=0.0):
    """Compute an approximate least-squares solution to A*x=b using LSMR.
    
    Arguments:
        A (function): Matrix-vector product function.
        A_transpose (function): Matrix_transpose-vector product function.
        b (torch.Tensor): A vector.
        dimension (int): Dimension of solution vector.
        device (torch.device): The device tensors will be allocated to.
        myRtol (float, optional): The relative residual tolerance.
        maxit (int, optional): The maximum number of iterations.
    """
    b = b.reshape(-1,1)
    
    # Initialize the Golub-Kahan bidiagonalization process.

    Mu = deepcopy(b)
    u = Mu
    beta = torch.norm(u)

    v = torch.zeros(dimension, 1, device=device)
    alpha = 0

    if beta > 0:
        u /= beta
        Nv = A_transpose(u)
        v = Nv
        alpha = torch.norm(v)

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

    h    = deepcopy(v)
    hbar = torch.zeros(dimension, 1, device=device)
    x    = torch.zeros(dimension, 1, device=device)

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
    if torch.is_tensor(normA2) == False:
        normA = torch.sqrt(torch.tensor(float(normA2), device=device))
    else:
        normA = torch.sqrt(normA2)
    condA   = 1
    normx   = 0

    # Items for use in stopping rules.
    normb  = beta
    istop  = 0
    normr  = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        return x, istop, itn, normr, normar, normA, condA, normx

    # Main iteration loop.
    while itn < maxit:
        itn += 1
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

        chat, shat, alphahat = symOrtho(alphabar, damp, device)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold   = rho
        c, s, rho = symOrtho(alphahat, beta, device)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold   = zeta
        thetabar  = sbar*rho
        cbar, sbar, rhobar = symOrtho(cbar*rho, thetanew, device)
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
        ctildeold, stildeold, rhotildeold = symOrtho(rhodold, thetabar, device)
        thetatilde    = stildeold* rhobar
        rhodold       =   ctildeold* rhobar
        betad         = - stildeold*betad + ctildeold*betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold   = (zetaold - thetatildeold*tautildeold)/rhotildeold
        taud          = (zeta - thetatilde*tautildeold)/rhodold
        d             = d + betacheck*betacheck
        if torch.is_tensor(d + (betad - taud)**2 + betadd*betadd) == False:
            normr = torch.sqrt(torch.tensor(
                float(d + (betad - taud)**2 + betadd*betadd), device=device))
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

        if istop > 0:
            break

    return x, test1, itn


def symOrtho(a, b, device):
    """This is used by the function LSMR.
    
    Arguments:
        a (float): A number.
        b (float): A number.
        device (torch.device): The device tensors will be allocated to.
    """
    if torch.is_tensor(a) == False:
        a = torch.tensor(float(a), device=device)
    if torch.is_tensor(b) == False:
        b = torch.tensor(float(b), device=device)
    if b==0:
        return torch.sign(a), 0, torch.abs(a)
    elif a==0:
        return 0, torch.sign(b), torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
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


def main():
    """Run an example of LSMR."""
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 1000
    A = torch.randn(2*n, n, device=device)
    A_T = A.transpose(0,1)
    H = lambda x : torch.mm(A, x)
    H_T = lambda x : torch.mm(A_T, x)
    b = torch.randn(2*n, device=device)
    x, test1, itn = LSMR(H, H_T, b, n, device, maxit=10)
    print(test1, itn)


if __name__ == '__main__':
    main()