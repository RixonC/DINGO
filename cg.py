import torch


def CG(A, b, device, rtol=1e-4, maxit=100):
    """Compute an approximate solution to A*x=b using CG.
    
    Arguments:
        A (function): Matrix-vector product function.
        b (torch.Tensor): A vector.
        device (torch.device): The device tensors will be allocated to.
        rtol (float, optional): The relative residual tolerance.
        maxit (int, optional): The maximum number of iterations.
    """
    b = b.reshape(-1,1)
    rtol2 = rtol**2
    x = torch.zeros(b.shape, device=device)
    r = b.clone()
    delta = torch.mm(r.transpose(0,1), r)
    bb = torch.mm(b.transpose(0,1), b)
    p = r.clone()
    iteration = 0
    best_rel_residual = torch.finfo().max

    if torch.norm(r) == 0:
        rel_res = 0
        return x, rel_res, iteration

    x_best = torch.zeros(b.shape[0], 1, device=device)

    while delta > rtol2*bb and iteration < maxit:
        Ap = A(p)
        pAp = torch.mm(p.transpose(0,1), Ap)
        if pAp <= 0:
            return None
        alpha = delta/pAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res_k = torch.norm(r)/torch.norm(b)
        if best_rel_residual > rel_res_k:
            x_best = x.clone()
            best_rel_residual = rel_res_k.clone()
        prev_delta = delta.clone()
        delta = torch.mm(r.transpose(0,1), r)
        p = r + (delta/prev_delta)*p
        iteration += 1

    x = x_best
    rel_res = best_rel_residual

    return x, rel_res, iteration


def main():
    """Run an example of CG."""
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 1000
    A = torch.randn(n,n,device=device)
    A = torch.mm(A.transpose(0,1),A)
    H = lambda x : torch.mm(A,x)
    b = torch.randn(n,device=device)
    result = CG(H, b, device, maxit=1000)
    if result is None:
        print("CG failed")
    else:
        x, rel_res, iteration = result
        print(rel_res, iteration)


if __name__ == '__main__':
    main()