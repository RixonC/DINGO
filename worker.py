from cg import CG
from copy import deepcopy
from lsmr import LSMR
from minres_qlp import minresQLP
import numpy as np
import torch
import torch.utils as utils


class Worker():
    """An instance of this class is held on each worker node. It contains the 
    local trainset and local computation instructions, e.g., functions to 
    compute the gradient on local data.
    
    Arguments:
        dataset (torch.utils.data.Dataset): A training dataset.
        batch_size (int): How many samples per batch to load.
        dataloader_processes (int) How many subprocesses to use for data 
            loading. 0 means that the data will be loaded in the main process.
        criterion (function): A function that computes the loss given outputs 
            and labels.
        device (torch.device): The device tensors will be allocated to.
        subproblem_tolerance (float): The tolerance used by the subproblem 
            solvers.
        subproblem_maximum_iterations (int): The maximum number of iterations 
            used by the subproblem solvers.
        regularization_parameter (float, optional): A regularization parameter.
    """
    
    def __init__(self, dataset, batch_size, dataloader_processes, criterion, 
                 device, subproblem_tolerance, subproblem_maximum_iterations, 
                 regularization_parameter=0.0):
        self.criterion = criterion
        self.device = device
        self.dataset = dataset
        self.dataloader = utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True, num_workers=dataloader_processes)
        self.dataloader_processes = dataloader_processes
        self.regularization_parameter = regularization_parameter
        self.s = len(dataset)
        self.subproblem_maximum_iterations = subproblem_maximum_iterations
        self.subproblem_tolerance = subproblem_tolerance


    def get_local_loss(self, model):
        """Returns the local loss (objective value).
    
        Arguments:
            model (torch.nn.Module): A model.
        """
        loss = 0
        with torch.no_grad():
            for data in iter(self.dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss += self.criterion(outputs, labels).item()
        
        if self.regularization_parameter != 0:
            weights = self.get_model_weights(model)
            loss += ((self.regularization_parameter/2.0)
                     *torch.norm(weights).pow(2))
            
        return loss * (1.0/self.s)


    def get_local_gradient(self, model):
        """Returns the local gradient.
    
        Arguments:
            model (torch.nn.Module): A model.
        """
        local_gradient = 0
        for data in iter(self.dataloader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            local_loss = self.criterion(outputs, labels)
            model.zero_grad()
            local_loss.backward()
            local_gradient += torch.cat(
                [T.grad.reshape(-1,1) for T in model.parameters()], dim=0).data
        
        if self.regularization_parameter != 0:
            weights = self.get_model_weights(model)
            local_gradient += self.regularization_parameter * weights
        
        return local_gradient * (1.0/self.s)


    def get_local_hessian_times_vector(self, model, vector):
        """Returns the local Hessian-vector product.
    
        Arguments:
            model (torch.nn.Module): A model.
            vector (torch.Tensor): A column vector.
        """
        hessian_times_vector = 0
        for data in iter(self.dataloader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = model(inputs)
            local_loss = self.criterion(outputs, labels)
            model.zero_grad()
            local_gradients = torch.autograd.grad(local_loss, 
                                                  model.parameters(), 
                                                  create_graph=True)
            local_gradient = torch.cat(
                [g.reshape(-1,1) for g in local_gradients], dim=0)
            model.zero_grad()
            z = torch.mm(local_gradient.transpose(0,1), vector)
            local_hess_vects = torch.autograd.grad(z, model.parameters())
            hessian_times_vector += torch.cat(
                [v.reshape(-1,1) for v in local_hess_vects], dim=0).data
        
        if self.regularization_parameter != 0:
            hessian_times_vector += self.regularization_parameter * vector
        
        return hessian_times_vector * (1.0/self.s)
    
    
    def get_model_weights(self, model):
        """Returns a column tensor of the model parameters.
    
        Arguments:
            model (torch.nn.Module): A model.
        """
        with torch.no_grad():
            weights = torch.cat([w.reshape(-1,1) for w in model.parameters()], 
                                 dim=0).data
        return weights


    def get_accuracy(self, model):
        """Compute the accuracy of the model on the locally stored dataset.
    
        Arguments:
            model (torch.nn.Module): A model.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.dataloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.train()
        return 100*correct/total


    ###########################################################################
    #------------------------ used by DINGO and GIANT ------------------------#


    def get_local_hessian_inverse_times_vector(self, model, vector, 
                                               method="CG"):
        """Returns an approximation of the local Hessian inverse (or 
        pseudoinverse) times the vector, using either CG or minresQLP method.
        
        Arguments:
            model (torch.nn.Module): A model.
            vector (torch.Tensor): A column vector.
            method (str, optional): Use 'CG' or 'minresQLP' method.
        """
        H = lambda v : self.get_local_hessian_times_vector(model, v)
        if method == "CG":
            result = CG(H, vector, self.device, rtol=self.subproblem_tolerance,
                        maxit=self.subproblem_maximum_iterations)
            if result is None: # CG failed
                return None
            else:
                return result[0]
        else:
            return minresQLP(H, vector, self.device,
                             rtol=self.subproblem_tolerance,
                             maxit=self.subproblem_maximum_iterations)[0]
    
    
    ###########################################################################
    #----------------------------- used by DINGO -----------------------------#
    

    def get_local_hessian_tilde_pseudoinverse_times_vector_tilde(self, 
            model, vector, phi):
        """Returns an approximation of the local Hessian_tilde pseudoinverse 
        times the vector, using LSMR method.
        
        Arguments:
            model (torch.nn.Module): A model.
            vector (torch.Tensor): A column vector.
            phi (float): The hyperparameter phi in the DINGO algorithm.
        """
        dimension = vector.shape[0]
        H = lambda v : self.get_local_hessian_times_vector(model, v)
        H_tilde = lambda v : torch.cat([H(v), phi*v], dim=0)
        H_tilde_transpose \
            = lambda v : H(v[0:int(len(v)/2.0)]) + phi*v[int(len(v)/2.0):]
        vector_tilde = torch.cat(
            [vector, torch.zeros(vector.shape, device=self.device)], dim=0)
        v = LSMR(H_tilde, H_tilde_transpose, vector_tilde, 
                 dimension, self.device, myRtol=self.subproblem_tolerance,
                 maxit=self.subproblem_maximum_iterations)[0]
        return v
    
    
    def get_local_hessian_tilde_pseudoinverse_times_gradient_tilde(self,
            model, gradient, phi):
        """Returns an approximation of the local Hessian_tilde pseudoinverse 
        times the gradient, using LSMR method.
        
        Arguments:
            model (torch.nn.Module): A model.
            gradient (torch.Tensor): The current full gradient.
            phi (float): The hyperparameter phi in the DINGO algorithm.
        """
        v = self.get_local_hessian_tilde_pseudoinverse_times_vector_tilde(
            model, gradient, phi)
        self.local_hessian_tilde_pseudoinverse_times_gradient_tilde = v
        return v


    def get_local_case_3_update_direction(self, model, hessian_times_gradient,
                                          gradient_norm_squared, theta, phi):
        """Returns the local update direction of DINGO under Case 3. Assumes 
        self.get_local_hessian_tilde_pseudoinverse_times_gradient_tilde has 
        been called prior.
        
        Arguments:
            model (torch.nn.Module): A model.
            hessian_times_gradient (torch.Tensor): The current Hessian-gradient 
                product.
            gradient_norm_squared (float): The square of the norm of the full 
                gradient.
            theta (float): The hyperparameter theta in the DINGO algorithm.
            phi (float): The hyperparameter phi in the DINGO algorithm.
        """
        z = self.local_hessian_tilde_pseudoinverse_times_gradient_tilde
        if (torch.mm(z.transpose(0,1), hessian_times_gradient)
            < theta*gradient_norm_squared):
            H = lambda v : self.get_local_hessian_times_vector(model, v)
            A = lambda v : H(H(v)) + phi*phi*v
            temp = CG(A, hessian_times_gradient, self.device,
                      rtol=self.subproblem_tolerance,
                      maxit=self.subproblem_maximum_iterations)
            if temp is None: # CG failed
                return None
            inverse_times_hessian_times_gradient = temp[0]
            lagrangian_numerator = -torch.mm(z.transpose(0,1), 
                hessian_times_gradient) + theta*gradient_norm_squared
            lagrangian_denominator = torch.mm(
                hessian_times_gradient.transpose(0,1),
                inverse_times_hessian_times_gradient)
            lagrangian = lagrangian_numerator / lagrangian_denominator
            direction = -z - lagrangian * inverse_times_hessian_times_gradient
        else:
            direction = -z
        return direction
    

    ###########################################################################
    #---------------------- used by InexactDANE and SGD ----------------------#
    
    
    def get_gradient_on_sample(self, model, sample):
        """Returns the gradient over the sample.
    
        Arguments:
            model (torch.nn.Module): A model.
            sample (tuple(torch.Tensor, torch.Tensor)): A tuple of inputs and 
                labels.
        """
        inputs, labels = sample
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = model(inputs)
        loss = self.criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        gradient = torch.cat(
            [T.grad.reshape(-1,1) for T in model.parameters()], dim=0).data
                         
        if self.regularization_parameter != 0:
            weights = self.get_model_weights(model)
            gradient += self.regularization_parameter * weights
            
        return gradient * (1.0/len(labels))
    
    
    ###########################################################################
    #-------------------------- used by InexactDANE --------------------------#
    
    
    def update_model(self, model, update_direction):
        """Add the update_direction to the model parameters.
        
        Arguments:
            model (torch.nn.Module): A model.
            update_direction (torch.Tensor): The update direction.
        """
        with torch.no_grad():
            parameter_numels \
                = [parameter.numel() for parameter in model.parameters()]
            direction_split = torch.split(update_direction, parameter_numels)
            for i, parameter in enumerate(model.parameters(), 0):
                parameter += direction_split[i].reshape(parameter.shape)
    
    
    def get_InexactDANE_subproblem_solution(self, model, gradient, 
                                            step_size=1.0, eta=1.0, mu=0.0, 
                                            AIDE_tau=0.0, AIDE_y=0.0):
        """Returns a solution to the local InexactDANE subproblem using SVRG.
        
        Arguments:
            model (torch.nn.Module): A model.
            gradient (torch.Tensor): The current full gradient.
            step_size (float, optional): The step size for SVRG solver.
            eta (float, optional): The hyperparameter eta in the InexactDANE 
                algorithm.
            mu (float, optional): The hyperparameter mu in the InexactDANE 
                algorithm.
            AIDE_tau (float, optional): The hyperparameter tau in the AIDE 
                algorithm.
            AIDE_y (torch.Tensor, optional): The tensor y in the AIDE 
                algorithm.
        """
        model_weights = self.get_model_weights(model)
        new_model = deepcopy(model)
        new_model_weights = self.get_model_weights(new_model)
        
        if AIDE_tau != 0:
            gradient = gradient + AIDE_tau * (model_weights - AIDE_y)
        
        sample_indices = np.random.choice(self.s, 
                                          self.subproblem_maximum_iterations)
        sample_set = utils.data.Subset(self.dataset, sample_indices)
        sample_loader = utils.data.DataLoader(sample_set, batch_size=1,
                                        num_workers=self.dataloader_processes)
        
        for sample in iter(sample_loader):
            if AIDE_tau == 0:
                sample_gradient_model = self.get_gradient_on_sample(model, 
                                                                    sample)
                sample_gradient_new_model = self.get_gradient_on_sample(
                                                            new_model, sample)
            else:
                sample_gradient_model \
                    = self.get_gradient_on_sample(model, sample) \
                    + AIDE_tau * (model_weights - AIDE_y)
                sample_gradient_new_model \
                    = self.get_gradient_on_sample(new_model, sample) \
                    + AIDE_tau * (new_model_weights - AIDE_y)
            update_direction \
                = sample_gradient_new_model - sample_gradient_model \
                + eta*gradient + mu*(new_model_weights - model_weights)
            update_direction *= -step_size
            self.update_model(new_model, update_direction)
            new_model_weights = self.get_model_weights(new_model)
        
        return new_model_weights
    
    
    ###########################################################################
    #------------------------------ used by SGD ------------------------------#
    
    
    def get_minibatch_grad(self, model):
        """Returns the local gradient over a random mini-batch of size s/5.
    
        Arguments:
            model (torch.nn.Module): A model.
        """
        minibatch_size = int(self.s / 5.0)
        sample_indices = np.random.choice(len(self.dataset), minibatch_size)
        sample_set = utils.data.Subset(self.dataset, sample_indices)
        sample_loader = utils.data.DataLoader(sample_set, 
            batch_size=len(sample_set), num_workers=self.dataloader_processes)
        sample = iter(sample_loader).next()
        sample_gradient = self.get_gradient_on_sample(model, sample)
        return sample_gradient
        
    
    
    
    
    
    
    
    
    
