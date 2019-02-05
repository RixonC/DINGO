import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
from worker import Worker


class ResultsRecorder(object):
    '''This class prints and plots results of DINGO, and computes test accuracy or error.

    Args:
        config: A dictionary of all necessary parameters.
        test_X: test features.
        test_Y: test labels.
    '''
    def __init__(self, config, test_X=None, test_Y=None):
        self.config = config
        self.test_X = test_X
        self.test_Y = test_Y
        if test_X is not None:
            self.worker = Worker(test_X, test_Y, config)


    def compute_test_accuracy(self, weights):
        '''Return the test accuracy or error.

        Args:
            weights: The current point.
        '''
        if self.test_X is not None:
            if self.config['obj_fun'] == 'softmax':
                n,d = self.test_X.shape
                C = int(float(len(weights))/d)
                W = weights.reshape(C, d).transpose(0,1) # [d,C]
                XW = torch.mm(self.test_X, W) # [n,C]
                large_vals = torch.max(XW, dim=1, keepdim=True)[0] # [n,1]
                large_vals = torch.max(large_vals, torch.Tensor([0])) #M(x), [n,1]
                #XW - M(x)/<Xi,Wc> - M(x), [n x C]
                XW_trick = XW - large_vals.repeat(1, C) # [n,C]
                #sum over b to calc alphax, [n x total_C]
                XW_1_trick = torch.cat((-large_vals, XW_trick), dim=1) # [n,C+1]
                #alphax, [n, ]
                sum_exp_trick = torch.exp(XW_1_trick).sum(dim=1, keepdim=True) # [n,1]
                inv_sum_exp = 1.0/sum_exp_trick # [n,1]
                inv_sum_exp = inv_sum_exp.repeat(1, C) # [n,C]
                probability = torch.mul(inv_sum_exp, torch.exp(XW_trick)) # [n,C]
                probability = torch.cat((probability, 1-probability.sum(dim=1, keepdim=True)), dim=1) # [n,C+1]
                predicted_labels = torch.max(probability, dim=1, keepdim=False)[1]
                actual_labels = torch.max(self.test_Y, dim=1, keepdim=False)[1]
                accuracy = 100*torch.sum(actual_labels == predicted_labels)/n
                return accuracy
            else:
                return self.worker.loss(weights)
        else:
            return None


    def print_row(self, iteration, lists, final_row=False):
        '''Print the row of the results table corresponding to the iteration.

        Args:
            iteration: The current iteration.
            lists: The dictionary of lists containing the results of DINGO.
            final_row: A boolean indicating if this is the last row of the results table.
        '''
        header = '\n     Iter    CCR   Case   AvgLagrLambda    ||p||      <p,Hg>     LS_exp     Alpha     Time      f        ||g||'
        if self.test_X is not None:
            header += '      Test'
        if (iteration) % 20 == 0: # Print the header every 20 iterations
            print(header)
        if final_row:
            f    = lists.get('loss_list')[-1]
            g    = lists.get('grad_norm_list')[-1]
            prt  = '%8g%93.2e%11.2e' % (iteration,f,g)
            if self.test_X is not None:
                test = lists.get('test_accuracy_list')[-1]
                prt += '%11.2e' % (test)
            print(prt)
        else:
            k      = iteration
            c      = lists.get('cumulative_communication_rounds_list')[k]
            case   = lists.get('cases_list')[k]
            al     = lists.get('avg_Lagrangian_lambda_list')[k]
            p      = lists.get('direction_norm_list')[k]
            ip     = lists.get('inner_prod_list')[k]
            expLS  = lists.get('ls_exp_list')[k]
            alpha  = lists.get('alpha_list')[k]
            t      = lists.get('time_list')[k]
            f      = lists.get('loss_list')[k]
            g      = lists.get('grad_norm_list')[k]
            prt = '%8g%8g%7g%14.2e%13.2e%12.2e%7g%13.2e%8.2f%11.2e%11.2e' % (k,c,case,al,p,ip,expLS,alpha,t,f,g)
            if self.test_X is not None:
                test = lists.get('test_accuracy_list')[k]
                prt += '%11.2e' % (test)
            print(prt)


    def print_plots(self, lists):
        '''Print the results plots.

        Args:
            lists: The dictionary of lists containing the results of DINGO.
        '''
        label = 'DINGO'
        colour = 'k'
        style = '-'

        if (self.test_X is not None):
            size=(4, 16)
            rc = 410
        else:
            size=(4, 12)
            rc = 310
            
        cumulative_communication_rounds_list = [0] + lists['cumulative_communication_rounds_list']

        plt.figure(1, figsize=size)
        plt.subplots_adjust(hspace = 0.3)

        plt.subplot(rc+1)
        plt.plot(cumulative_communication_rounds_list,
                 lists['loss_list'],
                 color=colour,
                 linestyle=style,
                 label=label)
        plt.xlabel('Communication Rounds')
        plt.ylabel(r'Objective Function: $f\:(\mathbf{w})$')
        if cumulative_communication_rounds_list[-1] > self.config['max_communication_rounds']:
            plt.xlim(xmax=self.config['max_communication_rounds'])

        plt.subplot(rc+2)
        plt.semilogy(cumulative_communication_rounds_list,
                     lists['grad_norm_list'],
                     color=colour,
                     linestyle=style,
                     label=label)
        plt.xlabel('Communication Rounds')
        plt.ylabel(r'Gradient Norm: $||\nabla f(\mathbf{w})||$')
        if cumulative_communication_rounds_list[-1] > self.config['max_communication_rounds']:
            plt.xlim(xmax=self.config['max_communication_rounds'])

        plt.subplot(rc+3)
        plt.semilogy(lists['alpha_list'],
                     color=colour,
                     linestyle=style,
                     label=label)
        plt.xlabel('Iteration')
        plt.ylabel(r'Line Search: $\alpha$')

        if self.test_X is not None:
            plt.subplot(rc+4)
            if self.config['obj_fun'] == 'softmax':
                plt.plot(cumulative_communication_rounds_list,
                         lists['test_accuracy_list'],
                         color=colour,
                         linestyle=style,
                         label=label)
                plt.xlabel('Communication Rounds')
                plt.ylabel('Test Classification Accuracy (%)')
                if cumulative_communication_rounds_list[-1] > self.config['max_communication_rounds']:
                    plt.xlim(xmax=self.config['max_communication_rounds'])
            else:
                plt.plot(cumulative_communication_rounds_list,
                         lists['test_accuracy_list'],
                         color=colour,
                         linestyle=style,
                         label=label)
                plt.xlabel('Communication Rounds')
                plt.ylabel('Test Error')
                if cumulative_communication_rounds_list[-1] > self.config['max_communication_rounds']:
                    plt.xlim(xmax=self.config['max_communication_rounds'])

        plt.savefig('./Plots/' + str(self.config['obj_fun']) + '_' + self.config['dataset'] + '_' \
            + str(self.config['num_partitions']) + '.pdf', bbox_inches='tight')
