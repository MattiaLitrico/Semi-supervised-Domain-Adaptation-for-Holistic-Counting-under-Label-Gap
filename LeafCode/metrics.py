import torch
from torch import nn
import numpy as np
from torch import norm

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class VarianceRegularizer(torch.nn.Module):
    def __init__(self, lambdaint=0.02):
        super(VarianceRegularizer,self).__init__()
        self.lambdaint = lambdaint

    def forward(self, x):
        return -(self.lambdaint*torch.std(x))
    
def print_metrics(metrics):
    print("DiC ...: {} ({})".format(metrics['dic'][0],metrics['dic'][1]))
    print("|DiC| .: {} ({})".format(metrics['abs_dic'][0],metrics['abs_dic'][1]))
    print("MSE ...: {}".format(metrics['mse']))
    print("% .....: {}".format(metrics['%']))
    print("R2 ....: {}".format(metrics['r2']))

def compute_metrics(y, x, round=True):
    absDic = AbsDic()
    dic = Dic()
    percAgreement = PercentageAgreement()
    mse = MSE()
    
    if round:
        x = np.round(x)
    diff = y-x

    res = {
        'dic': dic(diff),
        'abs_dic': absDic(diff),
        'mse' : mse(diff),
        '%' : percAgreement(diff),
        'r2': 0
    }

    return res

class AbsDic(torch.nn.Module):
    def __init__(self):
        super(AbsDic,self).__init__()

    def forward(self, diff):
        if torch.is_tensor(diff):
            return (torch.mean(torch.abs(diff)), torch.std(torch.abs(diff)))
        else:
            return (np.mean(np.abs(diff)), np.std(np.abs(diff)))


class Dic(torch.nn.Module):
    def __init__(self):
        super(Dic,self).__init__()

    def forward(self, diff):
        if torch.is_tensor(diff):
            return (torch.mean(diff), torch.std(diff))
        else:
            return (np.mean(diff), np.std(diff))

class PercentageAgreement(torch.nn.Module):
    def __init__(self):
        super(PercentageAgreement,self).__init__()

    def forward(self, diff):
        if torch.is_tensor(diff):
            return (torch.mean([diff==0])*100)
        else:
            return (np.mean([diff==0])*100)
        
class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE,self).__init__()

    def forward(self,dif):
        diff = (dif)**2
        if torch.is_tensor(diff):
            return (torch.mean(diff), torch.std(diff))
        else:
            return (np.mean(diff), np.std(diff))
