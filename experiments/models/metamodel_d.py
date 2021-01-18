from models.metamodel import NeuralCDE
import controldiffeq 
import torch

def normalizer(times, x):
    c = controldiffeq.NaturalCubicSpline(times, x)
    print(c.evaluate_1d(times[0]))

def GetNonzeroTensor(t, model):
    ts = model._forward(t) 
    while len(ts.shape) < 2:
        ts = model._forward(t) 
        return ts

class init_model(NeuralCDE):
    def __init__(self):
        f = lambda x : torch.Tensor(x)
        self.tensor  = f(4) 
        #self.l1 = torch.nn.Linear(self.tensor.size(0), 4)
        n = super().__init__(f, self.tensor.size(0), self.tensor.size(0)//2, self.tensor.size(0))
        
   
    def time(self, i):
         t = self.tensor.shape[0] - i
         cat = torch.cat((self.tensor[:t], torch.randn(t) ), dim=0)
         return torch.sum(cat)/cat.shape[0]
         
    def _forward(self, x):
        final_idx = self.tensor
        x = torch.cat((self.tensor, x))
        *coeffs, =  [x  for _ in range(x[:4].size(0))] 
        times = torch.Tensor([self.time(i) for i in range(int(self.tensor.size(0)))])
        normalizer(times, coeffs)
        x = self.flx_forward(times , coeffs,final_idx)
        return x


class test(init_model):
    def __init__(self, shape, size):
       super().__init__()
       self.l1 = torch.nn.Linear(shape,size) 

    def forward(self, x):
        x = self.l1(x)
        return x


def main():
    x = torch.Tensor(4)
    init= init_model()
    xp = init._forward(x)
    GetNonzeroTensor(xp, init)
    test(xp.shape[0], 4)

main()
