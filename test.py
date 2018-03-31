import random
import torch
from torch.autograd import Variable

a = [1, 2, 3]
random.shuffle(a)
a = Variable(torch.rand((512)))
b = torch.rand((512, 512, 512))
c = torch.rand((512))

torch.mm()
torch.matmul()
d = torch.mm(a, b)
e = torch.mm(d, torch.t(c))
print (a.size(), b.size(), c.size(), d.size(), e.size())
