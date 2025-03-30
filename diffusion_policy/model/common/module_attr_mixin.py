import torch.nn as nn
# PyTorch混入类, 用于扩展nn.Module的功能, 具体是可以直接获得model的device和dtype
class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
