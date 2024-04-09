from typing import Any, List, Tuple

from .parameter import Parameter

module_count:int = 0

class Module():

    _id: int

    def __init__(self) -> None:
        global module_count
        self._id = module_count
        module_count += 1
        return
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def getParmeterDict(self) -> dict:
        def getParmeterList(module:Module) -> List[Tuple[str, Parameter]]:
            parmeters = []
            for key in module.__dict__:
                value = module.__dict__[key]
                if isinstance(value, Module):
                    subparmeters = []
                    for name, parmeter in getParmeterList(value):
                        subparmeters.append((f"{key}.{name}", parmeter))
                    parmeters += subparmeters 
                if isinstance(value, Parameter):
                    parmeters.append((key, value))
            return parmeters
        parmeters = dict(getParmeterList(self))
        return parmeters
    
    def parmeters(self) -> List[Parameter]:
        parmeters_dict = self.getParmeterDict()
        parmeters = []
        for key in parmeters_dict:
            parmeters.append(parmeters_dict[key])
        return parmeters 
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError