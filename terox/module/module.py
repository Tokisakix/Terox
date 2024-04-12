from typing import Any, Dict, List, Tuple

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
    
    def getParmeterDict(self) -> Dict[str, Parameter]:
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
    
    def __str__(self) -> str:
        def _func(module:"Module", idx:int, arg:str) -> List[Tuple[int, str]]:
            module_info = [(idx, f"{arg}{module.__class__.__name__}" + "{\n")]
            for key in module.__dict__:
                value = module.__dict__[key]
                if isinstance(value, Module):
                    sub_info = _func(value, idx + 1, f"{key}:")
                    module_info += sub_info
                if isinstance(value, Parameter):
                    module_info.append((idx + 1, f"{key}:{value}\n"))
            module_info.append((idx, "}\n"))
            return module_info
        module_info = _func(self, 0, "")
        Info = "\n"
        for info in module_info:
            Info += " " * info[0] * 4 + info[1]
        return Info
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError