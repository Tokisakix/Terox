from ..autodiff.variable import Variable

parameter_count:int = 0

class Parameter():

    _id: int
    _value: Variable

    def __init__(self, _value:Variable) -> None:
        global parameter_count
        self._id = parameter_count
        parameter_count += 1
        self._value = _value
        return
    
    def __str__(self) -> str:
        info = f"<Parameter{self._value}>"
        return info
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def value(self) -> Variable:
        return self._value