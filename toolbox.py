from typing import Callable
import inspect


class Toolbox:
    def __init__(self):
        self.tools: list[Callable] = []

    def tool(self, function: Callable):
        self.tools.append(function)
        return function

    def list_tools(self) -> list:
        def process_param_type(parameter_type: str):
            if parameter_type == '_empty':
                return 'any'
            return parameter_type

        tools_json = []
        for tool in self.tools:
            parameters = []
            tool_signature = inspect.signature(tool)
            for name, param in tool_signature.parameters.items():
                param_type = param.annotation.__name__
                parameters.append({name: {
                    "type": process_param_type(param_type)
                }})
            tools_json.append({
                "name": tool.__name__,
                "description": tool.__doc__,
                "parameters": parameters
            })
        return tools_json


if __name__ == "__main__":
    toolbox = Toolbox()


    @toolbox.tool
    def add(a: int, b: int):
        """Add two numbers and returns the result."""
        return a + b


    @toolbox.tool
    def subtract(a: float, b: float):
        """Subtract two numbers and returns the result."""
        return a - b


    @toolbox.tool
    def multiply(a, b):
        """Multiply two numbers and returns the result."""
        return a * b


    print(toolbox.list_tools())
