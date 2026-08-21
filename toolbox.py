from typing import Callable
import inspect


class Toolbox:
    def __init__(self):
        self.tools: list[Callable] = []

    def tool(self, function: Callable):
        self.tools.append(function)
        return function

    def list_tools(self, json_mode: bool = False):
        if not json_mode:
            for tool in self.tools:
                print(f'Tool name: {tool.__name__}\nDescription: {tool.__doc__}')
        else:
            json_tools = []
            for tool in self.tools:
                parameters = []
                tool_signature = inspect.signature(tool)
                for name, param in tool_signature.parameters.items():
                    parameters.append({
                        "name": name,
                        "type": param.annotation.__name__
                    })
                json_tools.append({
                    "name": tool.__name__,
                    "description": tool.__doc__,
                    "parameters": parameters
                })
            return json_tools


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


    print(toolbox.list_tools(json_mode=True))
