from typing import Callable


class Toolbox:
    def __init__(self):
        self.tools = []

    def tool(self, function: Callable):
        self.tools.append(function)

    def list_tools(self, json_mode: bool=False):
        if not json_mode:
            for tool in self.tools:
                print(f'Tool name: {tool.__name__}\nDescription: {tool.__doc__}')
        else:
            json_tools = []
            for tool in self.tools:
                json_tools.append({
                    "name": tool.__name__,
                    "description": tool.__doc__
                })
            return json_tools

if __name__ == "__main__":
    toolbox = Toolbox()

    @toolbox.tool
    def add(a, b):
        """Add two numbers and returns the result."""
        return a + b

    @toolbox.tool
    def subtract(a, b):
        """Subtract two numbers and returns the result."""
        return a - b

    print(toolbox.list_tools(json_mode=True))
