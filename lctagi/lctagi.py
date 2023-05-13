from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, tool

from langchain.tools import BaseTool




class LCTAGI():
    def __init__(
        self,
        input_prompt,
        tools,
        ):
        self.input_prompt = input_prompt
        self.tools = tools
        

    def _decide_agent(self):
        print("deciding agent")
        
        decide_llm = OpenAI(temperature=0,model_name="gpt-3.5-turbo")

        i = ''
        for tool_ in self.tools:
            description = t
            i = i + ','
        i = '[' + i + ']'

        prompt = """
        You are an agent that determines if the input task is executable. 
        All you can do is to be included in {exec_list}. 
        Answer "Yes." if you can perform the task, or "No." if you cannot.
        ------
        The entered task is:{input}
        ------
        """.format(exec_list = i,input = input_)

        return decide_llm(prompt)


    def _tool_make(self):
        print("toolmaking agent")

        toolmaking_llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")


        prompt = """
        Create a python function that can execute {input} with a single string as input.
        The code below is the code created with the input "multiply two numbers".
        The output is only the code of the created python.
        ```
        from dataclasses import dataclass

        @dataclass
        class NewTool(BaseTool):
            name = "MultiplicationTool"
            description = "used for multiplication. The input is two numbers. For example, if you want to multiply 1 by 2, the input is '1,2'."

            def _run(self, query: str) -> str:
                "Use the tool."
                a, b = query.split(",")
                c = int(a) * int(b)
                result = c

            return result 

            async def _arun(self, query: str) -> str:
                "Use the tool asynchronously."
                raise NotImplementedError("BingSearchRun does not support async")
        ```
        """.format(input = input_)

        code = toolmaking_llm(prompt)


        file = open('/content/new_tool.py', mode='w')
        file.write(code)
        file.close()

        return code


    def _execute(self, input_, tools_):
        print("executing agent")

        excute_llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

        agent = initialize_agent(tools_, excute_llm, agent="zero-shot-react-description", verbose=True)

        agent.run(input_)

        return


    def run(self):
    
        output = deciding_agent(self)

        if output == "Yes":
            print('Execute it as it is executable without creating a tool.')
            executing_agent(input_, tools_)

        elif output == "No":
            print('It is necessary to create a tool, so run it after creating the tool.')
            new_tool = toolmaking_agent(input_)

            exec(new_tool)

            tools_.append(NewTool())

            executing_agent(input_, tools_)
        
        return 