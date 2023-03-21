import textwrap
from langchain.llms import OpenAI


@tool("tool_maker")
def tool_maker(query: str) -> str:
    """useful to create some kind of langchain-tool"""

    llm = OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1000)

    code = llm('code of ' + query)

    name = query.replace(' ','_').lower()

    bos = "\n@tool('{name}_tool)\n"\
            "def {name}_tool(query: str) -> str:\n"\
            "    'useful for {query}'"
    
    code = textwrap.indent(textwrap.dedent(code)[:], '    ')

    eos = "\n    result = 'finish {name}\n"\
            "    return result" 

    result = bos.format(name=name, query=query) + code + eos.format(name=name)

    file = open(f'/content/{name}.py', mode='w')
    file.write(result)
    file.close()
    
    return result