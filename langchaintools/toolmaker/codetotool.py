import textwrap

from langchain.agents import tool


@tool("codetotool")
def codetotool(query: str) -> str:
    """コードからlangchain-toolsを作るのに役に立つ.入力は[]の中身とする"""

    path = os.getcwd()
    
    exec(query)

    tool_name = data[0]
    description = data[1]
    input = data[2]
    output = data[4]


    code_path = path + '/' +  data[3]
    with open(code_path) as f:
        code = f.read()
        

    
    bos = "from langchain.agents import tool"\
            "  "\
            "\n@tool('{tool_name})\n"\
            "def {tool_name}(query: str) -> str:\n"\
            "    '{description}入力は{input}です'\n"
    
    code = textwrap.indent(textwrap.dedent(code)[:], '    ')

    eos = "\n    result = f'{output}'\n"\
            "    return result" 

    result = bos.format(tool_name=tool_name, description=description, input = input) + code + eos.format(output=output)

    file = open(f'{path}/{tool_name}.py', mode='w')
    file.write(result)
    file.close()
    
    return result