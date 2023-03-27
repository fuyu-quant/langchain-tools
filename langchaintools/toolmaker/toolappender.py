from langchain.agents import tool

@tool("toolappender")
def toolappender(query: str) -> str:
    """toolをtoolsに追加するのに役に立ちます.入力は追加したいtoolの名前です"""
    # 入力は
    code1 = f'import {query}'
    exec(code1, globals())
    code2 = f"tools.append(Tool(name = '{query}', func = {query}.{query}, description=''))"
    exec(code2, globals())
    code3 = "agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)"
    exec(code3, globals())

    result = f'{query}のtoolsへの追加が完了しました'

    return result