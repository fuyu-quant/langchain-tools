# langchain-tools

## Contents
* [langchain-tools](#langchain-tools-1)
* [tools](#tools)
    * [Example](#examples)
* [toolmaker](#toolmaker)
    * [codetotool](#codetotool)
    * [texttotool](#texttotool)
    * [toolappender](#toolappender)
* [Reference](#reference)

## langchain-tools

You can install it with the following command
```
pip install git+https://github.com/fuyu-quant/langchain-tools.git
```

## tools
LLM will recognize and use the tool by configuring the tool as follows
```
tools = [
    Tool(name = 'lgbm_train_tool',func = mltools.lgbm_train_tool, description=""),
    Tool(name = 'lgbm_inference_tool',func = mltools.lgbm_inference_tool, description="")
    ]
```
For more information, see the following run example

* Examples  
    * LightGBM  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_LightGBM.ipynb)

## toolmaker
### toolmaker
TOOL to create a new TOOL from text

```
tools = [Tool(name = 'toolmaker', func = toolmaker.toolmaker, description="")]
```
※The accuracy is still not very good, so if you have any suggestions for improvement, I'd like to hear them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_toolmaker.ipynb)




### codetotool
Not yet created.
tool to create a new tool from code

```
tools = [Tool(name = 'toolmaker', func = toolmaker.toolmaker, description="")]
```
※The accuracy is still not very good, so if you have any suggestions for improvement, I'd like to hear them.



### texttotool
Not yet created.


### toolappender
Not yet created.
tool to enable the created tool to be used in LLM.

```
tools = [Tool(name = 'toolappender', func = toolappender, description="")]

agent.run("toolappenderでmultiply_toolをtoolsに追加する")
```


## Reference
* https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_tools.html