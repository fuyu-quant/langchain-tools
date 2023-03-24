# langchain-tools

## Contents
* [langchain-tools](#langchain-tools-1)
* [tools](#tools)
    * [Example](#examples)
* [toolmaker](#toolmaker)
    * [Example](#examples-1)
* [Reference](#reference)

## langchain-tools
* tools
    * mltools
* toolmaker

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

### Examples
* This is a notebook for a demonstration of learning LightGBM using only natural language
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_LightGBM.ipynb)

## toolmaker
tool to make the tool you want yourself.

```
tools = [Tool(name = 'toolmaker', func = toolmaker.toolmaker, description="")]
```
※The accuracy is still not very good, so if you have any suggestions for improvement, I'd like to hear them.

### Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_LightGBM.ipynb)

## Reference
* https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_tools.html