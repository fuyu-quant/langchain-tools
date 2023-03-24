# langchain-tools

## Contents
* [langchain-tools](#langchain-tools-1)
* [tools](#tools)
    * [Examples](#examples)
* [toolmaker](#toolmaker)
    * [Examples](#examples-1)
* [Reference](#reference)



## langchain-tools
* tools
    * mltools
* toolmaker

以下のコマンドでinstallできます
```
pip install git+https://github.com/fuyu-quant/langchain-tools.git
```

## tools

以下のようにtoolを設定することでLLMがtoolを認識し使うことができるようになります
```
tools = [
    Tool(name = 'lgbm_train_tool',func = mltools.lgbm_train_tool, description=""),
    Tool(name = 'lgbm_inference_tool',func = mltools.lgbm_inference_tool, description="")
    ]
```
詳細については以下の実行例を参照してください

* Examples
    * LightGBMを自然言語だけで学習させるデモのノートブック   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_LightGBM.ipynb)

## toolmaker
自分自身で欲しいtoolを作るためのtool

```
tools = [Tool(name = 'toolmaker', func = toolmaker.toolmaker, description="")]
```

※まだ精度があまり良くないので改善案などがあれば教えていただきたいです

* Examples  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_toolmaker.ipynb)

## Reference
* https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_tools.html