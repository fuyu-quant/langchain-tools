from langchain.agents import tool

import lightgbm as lgbm
import pickle




@tool("lgbm_inference_tool")
def lgbm_inference_tool(query: str) -> str:
    """useful for inference with LightGBM."""

    df = pd.read_csv('/content/Boston.csv', index_col = 0)[406:]
    x = df.drop(['medv'], axis = 1)
    #y = df['medv']

    lgbm_model = pickle.load(open('trained_model.pkl', 'rb'))

    y_pred = lgbm_model.predict(x, num_interation=lgbm_model.best_iteration)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv('/content/inference.csv')


    result = "LightGBMの推論が完了しました" 
    return result