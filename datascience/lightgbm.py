from langchain.agents import tool

import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import pickle


@tool("lgbm_train_tool")
def lgbm_train_tool(query: str) -> str:
    """useful for learning LightGBM"""

    global lgbm

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression'
    }

    df = pd.read_csv('/content/Boston.csv', index_col = 0)
    x = df.drop(['medv'], axis = 1)
    y = df['medv']

    x_train,x_valid,y_train,y_valid = train_test_split(x, y ,test_size = 0.2, random_state=3655)

    categorical_features = []


    lgb_train = lgbm.Dataset(x_train,y_train,categorical_feature=categorical_features,free_raw_data=False)
    lgb_eval = lgbm.Dataset(x_valid,y_valid,reference=lgb_train,categorical_feature=categorical_features,free_raw_data=False)


    lgbm_model = lgbm.train(params,lgb_train,
                 valid_sets=[lgb_train,lgb_eval],
                 verbose_eval=10,
                 num_boost_round=1000,
                 early_stopping_rounds= 20)
    
    file = 'trained_model.pkl'
    pickle.dump(lgbm_model, open(file, 'wb'))
    del lgbm

    result = "LightGBMの学習が完了しました"
    return result




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