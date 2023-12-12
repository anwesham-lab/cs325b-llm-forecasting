import json
import numpy as np
import xgboost as xgb

with open('train_untransformed.json', 'r') as f:
    train = json.load(f)

with open('test_untransformed.json', 'r') as f:
    test = json.load(f)

train_x = np.array(train['x'])
train_y = np.array(train['y'])
test_x = np.array(test['x'])
test_y = np.array(test['y'])

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

num_rounds = 100

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mape',
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 5
}
bst = xgb.train(params, dtrain, num_rounds)

pred = bst.predict(dtest)

results = {"gt":test_y.tolist(), "pred": pred.tolist()}

with open('xgboost_untransformed_inference_results.json', 'w') as f:
    test = json.dump(results, f)