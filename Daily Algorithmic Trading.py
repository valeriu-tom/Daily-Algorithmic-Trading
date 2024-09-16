import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import yfinance as yf
from my_library import visibility_graphs
from my_library import relative_async_index
from my_library import perm_reversibility_indic
import pandas_ta as ta
import math
import scipy
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from ts2vg import HorizontalVG, NaturalVG
import networkx as nx
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.preprocessing import StandardScaler, RobustScaler
# from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from skorch import NeuralNetClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adamax


btc = yf.download(['btc-usd'], period = '1y', interval='1h')

# Creating the features

rsi_periods = list(range(2, 25))
rsis = pd.DataFrame()
for p in rsi_periods:
    rsis[p] = ta.rsi(btc.Close, p)
# rsi_means = rsis.mean()
# rsis -= rsi_means
rsis = rsis.dropna()
    
pca = PCA(n_components = 2, whiten = False, svd_solver = 'auto')
rsi_pca = pd.DataFrame(pca.fit_transform(rsis), columns = ['PC1', 'PC2'])

rvis = pd.DataFrame()
for p in rsi_periods:
    rvis[p] = ta.rvi(btc.Close, p)

rvis = rvis.dropna()
svd = TruncatedSVD(n_components = 2, n_iter=7, n_oversamples=20, random_state=42)
rvi_pca = pd.DataFrame(svd.fit_transform(rvis), columns = ['RPC1', 'RPC2'])

adx_periods = list(range(7, 30))
adxs = pd.DataFrame()
for p in adx_periods:
    adxs[p] = ta.adx(btc.High, btc.Low, btc.Close, p).iloc[:, 0]

adxs = adxs.dropna()
adx_pca = pd.DataFrame(pca.fit_transform(adxs), columns = ['APC1', 'APC2'])

# find shortest path length in a rolling window:
btc_close = btc['Close'].to_numpy()
pos, neg = visibility_graphs.short_path_len(btc_close, 12)

# Relative Asyncronous Index indicator with 1 week lookback:
rai = relative_async_index.rw_rai(btc_close, 168)
rai_s = pd.Series(rai).ewm(24).mean()

# Time reversability index based on permutation entropy:
ptsr = perm_reversibility_indic.rw_otsr(btc_close, 168)
pe = perm_reversibility_indic.permutation_entropy(btc_close, 3, 28)

# Volume spread indicator:
def vs_indicator(data: pd.DataFrame, window: int = 168):
    price_range = data['High'] - data['Low']
    p_range = price_range.diff()
    volume = data.Volume.diff()
    p_range = pd.DataFrame(p_range)
    p_range = pd.DataFrame(RobustScaler().fit_transform(p_range))
    volume = pd.DataFrame(volume)
    volume = pd.DataFrame(RobustScaler().fit_transform(volume), columns=['vol'])
    vol = sm.add_constant(volume)
    mod = RollingOLS(p_range, vol, window=window).fit()
    df = pd.concat([mod.params, mod.rsquared.rename('Rsq')], axis=1)
    df['pred'] = df.const + df.vol*volume.vol
    df['pred'] = df.apply(lambda row: 0 if row['vol'] <= 0.0 or row['Rsq'] < 0.1 else row['pred'], axis=1)
    indicator = df['pred'] - p_range.iloc[:,0]
    # indicator.index = data.index
    return indicator

vol_spread = vs_indicator(btc, 168)

# use adx_pca length as the length for the rest of indicators
rsi_pca = rsi_pca.tail(len(adx_pca))
rsi_pca.reset_index(drop=True, inplace=True)
rvi_pca = rvi_pca.tail(len(adx_pca))
rvi_pca.reset_index(drop=True, inplace=True)
pos = pd.Series(data=pos, name='Pos').tail(len(adx_pca)).reset_index(drop=True)
neg = pd.Series(data=neg, name='Neg').tail(len(adx_pca)).reset_index(drop=True)
rai_s = pd.Series(rai_s, name='RAI').tail(len(adx_pca)).reset_index(drop=True)
ptsr = pd.Series(data=ptsr, name='PTSR').tail(len(adx_pca)).reset_index(drop=True)
pe = pd.Series(data=pe, name='Perm_Entropy').tail(len(adx_pca)).reset_index(drop=True)
vol_spread = pd.Series(vol_spread, name='Vol_Spread').tail(len(adx_pca)).reset_index(drop=True)

X = pd.concat([rsi_pca, rvi_pca, adx_pca, pos, neg, rai_s, ptsr, pe, vol_spread], axis=1)
X = X.dropna()
y = np.where(btc["Close"].pct_change() > 0, 1, 0)
y = pd.Series(y).tail(len(X))
# sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# Split into train, test
Xtrain = X.iloc[:len(X)-24]
Xtest = X.iloc[-24:]
ytrain = y.iloc[:len(y)-24]
ytest = y.iloc[-24:]

tscv = TimeSeriesSplit(n_splits=5, gap=1, test_size=48)

for train_index, test_index in tscv.split(Xtrain):
    X_train, X_test = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]

    rf = RandomForestClassifier(n_estimators=50, bootstrap=False, max_features=7, max_depth=None,
                                min_samples_leaf=1, min_samples_split=3, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    score = rf.score(X_test, y_test)
    print(f"Model Score: {score}")


confusion_matrix(ytest, rf.predict(Xtest))
print(f'Random Forest Accuracy: {round(accuracy_score(ytest, rf.predict(Xtest)), 4)}')
print(f'RF MatthewsCorr: {round(matthews_corrcoef(ytest, rf.predict(Xtest)), 4)}')


for train_index, test_index in tscv.split(Xtrain):
    X_train, X_test = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
    y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]

    knn = KNeighborsClassifier(n_neighbors=18, leaf_size=6, weights='uniform', p=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    print(f"Model Score: {score}")

confusion_matrix(ytest, knn.predict(Xtest))
print(f'KNN: {round(accuracy_score(ytest, knn.predict(Xtest)), 4)}')
print(f'KNN MatthewsCorr: {round(matthews_corrcoef(ytest, knn.predict(Xtest)), 4)}')


# From time to time (once or twice a month) we can perform hyperparameter tuning by uncompeting the following lines:
# Note: this is a rather extensive grid, resulting in a long estimation time
#param_grid_rf = {
#    "min_samples_leaf": [1, 2, 3],
#    "min_samples_split": [1, 2, 3],
#    "n_estimators": [50, 75, 100, 150, 200, 250, 300],
#    "max_depth": [2, 4, 5, 6, 9, 12, None],
#    "max_features": ['sqrt', 6, 8, None],
#    "class_weight": ['balanced', None], }

#grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring='accuracy', n_jobs=-1).fit(X_train, Y_train)
#grid_search_rf.best_params_

#knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
#param_grid_kn = {
#    "kneighborsclassifier__n_neighbors": [3, 5, 6, 8, 10, 15, 20, 24, 30],
#    "kneighborsclassifier__weights": ['uniform', 'distance'],
#    "kneighborsclassifier__leaf_size": [20, 30, 35, 40, 50],
#    "kneighborsclassifier__p": [1, 2, 3], }

#grid_search_knn = GridSearchCV(knn, param_grid_kn, cv=tscv, scoring='accuracy', n_jobs=-1).fit(X_train, Y_train)
#grid_search_knn.best_params_


# Feed-Forward NN in Pytorch

X_scaled = StandardScaler().fit_transform(X.to_numpy()).astype(np.float32)
Y = np.array(y, dtype=np.float32)
Y = Y.reshape(-1, 1)
# X_torch = torch.tensor(X_scaled, dtype = torch.float32)
# y_torch = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1,1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(12, 15)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # another option is dropout(0.1) and weight_constraint of 3
        self.output = nn.Linear(15, 1)
        self.act_output = nn.Sigmoid()
        self.weight_constraint = 4.0
        init.zeros_(self.hidden1.weight)
        init.zeros_(self.output.weight) # xavier_normal_ is another option instead of zeros_

    def forward(self, x):
        # maxnorm weight before actual forward pass
        with torch.no_grad():
            norm = self.hidden1.weight.norm(2, dim=0, keepdim=True).clamp(min=self.weight_constraint/2)
            desired = torch.clamp(norm, max=self.weight_constraint)
            self.hidden1.weight *= (desired/norm)
        x = self.act1(self.hidden1(x))
        x = self.dropout(x)
        x = self.act_output(self.output(x)).view(-1,1)
        return x

#def index_to_slice(indices, array):
#    return np.array([array[i] for i in indices])

tscv = TimeSeriesSplit(n_splits=4, gap=1, test_size=192)

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    net = NeuralNetClassifier(Net, criterion = nn.BCELoss, optimizer = optim.Adamax, lr = 0.1,
                              max_epochs = 50, batch_size = 100, iterator_train__shuffle = False,
                              train_split = False, verbose = 0)
    net.fit(X_train, y_train)
    predictions = net.predict(X_test)
    score = net.score(X_test, y_test)
    print(f"Model Score: {score}")
    
confusion_matrix(y, net.predict(X_scaled))
print(f'FFNN: {round(accuracy_score(y, net.predict(X_scaled)), 4)}')
print(f'FFNN MatthewsCorr: {round(matthews_corrcoef(y, net.predict(X_scaled)), 4)}')

# To perform Neural Network hyperparameter tunning I recommand the skorch library
# skorch is a wrapper which allows pytorch models to be treated as sklearn models
# there are many hyperparameters available for a neural netwrok, I recomand tunning 1 or 2 at a time, but here is an example with several at a time:

#import random
#from skorch import NeuralNetClassifier
#import torch.nn.init as init
#X_train_scaled = StandardScaler().fit_transform(X_train.to_numpy())
#X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
#y_train_torch = torch.tensor(Y_train, dtype=torch.float32).reshape(-1,1)

#class Net(nn.Module):
#    def __init__(self, n_neurons=10):
#        super().__init__()
#        self.layer = nn.Linear(34, n_neurons)
#        self.act = nn.ReLU()
#        self.output = nn.Linear(n_neurons, 1)
#        self.act_output = nn.Sigmoid()

#    def forward(self, x):
#        x = self.act(self.layer(x))
#        x = self.act_output(self.output(x))
#        return x

#net = NeuralNetClassifier(
#    Net,
#    criterion = nn.BCELoss,
#    optimizer = optim.Adamax,
#    verbose = False )

#param_grid_network = {
#    'batch_size': [20, 50, 100, 200],
#    'max_epochs': [10, 50, 100],
#    'module__n_neurons': [5, 10, 15, 20, 30, 40],
#    'optimizer__lr': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2], }

#grid_search_network = GridSearchCV(net, param_grid_network, cv=tscv, n_jobs=-1)
#grid_result = grid_search_network.fit(X_train_torch, y_train_torch)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))



#####  Transformer Model  #####
# We can add a transfomer model to the ensamble, but only if no more than 2 orders are placed per day, because the model is very slow
#from darts.models import TransformerModel
#import torchmetrics
#from pytorch_lightning.callbacks import Callback
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#class AccuracyLogger(Callback):
#    def __init__(self):
#        self.validation_accuracy = []

#    def on_validation_epoch_end(self, trainer, pl_module):
#        accuracy = trainer.callback_metrics.get('val_accuracy')
#        if accuracy is not None:
#            self.validation_accuracy.append(accuracy.item())
#            print(f"Validation Accuracy: {accuracy:.4f}")

#my_stopper = EarlyStopping(
#    monitor="val_loss",
#    patience=10,
#    min_delta=0.05,
#    mode='min', )

#for train_index, test_index in tscv.split(X_scaled):
#    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
#    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#    nn_transform = TransformerModel( input_chunk_length = 32, output_chunk_length = 1, n_jobs=-1,
#                                     batch_size = 64, n_epochs = 200, model_name = 'BTC_Transformer',
#                                     nr_epochs_val_period = 1, d_model = 24, nhead=4,
#                                     num_encoder_layers = 3, num_decoder_layers = 3, torch_metrics = torchmetrics.Accuracy('binary'),
#                                     dim_feedforward = 128, dropout= 0.1, activation = "GLU", random_state=42,
#                                     likelihood=None, optimizer_kwargs={'lr': 0.01}, save_checkpoints=False, force_reset=False,
#                                     pl_trainer_kwargs = {"callbacks": [AccuracyLogger(), my_stopper]} )
    
#    nn_transform.fit(series = TimeSeries.from_series(y_train), past_covariates = TimeSeries.from_values(X_train),
#                     future_covariates=None, val_series = TimeSeries.from_series(y_test),
#                     val_past_covariates = TimeSeries.from_values(X_test), verbose=False)

  
#LOAD = False     # True = load previously saved model from disk?  False = (re)train the model
#SAVE = "\_TForm_model10e.pth.tar"   # file name to save the model under
#mpath = os.path.abspath(os.getcwd()) + SAVE 
#if LOAD:
#    print("have loaded a previously saved model from disk:" + mpath)
#    model = TransformerModel.load_model(mpath)          
#else:
#    model.fit(  ts_ttrain, 
#                past_covariates=cov_t, 
#                verbose=True)
#    print("have saved the model after training:", mpath)
#    model.save_model(mpath)



# Build the ensamble model:
from sklearn.linear_model import LogisticRegression

rf_predictions = rf.predict_proba(X)[:, 1]
knn_predictions = knn.predict_proba(X)[:, 1]
ffnn_predictions = net.predict_proba(X_scaled)[:, 1]

stacked_features = np.column_stack((rf_predictions, knn_predictions, ffnn_predictions))

meta_model = LogisticRegression()
meta_model.fit(stacked_features, y)

print(f'Ensamble Accuracy: {round(accuracy_score(y, meta_model.predict(stacked_features) ), 4)}')
print(f'Ensamble MatthewsCorr: {round(matthews_corrcoef(y, meta_model.predict(stacked_features)), 4)}')


# Predict the features 1 step ahead using Theta method:
from darts.models import Theta, TBATS, NBEATSModel
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from darts.models.forecasting.sf_auto_theta import StatsForecastAutoTheta
from darts.utils.utils import SeasonalityMode
from sklearn.metrics import mean_absolute_error

darts_X = X.tail(1440)

#for column in X.columns:
#    selected_feature = darts_X[[column]]
#    series = TimeSeries.from_dataframe(selected_feature)
#    model = TBATS(use_box_cox=False, use_trend=False, seasonal_periods=None)
#    model.fit(series)
#    forecast = model.predict(4)
#    future_x = pd.DataFrame(forecast, columns=[f"{column}"])    
#future_x_concat = pd.concat([future_x], keys=X.columns)

def evaluate_model(model, val_series):
    forecast = model.predict(len(val_series))
    mae = mean_absolute_error(val_series.pd_dataframe().values, forecast.pd_dataframe().values)
    return mae

thetas = np.round(np.linspace(0.1, 4, 15), 4)
best_theta_value = []
feature_pred = pd.DataFrame()
for column in darts_X.columns:
    selected_feature = darts_X[[column]]
    series = TimeSeries.from_dataframe(selected_feature)
    train_series, val_series = train_test_split(series, test_size=0.025)
    best_mae = float('inf')
    best_theta= None
    for t in thetas:
        try:
            model = Theta(theta=t, seasonality_period=24, season_mode=SeasonalityMode.ADDITIVE)
            model.fit(train_series)
            mae = evaluate_model(model, val_series)
            if mae < best_mae:
                best_mae = mae
                best_theta = t  
        except Exception as e:
            print(f"Failed to fit model with theta={t}: {e}")
            continue  # Skip this iteration and move to the next theta value
    try:
        model = Theta(theta = best_theta, seasonality_period=24, season_mode=SeasonalityMode.ADDITIVE)
        model.fit(series)
        forc = model.predict(4)
        forecast = forc.pd_series()
        best_theta_value.append(best_theta)
        feature_pred[forecast.name] = forecast
    except Exception as e:
        print(f"Failed to fit final model with theta={best_theta}: {e}")

#feature_pred = pd.DataFrame()
#for column in X.columns:
#    selected_feature = darts_X[[column]]
#    series = TimeSeries.from_dataframe(selected_feature)
#    model = StatsForecastAutoTheta(season_length=24, decomposition_type='additive')
#    model.fit(series)
#    forc = model.predict(4)
#    forecast = forc.pd_series()
#    feature_pred._append(forecast)


### More advanced methods than Theta can used for univariate forecasting, such as N-BEATS, but note that the model is very slow and the small increase in accuracy may not be worth the computation
#nn_feature_pred = pd.DataFrame()
#for column in darts_X.columns:
#    feature = darts_X[[column]]
#    series = TimeSeries.from_dataframe(feature)
#    mod = NBEATSModel(input_chunk_length = 32,
#                      output_chunk_length = 1,
#                      num_stacks = 64,
#                      layer_widths = 15,
#                      batch_size = 64,
#                      n_epochs = 50,
#                      nr_epochs_val_period = 1,
#                      likelihood = None,
#                      optimizer_kwargs = {"lr": 0.01},
#                      log_tensorboard = False,
#                      generic_architecture = True,
#                      random_state = 42,
#                      force_reset = False,
#                      save_checkpoints = False )
#    mod.fit(series, verbose = False)
#    forc = mod.predict(4)
#    forecast = forc.pd_series()
#    nn_feature_pred[forecast.name] = forecast



rf_forecasts = rf.predict_proba(feature_pred)[:, 1]
knn_forecasts = knn.predict_proba(feature_pred)[:, 1]
feature_pred_scaled = StandardScaler().fit_transform(feature_pred.to_numpy()).astype(np.float32)
ffnn_forecasts = net.predict_proba(feature_pred_scaled)[:, 1]

forc_features = np.column_stack((rf_forecasts, knn_forecasts, ffnn_forecasts))
final_pred = meta_model.predict(forc_features)
print(final_pred[0])


# Send buy order to alpaca
import alpaca_trade_api as tradeapi

# Insert your keys bellow:
API_KEY = ' '
SECRET_KEY = ' '
BASE_URL = ' '

api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL, api_version = 'v2')

current_price = api.get_latest_crypto_trades("BTC/USD").get('BTC/USD').p
# amount_to_invest = float(api.get_account().cash)

try:
    qty_to_sell = float(api.get_position("BTCUSD").qty)
except tradeapi.rest.APIError:
    qty_to_sell = 0  # no position found

if qty_to_sell > 0:
    amount_to_invest = 0
else:
    amount_to_invest = 1000

qty_to_buy = amount_to_invest / current_price

orders = api.list_orders(status='filled')
last_bought_order = None
for order in orders:
    if order.symbol == 'BTC/USD' and order.side == 'buy':
        if last_bought_order is None or order.submitted_at > last_bought_order.submitted_at:
            last_bought_order = order
last_price = float(last_bought_order.filled_avg_price)

# Check forecast and place order
if final_pred[0] == 1 and qty_to_buy > 0:
    api.submit_order(
        symbol = 'BTC/USD',
        qty = qty_to_buy,
        side = 'buy',
        type = 'market',
        time_in_force = 'gtc' )
elif final_pred[0] == 0 and qty_to_sell > 0 and current_price >= last_price:
    api.submit_order(
        symbol = 'BTC/USD',
        qty = qty_to_sell,
        side = 'sell',
        type = 'market',
        time_in_force = 'gtc' )



