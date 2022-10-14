import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
import statsmodels.api as sm
import statsmodels.formula.api as smf
from   statsmodels.tsa.holtwinters import SimpleExpSmoothing
from   statsmodels.tsa.arima_model import ARIMA
from   statsmodels.tsa.holtwinters import ExponentialSmoothing
from plotly import graph_objects as go

#Titulo
st.subheader("""Prevendo Faturamento\n
App que utiliza modelos de series temporais para prever o faturamento""")

#Dataset

data_location = st.file_uploader('Fatu.csv', type='csv')
data = pd.read_csv(data_location, sep = ",", encoding="latin-1", low_memory=False)
df = data.set_index('DATA')
df_treino = pd.DataFrame({'ds':df.index, 'y':df.VALOR})
df_treino["DATAINDEX"] = pd.Series(np.arange(len(df)), index = df.index)
df_treino["DATAINDEX_SQ"] = df_treino["DATAINDEX"] ** 2
df_treino["LOG_VALOR"] = np.log(df_treino['y'])

#Cabeçalho


#Criando a sidebar

st.sidebar.header('Escolha o modelo de Series Temporais')

meses = st.slider('Quantidade de meses de previsão',1,24)

#fazendo a lista de modelos
modelo = ['Tendência Linear','Tendendia Quadratica','Transformação Logarítmica','Simple Smoothing 03','Simple Smoothing 05','Simple Smoothing 08','Triple Exponential Smoothing','ARIMA','SARIMA','Prophet','Ensemble']
modelo_escolhido = st.sidebar.selectbox('Escolha um modelo', modelo)

#criando o dataset de previsão
modelo = Prophet()
modelo.fit(df_treino)
fut = modelo.make_future_dataframe(periods=meses+1, freq='M')
filtro = fut > '2022-08-31T00:00:00'
fut_2 = fut[filtro].dropna()
fut_2["DATAINDEX"] = fut_2.index
fut_2["DATAINDEX_SQ"] = fut_2["DATAINDEX"] ** 2
futuro_ = fut_2.set_index('ds')
df_ensemble = futuro_
if modelo_escolhido == 'Tendência Linear':
    model_linear = smf.ols(formula='y ~ DATAINDEX', data=df_treino).fit()
    previsao = model_linear.predict(futuro_.DATAINDEX)


elif modelo_escolhido == 'Tendendia Quadratica':
    model_quadratic = smf.ols('y ~ DATAINDEX + DATAINDEX_SQ', data=df_treino).fit()
    previsao = model_quadratic.predict(futuro_[["DATAINDEX", "DATAINDEX_SQ"]])

elif modelo_escolhido == 'Transformação Logarítmica':
    model_log = smf.ols('LOG_VALOR ~ DATAINDEX ', data=df_treino).fit()
    previsao_log = model_log.predict(futuro_[["DATAINDEX"]])
    previsao = np.exp(previsao_log)

elif modelo_escolhido == 'Simple Smoothing 03':
    model_exp_smoothing_03 = SimpleExpSmoothing(df_treino.y).fit(smoothing_level=0.3, optimized=False)
    previsao = model_exp_smoothing_03.forecast(len(futuro_))

elif modelo_escolhido == 'Simple Smoothing 05':
    model_exp_smoothing_05 = SimpleExpSmoothing(df_treino.y).fit(smoothing_level=0.5, optimized=False)
    previsao = model_exp_smoothing_05.forecast(len(futuro_))

elif modelo_escolhido == 'Simple Smoothing 08':
    model_exp_smoothing_08 = SimpleExpSmoothing(df_treino.y).fit(smoothing_level=0.8, optimized=False)
    previsao = model_exp_smoothing_08.forecast(len(futuro_))

elif modelo_escolhido == 'Triple Exponential Smoothing':
    model = ExponentialSmoothing(endog = df_treino.y, trend = 'add', seasonal = 'mul', seasonal_periods = 30).fit()
    previsao = model.forecast(steps=len(futuro_))

elif modelo_escolhido == 'ARIMA':
    treino_ARIMA = df_treino['y']
    model_ARIMA_003 = sm.tsa.arima.ARIMA(treino_ARIMA, order=(1, 1, 1))
    results_ARIMA_003 = model_ARIMA_003.fit()
    previsao = results_ARIMA_003.predict(len(treino_ARIMA), end=len(treino_ARIMA)+meses-1, dynamic=True)

elif modelo_escolhido == 'SARIMA':
    treino_SARIMA = df_treino['y']
    model_SARIMA = sm.tsa.statespace.SARIMAX(treino_SARIMA, order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
    results_SARIMA = model_SARIMA.fit()
    previsao = results_SARIMA.predict(len(treino_SARIMA), end=len(treino_SARIMA)+meses-1, dynamic=True)

elif modelo_escolhido == 'Prophet':
    futuro_ds = fut[filtro].dropna()
    modelo = Prophet(seasonality_mode='additive', changepoint_prior_scale=0.01, yearly_seasonality=True)
    modelo.fit(df_treino)
    yhat = modelo.predict(futuro_ds)
    data_prophet = yhat.set_index('ds')
    previsao = data_prophet['yhat']

elif modelo_escolhido == 'Ensemble':
    st.write('O Ensemble está considerando a média dos modelos: Tendência Linear, Transformação Logarítmica e Prophet')

    # linear
    model_linear = smf.ols(formula='y ~ DATAINDEX', data=df_treino).fit()
    previsao_linear = model_linear.predict(futuro_.DATAINDEX)

    #prophet
    futuro_ds = fut[filtro].dropna()
    modelo = Prophet(seasonality_mode='additive', n_changepoints=False, changepoint_prior_scale=0.01, yearly_seasonality=True)
    modelo.fit(df_treino)
    yhat = modelo.predict(futuro_ds)
    data_prophet = yhat.set_index('ds')
    previsao_prop = data_prophet['yhat']

    #logaritmica
    model_log = smf.ols('LOG_VALOR ~ DATAINDEX ', data=df_treino).fit()
    previsao_log = model_log.predict(futuro_[["DATAINDEX"]])
    previsao_log_ = np.exp(previsao_log)
    df_ensemble['DATAINDEX'] = previsao_log_

    #concat
    df_ensemble = df_ensemble.drop(['DATAINDEX_SQ'],axis=1)
    df_ensemble = pd.concat([df_ensemble, previsao_linear], axis=1)
    df_ensemble = pd.concat([df_ensemble, previsao_prop], axis=1)
    previsao = df_ensemble.mean(axis=1)

#Grafico

layout = go.Layout({
    'title': {
        'text': ('Gráfico de Faturamento - {0}'.format(modelo_escolhido)),
        'font': {
            'size': 20
        }
    }
})
fig = go.Figure(layout=layout)

fig.add_trace(go.Scatter(x = df_treino.index, y = df_treino['y'],
                                 mode = 'lines',
                                 name = 'Faturamento'))

fig.add_trace(go.Scatter(x = previsao.index, y = previsao,
                                 mode = 'lines',
                                 name = 'Previsão de Faturamento'))

st.plotly_chart(fig, use_container_width=True)

