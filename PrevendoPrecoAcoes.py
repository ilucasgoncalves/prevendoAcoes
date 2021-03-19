import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#%%
#Importando Dados

empresa = 'FB'

inicio = dt.datetime(2012,1,1)
final = dt.datetime(2020,1,1)

dados = web.DataReader(empresa, 'yahoo', inicio, final)

#Preparar Dados
normalizando = MinMaxScaler(feature_range=(0,1))
dados_normalizados = normalizando.fit_transform(dados['Close'].values.reshape(-1,1))

previsao_dias = 60

x_treinar, y_treinar = [], []

for x in range(previsao_dias, len(dados_normalizados)):
    x_treinar.append(dados_normalizados[x-previsao_dias:x, 0])
    y_treinar.append(dados_normalizados[x, 0 ])
    
x_treinar, y_treinar = np.array(x_treinar), np.array(y_treinar)
x_treinar = np.reshape(x_treinar, (x_treinar.shape[0], x_treinar.shape[1], 1))


#Construindo nosso modelo de rede neural
modelo = Sequential()

modelo.add(LSTM(units=50, return_sequences=True, input_shape=(x_treinar.shape[1], 1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 1)) #Prevendo o proximo valor da acao

modelo.compile(optimizer = 'adam', loss = 'mean_squared_error')
modelo.fit(x_treinar, y_treinar, epochs = 25, batch_size = 32)

###Testando a precisao do nosso modelo em dados existentes

#preparando alguns dados para teste
teste_inicio = dt.datetime(2020,1,1)
teste_final = dt.datetime.now()

dados_teste = web.DataReader(empresa, 'yahoo', teste_inicio, teste_final)
precos_reais = dados_teste['Close'].values

total_dados = pd.concat((dados['Close'], dados_teste['Close']), axis = 0)

modelo_entrada = total_dados[len(total_dados) - len(dados_teste) - previsao_dias:].values
modelo_entrada = modelo_entrada.reshape(-1, 1)
modelo_entrada = normalizando.transform(modelo_entrada)


#Fazer previsoes nos valores de teste

x_teste = []

for x in range(previsao_dias, len(modelo_entrada)):
    x_teste.append(modelo_entrada[x-previsao_dias:x, 0])
    
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))

previsao_precos = modelo.predict(x_teste)
previsao_precos = normalizando.inverse_transform(previsao_precos)

#Representando Graficamente as Previsoes
plt.plot(precos_reais, color ='red', label = f"Valor Real das acoes de {empresa}")
plt.plot(previsao_precos, color="green", label = f"Previsao das acoes de {empresa}" )
plt.title(f"{empresa} Preco Acao")
plt.xlabel('Tempo')
plt.ylabel(f"{empresa} Preco Acao")
plt.legend()
plt.show()


#%%
#Prevendo os proximos dias

dados_reais =  [modelo_entrada[len(modelo_entrada) + 1 - previsao_dias:len(modelo_entrada + 1), 0]]
dados_reais =  np.array(dados_reais)
dados_reais = np.reshape(dados_reais, (dados_reais.shape[0], dados_reais.shape[1], 1))

previsao = modelo.predict(dados_reais)
previsao = normalizando.inverse_transform(previsao)

print(f"Previsao para amanha: {previsao}")