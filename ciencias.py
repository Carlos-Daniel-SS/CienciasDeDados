
'''
Passo a Passo de um Projeto de Ciência de Dados
Passo 1: Entendimento do Desafio
Passo 2: Entendimento da Área/Empresa
Passo 3: Extração/Obtenção de Dados
Passo 4: Ajuste de Dados (Tratamento/Limpeza)
Passo 5: Análise Exploratória
Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
Passo 7: Interpretação de Resultados

'''


# Script para realizar previsões de vendas baseado no investimento em propagandas

# Importando bibliotecas para importar e realizar o tratamento dos dados:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliotecas para criar e treinar as inteligencias artificiais:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv('advertising.csv')
print(tabela.info())

# Calculando a correlação de cada produto com os valores das vendas e exibindo em uma gráfico
print(tabela.corr())
sns.heatmap(tabela.corr(), cmap = 'Greens', annot = True)
plt.show()

# Realizando a divisão da base de dados para treinamento e teste da IA:

y = tabela['Vendas']
x = tabela.drop('Vendas', axis=1)

# DIvisão realiza pela biblioteca "from sklearn.model_selection import train_test_split"

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, random_state=1)

# Criando dois modelos Inteligencias artificiais:

modelo_regressaolinear = LinearRegression()     # -> IA de regressão linear 
modelo_arvoredecisao = RandomForestRegressor()  # -> IA de Arvore de decisão

# Realizando o treinamento dos modelos de IA:

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Realiando a previsão com cada modelo de IA:

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# Aqui indica o percentual de precisão de cada modelo:

print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))

# Criando  nova tabela para realizar previsão em novos dados:

nova_tabela = pd.read_csv('novos.csv')

# Realizando previsão:
previsao = modelo_arvoredecisao.predict(nova_tabela)
print((previsao))

# Inserindo a coluna "Vendas" de acordo com resultados da previsão:
nova_tabela['Vendas'] = previsao
print(nova_tabela)