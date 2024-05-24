from sklearn import preprocessing
from pickle import dump
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dados = pd.read_csv('Training/CSV/FinancialFraudDetection.csv', sep=',')

dados_numericos = dados[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
classe = dados[['isFraud']]
dados_categoricos = dados[['type']]

# NORMALIZAÇÃO DADOS CATEGÓRICOS
dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, prefix_sep='_', dtype=int)
colunas_categoricas = dados_categoricos_normalizados.columns
dump(colunas_categoricas, open("Training/Models/colunas_categoricas.pkl", "wb"))

# NORMALIZAÇÃO DADOS NUMÉRICOS
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
dump(modelo_normalizador, open('Training/Models/Normalizer.pkl', 'wb'))

# CRIAR UM DATAFRAME COM OS DADOS NORMALIZADOS
dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how='left')

# BALANCEAMENTO
# SEGMENTAR OS DADOS EM ATRIBUTOS E CLASSES
dados_atributos = dados_normalizados_final

# IMPRIMIR AS FRQUÊNCIAS DAS CLASSES ANTES DE BALANCEAR
print('Frequencia de classes antes do balanceamento')
classes_count = classe.value_counts()
print(classes_count)

# CONSTRUIR UM OBJETO A PARTIR DO SMOTE
resampler = SMOTE()

# EXECUTAR O BALANCEAMENTO
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, classe)

# VERIFICAR A FREQUENCIA DAS CLASSES APÓS O BALANCEAMENTO
print('Frequencia de classes após balanceamento')
classes_count = dados_classes_b.value_counts()
print(classes_count)

# CONVERTER OS DADOS BALANCEADOS EM DATAFRAMES
dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_atributos_b.columns = dados_atributos.columns

dados_classes_b = pd.DataFrame(dados_classes_b)
dados_classes_b.columns = ['isFraud']

dados_finais_b = dados_atributos_b.join(dados_classes_b, how='left')

# HIPERPARAMETRIZAÇÃO

# Calculo Amostral para realização das hiperparametrizações
# População pós balanceamento: 12.708.814
# Erro amostral: 3%
# Nível de confiança: 95%
# Distribuição da população: 80/20
### Resultado: 683

dados_finais_b_train = dados_finais_b.sample(683)
class_train_b = dados_finais_b_train['isFraud']
atr_train_b = dados_finais_b_train.drop(columns=['isFraud'])

# Montagem da grade de parâmentros
# Número de árvores na floresta
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
# Número de atributos considerados em cada segmento
max_features = ['auto', 'srqt']
# Número máximo de folhas em cada árvore
max_depth = [int(x) for x in np.linspace(10,110, num = 3)]
max_depth.append(None)

# Alternative
random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth}

# INICIAR A BUSCA PELO MELHORES HIPERPARAMETROS

# rf = instanciação da randomForest
rf_grid = RandomForestClassifier()
rf_grid = GridSearchCV(rf_grid,random_grid,refit=True,verbose=2)
rf_grid.fit(atr_train_b, class_train_b)

print("MELHORES HIPERPARÂMETROS")
print(rf_grid.best_params_)

# TREINAR O MODELO
rf = RandomForestClassifier(rf_grid.best_params_)
FinancialFraudRF = rf.fit(dados_atributos_b, dados_classes_b.values.ravel())
dump(FinancialFraudRF, open('Training/Models/synthetic_financial_fraud_detection_RF_2024.pkl', 'wb'))

# CROSS VALIDATION
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(FinancialFraudRF, dados_atributos_b, dados_classes_b.values.ravel(), scoring=scoring)
print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())

x = np.array(["test_precision_macro", "test_recall_macro"])
y = np.array([scores_cross['test_precision_macro'].mean(), scores_cross['test_recall_macro'].mean()])

plt.bar(x, y)
plt.savefig('Training/CrossValidation.png')

plt.show()

# Resultados:
# test_precision_macro = 0.9092801273458389
# test_recall_macro = 0.8602703588552417