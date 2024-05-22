import pandas as pd
from pickle import load

def new_instance(step:int, type:str, amount:float, oldbalanceOrg:float, newbalanceOrig:float, oldbalanceDest:float, newbalanceDest:float):
    synthetic_financial_fraud_detection_RF_model = load(open("Training/Models/synthetic_financial_fraud_detection_RF_2024.pkl", "rb"))
    colunas_categoricas = pd.read_pickle("Training/Models/colunas_categoricas.pkl")
    normalizador = load(open("Training/Models/Normalizer.pkl", "rb"))
    dados_categoricos_base = pd.DataFrame(columns=colunas_categoricas)

    df_base = pd.DataFrame([[step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]], columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

    # SEPARAÇÃO DOS DATAFRAMES
    dados_numericos = df_base[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    dados_categoricos = df_base[['type']]

    # NORMALIZAÇÃO DOS DADOS CATEGÓRICOS
    dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, prefix_sep='_', dtype=int)

    # NORMALIZAÇÃO DOS DADOS NUMÉRICOS
    dados_numericos_normalizados = normalizador.transform(dados_numericos)
    dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

    # JUNÇÃO DOS DADOS NORMALIZADOS COM A BASE DE COLUNAS
    dados_completos = pd.concat([dados_categoricos_base, dados_categoricos_normalizados], axis=0)
    dados_completos = dados_completos.where(pd.notna(dados_completos), other=0)
    dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')

    # EXECUÇÃO DO PREDICT
    predict = synthetic_financial_fraud_detection_RF_model.predict(dados_completos)

    print(predict)