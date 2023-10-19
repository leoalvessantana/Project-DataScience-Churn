# =============================================================================
                                ## BIBLIOTECAS ##
# =============================================================================
import time

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xgboost as xgb
from PIL import Image

# =============================================================================
                                ## FUNÇÕES ##
# =============================================================================

def tratamentos(dados):
    '''Função para colocar os dados de entrada na forma como o modelo foi treinado'''
    
    ## Transformando os dados categóricos em binárias
    traducao_dic = {'Yes': 1,
                'No': 0,
                'Female': 0,
                'Male': 1}
    dadosmodificados = dados[['customer_gender', 'customer_partner', 'customer_dependents', 'phone_service', 'paperless_billing']].replace(traducao_dic)
  
    
    ## Usando get_dummies() nas variáveis com mais de duas categorias
    dummie_dados = pd.get_dummies(dados[['multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 'payment_method' ]])
    

    ##junção dos dados trasformados com os que já tinhamos
    dados_final = pd.concat([dadosmodificados, dummie_dados, dados[['customer_tenure', 'monthly_charges', 'total_charges']]], axis=1)

    return dados_final



def previsao_churn(dados_entrada, dados, modelo):
    '''Função para fazer a previsão de churn'''
    
    # concatenando o dado de entrada com nossos dados

    df_entrada = pd.DataFrame([dados_entrada], columns = dados.columns)
    dados_final =  pd.concat([dados, df_entrada], ignore_index=True)
    
    # tratando os dados
    df = tratamentos(dados_final)
    
    # predição apenas do dado de entrada
    predicao = modelo.predict(df.iloc[-1:])[0]
    
    # probabilidade do resultado esta correto
    probabilidade = modelo.predict_proba(df.iloc[-1:])[0, 0]

    
    return predicao, probabilidade




def exibir_previsao(dados_entrada, dados, modelo):
    '''Função para exibir a previsão'''
    predicao, probabilidade = previsao_churn(dados_entrada, dados, modelo)
    probab = round(probabilidade * 100, 2) 
    st.markdown("<h3 style='text-align: center;'>Resultado da previsão</h3>", unsafe_allow_html=True)  

    if predicao == 0:
        st.error('Churn: Não', icon="❌")
        st.progress(round(probabilidade * 100), text=f':black[Probabilidade de permanecer como nosso cliente: {round(probabilidade * 100, 2)}%]')
    else:
        st.success('Churn: Sim', icon="✅")
        st.progress(round(probabilidade * 100), text=f':black[Probabilidade de deixar de ser nosso cliente: {round(probabilidade * 100, 2)}%]')


    


    
    

# =============================================================================
                                ## VISUAL ##
# =============================================================================

# Configurando o tema do aplicativo
st.set_page_config(
    page_title="Churn Predictor - Novexus",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Configurações da página Streamlit (streamlit run app.py)  
im = Image.open('imagens/Logo.png', 'r')
st.image(im, use_column_width = True)
st.markdown("<h1 style='text-align: center;'>Churn Predictor</h1>", unsafe_allow_html=True)  # titulo






# Coletar informações do usuário 

st.header('Informações gerais do cliente')    
coluna1, coluna2, coluna3, coluna4 = st.columns(4)
with coluna1:
    customer_gender = st.radio('Gênero', ['Female', 'Male'])
with coluna2:
    customer_senior_citizen = st.radio('Mais de 65 anos', ['Yes', 'No'])
with coluna3:
    customer_partner = st.radio('Possui parceiro(a)', ['Yes', 'No'])
with coluna4:
    customer_dependents = st.radio('Possui dependentes', ['Yes', 'No'])

    
st.header('Serviços telefônicos')
coluna1, coluna2 = st.columns(2)
with coluna1:
    phone_service = st.radio('Assinatura de serviço telefônico', ['Yes', 'No'])
with coluna2:
    multiple_lines = st.radio('Assinatura de mais de uma linha de telefone', ['Yes', 'No'])

    
st.header('Serviços de internet')
with st.expander("Opções de Assinatura"):
    internet_service = st.radio('Assinatura de um provedor internet', ['DSL', 'Fiber optic', 'No'])
with st.expander("Opções Adicionais"):
    online_security = st.radio('Assinatura adicional de segurança online', ['No', 'Yes'])
    online_backup = st.radio('Assinatura adicional de backup online', ['Yes', 'No'])
    device_protection = st.radio('Assinatura adicional de proteção no dispositivo', ['No', 'Yes'])
    tech_support = st.radio('Assinatura adicional de suporte técnico (menos tempo de espera)', ['Yes', 'No'])
    streaming_tv = st.radio('Assinatura de TV a cabo', ['Yes', 'No'])
    streaming_movies = st.radio('Assinatura de streaming de filmes', ['No', 'Yes'])

st.header('Assuntos financeiros do cliente')
customer_tenure = st.slider('Meses de contrato', 0, 60, 1)
coluna1, coluna2, coluna3 = st.columns(3)
with coluna1:
    contract = st.radio('Tipo de contrato', ['One year', 'Month-to-month', 'Two year'])
with coluna2:
    paperless_billing = st.radio('Receber a fatura online', ['Yes', 'No'])
with coluna3:
    payment_method = st.radio('Forma de pagamento', ['Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)'])
monthly_charges = st.number_input("Total de gastos por mês")
total_charges = st.number_input("Total de gastos")



dados_entrada = {
        'customer_gender': customer_gender,
        'customer_senior_citizen': customer_senior_citizen,
        'customer_partner': customer_partner,
        'customer_dependents': customer_dependents,
        'customer_tenure': customer_tenure,        
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,                          
        'paperless_billing': paperless_billing,     
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges
    }






# =============================================================================
                                ## PREVENDO O CHURN ##
# =============================================================================

# Carregando os dados usados no treinamento
dados = pd.read_csv('Notebooks/dados_etapa1.csv')
dados = dados.iloc[:, 2:] # Removendo Churn


# Carregando o modelo treinado
# modelo = pickle.load(open('Notebooks/XGBClassifier.pkl', 'rb'))
modelo = joblib.load('XGBClassifier.pkl')



# Fazendo a previsao    
if st.button('Fazer previsão'):
    with st.spinner('Fazendo a previsão...'):
        time.sleep(2)
        exibir_previsao(dados_entrada, dados, modelo)





    
    
    