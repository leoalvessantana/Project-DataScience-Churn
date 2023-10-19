# Churn Rate para uma operadora de telecomunicações

<div align="center">
  <img src="imagens\Logo.png" style="width: 100%;">
</div>

## Introdução

Esse é um projeto desenvolvido por mim durante o Challenge Dados 3ª Edição promovido pela Alura, uma plataforma online de cursos de tecnologia.
A ideia do Challenge é simular uma experiência de trabalho. 


## Contexto do Problema

Nesse desafio vamos ajudar uma operadora de telecomunicações ficticia, chamada **Novexus**, na identificação de clientes que teriam uma maior chance de deixar a Novexus. 
Na reunião inicial com as pessoas responsáveis pela área de vendas da empresa, foi explicada a importância de reduzir a Taxa de Evasão de Clientes, também conhecida como **Churn Rate**.
Dessa forma, vamos investigar algumas características de clientes ou dos planos de clientes para tentar **CLASSIFICAR** essas pessoas como potenciais candidatas a deixar a empresa ou não.

Neste desafio, vamos explorar, tratar e modelar dados em busca de insights valiosos. Vamos focar na otimização e disponibilização do modelo para alcançar o melhor resultado possível na tomada de decisões estratégicas da Novexus. Além disso, como tarefa especial vamos desenvolver um APP Web que facilite a classificação de potenciais clientes, tornando todo o processo mais ágil e eficiente.



## Etapa 1 - Limpeza e análise exploratória dos dados


Inicialmente tratamos e analisamos a base de dados fornecida pela Novexus. Conseguimos extrair alguns insiths que nos dizem o motivo pelo quais os clientes estão deixando de utilizar 
os serviços da empresa. 

[Notebook da etapa 1](Notebooks/etapa1.ipynb)


## Etapa 2 - Construindo e otimizando modelos de machine learning

Nessa etapa o objetivo é encontrar o modelo de machile learning que melhor se comporte com os nossos dados.

[Notebook da etapa 2](Notebooks/etapa2.ipynb)


## Etapa 3 - Deploy do modelo de machine learning


Nessa última etapa utilizamos o Streamlit para fazer o deploy do nosso modelo. Nesse nosso aplicativo Web ao colocar as informações do cliente mostramos se o cliente é 
um potencial candidato a deixar a empresa ou não.


