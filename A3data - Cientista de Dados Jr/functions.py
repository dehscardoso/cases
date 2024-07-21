import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import make_scorer, roc_auc_score, precision_score, recall_score, f1_score, fbeta_score


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 100)



def plot_churn_rate_by_category(df, categories):
    """
    Plota gráficos de barras empilhadas mostrando a taxa de churn por várias categorias.

    Parâmetros:
    df (DataFrame): O dataframe contendo os dados.
    categories (list): Lista dos nomes das colunas categóricas para as quais as taxas de churn serão plotadas.
    """
    
    # Configurar o número de subplots (2 colunas)
    num_categories = len(categories)
    num_rows = (num_categories + 1) // 2  # Número de linhas necessárias para 2 colunas
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
    axes = axes.flatten()  
    
    for i, category in enumerate(categories):
        # Calcular a taxa de churn e não-churn por categoria
        churn_counts = df.groupby([category, 'Churn']).size().unstack(fill_value=0)
        churn_rates = churn_counts.divide(churn_counts.sum(axis=1), axis=0) * 100
        
        # Resetar o índice para tornar a categoria uma coluna
        churn_rates = churn_rates.reset_index()
        
        churn_rates.columns = [category, 'No Churn (%)', 'Churn (%)']
        
        # Criar o gráfico 
        ax = axes[i]
        churn_rates.plot(x=category, kind='bar', stacked=True, ax=ax,
                         color=['#66B2FF', '#FF9999'], edgecolor='none')

        # Adicionar as porcentagens no gráfico
        for j, row in churn_rates.iterrows():
            no_churn_height = row['No Churn (%)']
            churn_height = row['Churn (%)']
            ax.text(j, no_churn_height / 2, f'{no_churn_height:.1f}%', ha='center', va='center', color='black')
            ax.text(j, no_churn_height + churn_height / 2, f'{churn_height:.1f}%', ha='center', va='center', color='black')
        
        ax.set_title(f'Taxa de Churn por {category}')
        ax.set_xlabel(category)
        ax.set_ylabel('Porcentagem (%)')
        ax.legend(['No Churn', 'Churn'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
    
    # Remover eixos vazios, se houver
    for j in range(num_categories, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

    
    
    
def plot_variable_distribution(data, variable):
    """
    Plota a distribuição de uma única variável para clientes que saíram e para clientes que não saíram.
    
    Parâmetros:
    data (DataFrame): Um DataFrame com a coluna da variável a ser plotada e a coluna 'Churn'.
    variable (str): O nome da variável a ser plotada.
    """
    
    sns.set_context("paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    # Plotar a distribuição para clientes que não saíram
    sns.kdeplot(data[variable][data["Churn"] == 'No'],
                color='#66B2FF', shade=True, label="Not Churn")
    
    # Plotar a distribuição para clientes que saíram
    sns.kdeplot(data[variable][data["Churn"] == 'Yes'],
                color='#FF9999', shade=True, label="Churn")
    
    # Configurar legendas e rótulos
    plt.legend(loc='upper right')
    plt.ylabel('Density')
    plt.xlabel(variable)
    plt.title(f'Distribution of {variable} by Churn')
    
    # Exibir o gráfico
    plt.show()
    
    
def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """
    Treina um modelo LightGBM e visualiza a importância das características.

    Parâmetros:
    - X_train (pd.DataFrame): Características de treinamento.
    - y_train (pd.Series ou np.array): Alvo de treinamento.
    - X_test (pd.DataFrame): Características de teste.
    - y_test (pd.Series ou np.array): Alvo de teste.

    Retorna:
    - gbm (lgb.Booster): Modelo LightGBM treinado.
    - importance_df (pd.DataFrame): DataFrame com os nomes das características e suas importâncias.
    """
    # Define os conjuntos de dados para o LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Define os parâmetros para o LightGBM
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # Treina o modelo
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10)
    
    # Obtém a importância das características
    feature_importance = gbm.feature_importance()
    feature_names = gbm.feature_name()
    
    # Cria um DataFrame para a importância das características
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Plota a importância das características
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importância das Características')
    plt.ylabel('Características')
    plt.title('Importância das Características usando LightGBM')
    plt.show()
    
    # Retorna o modelo treinado e o DataFrame de importância das características
    return gbm, importance_df



def evaluate_models(X_train, y_train, models):
    """
    Avalia uma lista de modelos usando validação cruzada e retorna um DataFrame com os resultados.

    Parâmetros:
    X_train (array-like): Características de treinamento.
    y_train (array-like): Rótulos de treinamento.
    models (list of tuples): Lista de tuplas contendo o nome do modelo e o modelo em si.

    Retorna:
    pd.DataFrame: DataFrame contendo as métricas de avaliação para cada modelo.
    """
    
    acc_results = []
    auc_results = []
    precision_results = []
    recall_results = []
    f1_results = []
    f2_results = []
    names = []

    result_col = ["Algorithm", "ROC AUC Mean", "ROC AUC STD", "Accuracy Mean", "Accuracy STD",
                  "Precision Mean", "Precision STD", "Recall Mean", "Recall STD",
                  "F1 Score Mean", "F1 Score STD", "F2 Score Mean", "F2 Score STD"]
    
    model_results = pd.DataFrame(columns=result_col)

    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)

    # Definir scorers
    auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    precision_scorer = make_scorer(precision_score, average='macro')
    recall_scorer = make_scorer(recall_score, average='macro')
    f1_scorer = make_scorer(f1_score, average='macro')
    f2_scorer = make_scorer(fbeta_score, beta=2, average='macro')

    for name, model in models:
        names.append(name)

        cv_acc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        cv_auc_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=auc_scorer)
        cv_precision_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=precision_scorer)
        cv_recall_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=recall_scorer)
        cv_f1_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=f1_scorer)
        cv_f2_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=f2_scorer)

        acc_results.append(cv_acc_results)
        auc_results.append(cv_auc_results)
        precision_results.append(cv_precision_results)
        recall_results.append(cv_recall_results)
        f1_results.append(cv_f1_results)
        f2_results.append(cv_f2_results)

        model_results.loc[len(model_results)] = [name, 
                                               round(cv_auc_results.mean()*100,2),
                                               round(cv_auc_results.std()*100,2),
                                               round(cv_acc_results.mean()*100,2),
                                               round(cv_acc_results.std()*100,2),
                                               round(cv_precision_results.mean()*100,2),
                                               round(cv_precision_results.std()*100,2),
                                               round(cv_recall_results.mean()*100,2),
                                               round(cv_recall_results.std()*100,2),
                                               round(cv_f1_results.mean()*100,2),
                                               round(cv_f1_results.std()*100,2),
                                               round(cv_f2_results.mean()*100,2),
                                               round(cv_f2_results.std()*100,2)]

    return model_results.sort_values(by=['ROC AUC Mean'], ascending=False)
