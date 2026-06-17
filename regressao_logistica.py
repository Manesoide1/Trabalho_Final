from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import time
import controla_bd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# IMPORTAÇÕES NECESSÁRIAS PARA REGRESSÃO LOGÍSTICA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

def prepara_df_para_regressao_logistica():
    """
    Prepara dados para regressão logística, seguindo mesma lógica do KNN
    Usa os mesmos dados e mesma proporção de treinamento/teste
    """
    conexao, cursor = controla_bd.conecta_banco_de_dados()
    controla_bd.cria_banco_e_tabelas(cursor)

    df_final = pd.DataFrame()

    for codigo_orgao in pd.read_sql("SELECT codigo_orgao_superior FROM despesas_governo_federal", conexao)['codigo_orgao_superior'].unique():
        # Pegando os dados do órgão específico
        query = f"SELECT * FROM despesas_governo_federal WHERE codigo_orgao_superior = {codigo_orgao}"
        
        print(f"\nBuscando dados no MySQL...")
        df = pd.read_sql(query, conexao)
        print(f"Dados carregados! Total de registros: {len(df)}")

        df = df.sample(n=int(df.shape[0] * 0.05), random_state=42)  # Reduzindo a amostra para 5%

        print(f"Dados selecionados! Total de registros: {len(df)}")

        if df.empty:
            print(f"Atenção: A tabela está vazia.")
            return

        print("Limpando dados e detectando anomalias...")
        colunas_gasto = ['valor_empenhado', 'valor_liquidado', 'valor_pago']
        
        df = df.dropna(subset=colunas_gasto)
        
        if df.empty:
            print(f"Atenção: Após a limpeza, não restaram dados válidos.")
            return
        
        # Detectar anomalias (mesmo critério do KNN)
        dif_emp_liq = df['valor_empenhado'] - df['valor_liquidado']
        dif_liq_pag = df['valor_liquidado'] - df['valor_pago']

        mediana_emp_liq = dif_emp_liq.median()
        mediana_liq_pag = dif_liq_pag.median()

        mad_emp_liq = (dif_emp_liq - mediana_emp_liq).abs().median()
        mad_liq_pag = (dif_liq_pag - mediana_liq_pag).abs().median()

        desvio_robusto_emp_liq = 1.4826 * mad_emp_liq
        desvio_robusto_liq_pag = 1.4826 * mad_liq_pag
        
        multiplicador_sigma = 3
        limite_financeiro_minimo = 5000

        df['anomalia'] = np.where(
            ((abs(dif_emp_liq - mediana_emp_liq) > desvio_robusto_emp_liq * multiplicador_sigma) & 
            (abs(dif_emp_liq) > limite_financeiro_minimo)) |
            
            ((abs(dif_liq_pag - mediana_liq_pag) > desvio_robusto_liq_pag * multiplicador_sigma) & 
            (abs(dif_liq_pag) > limite_financeiro_minimo)) |
            
            ((df['valor_pago'] < 0) & (df['valor_liquidado'] >= 0)) |
            ((df['valor_liquidado'] < 0) & (df['valor_empenhado'] >= 0)) |
            
            ((df['valor_empenhado'] < 0) & (df['valor_liquidado'] == 0) & (df['valor_pago'] == 0)) |
            ((df['valor_liquidado'] < 0) & (df['valor_empenhado'] == 0) & (df['valor_pago'] == 0)) |
            ((df['valor_pago'] < 0) & (df['valor_liquidado'] == 0) & (df['valor_empenhado'] == 0)),
            1,
            0
        )

        df_final = pd.concat([df_final, df], ignore_index=True)
    
    print(f"Preparação concluída! Total de registros: {len(df_final)}")
    print(f"Total de anomalias: {df_final['anomalia'].sum()}")
    print(f"Total de não-anomalias: {len(df_final) - df_final['anomalia'].sum()}")

    cursor.close()
    conexao.close()
    
    return df_final

def regressao_logistica(df):
    """
    Executa regressão logística para classificação de anomalias
    Mesma proporção de treinamento/teste: 30% teste, 70% treinamento
    """
    print("\n" + "="*60)
    print("REGRESSÃO LOGÍSTICA")
    print("="*60)
    
    # Selecionar features e target
    # Usar as mesmas features que o KNN para comparação
    X = df[['valor_empenhado', 'valor_liquidado', 'valor_pago']].values
    y = df['anomalia'].values

    print(f"\nTotal de amostras: {len(X)}")
    print(f"Features: valor_empenhado, valor_liquidado, valor_pago")
    print(f"Target: anomalia (binária: 0=normal, 1=anomalia)")

    # Dividir em treino e teste com mesma proporção do KNN (30% teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print(f"\nAmostras de treinamento: {len(X_treino)}")
    print(f"Amostras de teste: {len(X_teste)}")
    print(f"Proporção de anomalias (treino): {y_treino.sum() / len(y_treino):.2%}")
    print(f"Proporção de anomalias (teste): {y_teste.sum() / len(y_teste):.2%}")

    # Normalizar os dados
    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino)
    X_teste_escalado = scaler.transform(X_teste)

    # Treinar o modelo
    print("\nTreinando modelo de regressão logística...")
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_treino_escalado, y_treino)

    # Validação cruzada (cv=5)
    scores = cross_val_score(modelo, X_treino_escalado, y_treino, cv=5, scoring='accuracy')
    print(f"\nValidação Cruzada (cv=5):")
    print(f"  Scores Acurácia: {scores}")
    print(f"  Média Acurácia: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Fazer previsões
    y_pred_treino = modelo.predict(X_treino_escalado)
    y_pred_teste = modelo.predict(X_teste_escalado)
    
    # Probabilidades para ROC curve
    y_proba_treino = modelo.predict_proba(X_treino_escalado)[:, 1]
    y_proba_teste = modelo.predict_proba(X_teste_escalado)[:, 1]

    # Calcular métricas
    accuracy_treino = accuracy_score(y_treino, y_pred_treino)
    accuracy_teste = accuracy_score(y_teste, y_pred_teste)
    
    precision_treino = precision_score(y_treino, y_pred_treino, zero_division=0)
    precision_teste = precision_score(y_teste, y_pred_teste, zero_division=0)
    
    recall_treino = recall_score(y_treino, y_pred_treino, zero_division=0)
    recall_teste = recall_score(y_teste, y_pred_teste, zero_division=0)
    
    f1_treino = f1_score(y_treino, y_pred_treino, zero_division=0)
    f1_teste = f1_score(y_teste, y_pred_teste, zero_division=0)

    # Calcular AUC-ROC
    fpr, tpr, _ = roc_curve(y_teste, y_proba_teste)
    auc_score = auc(fpr, tpr)

    print("\n" + "="*60)
    print("MÉTRICAS DO MODELO")
    print("="*60)
    print(f"\nAcurácia (Accuracy):")
    print(f"  Treinamento: {accuracy_treino:.4f}")
    print(f"  Teste: {accuracy_teste:.4f}")
    
    print(f"\nPrecisão (Precision):")
    print(f"  Treinamento: {precision_treino:.4f}")
    print(f"  Teste: {precision_teste:.4f}")
    
    print(f"\nRevogação (Recall/Sensibilidade):")
    print(f"  Treinamento: {recall_treino:.4f}")
    print(f"  Teste: {recall_teste:.4f}")
    
    print(f"\nF1-Score:")
    print(f"  Treinamento: {f1_treino:.4f}")
    print(f"  Teste: {f1_teste:.4f}")
    
    print(f"\nAUC-ROC (Teste): {auc_score:.4f}")

    print(f"\nCoeficientes:")
    print(f"  valor_empenhado: {modelo.coef_[0][0]:.6f}")
    print(f"  valor_liquidado: {modelo.coef_[0][1]:.6f}")
    print(f"  valor_pago: {modelo.coef_[0][2]:.6f}")
    print(f"Intercepto: {modelo.intercept_[0]:.6f}")

    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO (TESTE)")
    print("="*60)
    print(classification_report(y_teste, y_pred_teste, 
                              target_names=['Normal', 'Anomalia']))

    # Gerar gráficos
    gera_graficos_regressao_logistica(
        y_treino, y_teste, y_pred_treino, y_pred_teste,
        y_proba_treino, y_proba_teste, fpr, tpr, auc_score
    )

def gera_graficos_regressao_logistica(y_treino, y_teste, y_pred_treino, y_pred_teste,
                                      y_proba_treino, y_proba_teste, fpr, tpr, auc_score):
    """
    Gera gráficos para visualização do modelo de regressão logística
    """
    print("\nGerando gráficos...")

    # Calcular matrizes de confusão
    cm_treino = confusion_matrix(y_treino, y_pred_treino)
    cm_teste = confusion_matrix(y_teste, y_pred_teste)

    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Matriz de Confusão (Treinamento)",
            "Matriz de Confusão (Teste)",
            "Distribuição de Probabilidades Preditas (Treinamento)",
            "Curva ROC (Teste)"
        ),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # Subplot 1: Matriz de Confusão Treinamento
    fig.add_trace(
        go.Heatmap(
            z=cm_treino,
            x=['Normal', 'Anomalia'],
            y=['Normal', 'Anomalia'],
            text=cm_treino,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )

    # Subplot 2: Matriz de Confusão Teste
    fig.add_trace(
        go.Heatmap(
            z=cm_teste,
            x=['Normal', 'Anomalia'],
            y=['Normal', 'Anomalia'],
            text=cm_teste,
            texttemplate="%{text}",
            colorscale='Greens',
            showscale=False
        ),
        row=1, col=2
    )

    # Subplot 3: Histograma de probabilidades
    fig.add_trace(
        go.Histogram(
            x=y_proba_treino[y_treino == 0],
            name='Normal (Treino)',
            opacity=0.7,
            marker_color='#3498db'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=y_proba_treino[y_treino == 1],
            name='Anomalia (Treino)',
            opacity=0.7,
            marker_color='#e74c3c'
        ),
        row=2, col=1
    )

    # Subplot 4: Curva ROC
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC={auc_score:.4f})',
            line=dict(color='#2ecc71', width=3)
        ),
        row=2, col=2
    )
    
    # Linha diagonal (classificador aleatório)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Aleatório (AUC=0.5)',
            line=dict(color='#95a5a6', width=2, dash='dash')
        ),
        row=2, col=2
    )

    # Atualizar layout
    fig.update_xaxes(title_text="Predito", row=1, col=1)
    fig.update_yaxes(title_text="Real", row=1, col=1)
    
    fig.update_xaxes(title_text="Predito", row=1, col=2)
    fig.update_yaxes(title_text="Real", row=1, col=2)
    
    fig.update_xaxes(title_text="Probabilidade Predita", row=2, col=1)
    fig.update_yaxes(title_text="Frequência", row=2, col=1)
    
    fig.update_xaxes(title_text="Taxa de Falso Positivo (FPR)", row=2, col=2)
    fig.update_yaxes(title_text="Taxa de Verdadeiro Positivo (TPR)", row=2, col=2)

    fig.update_layout(
        title_text="Regressão Logística: Detecção de Anomalias",
        template="plotly_white",
        height=900,
        showlegend=True
    )

    print("Salvando gráfico...")
    fig.write_html("regressao_logistica_graficos.html")
    #fig.show()

if __name__ == "__main__":
    df = prepara_df_para_regressao_logistica()
    regressao_logistica(df)
