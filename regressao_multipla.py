from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import time
import controla_bd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# IMPORTAÇÕES NECESSÁRIAS PARA REGRESSÃO MÚLTIPLA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def prepara_df_para_regressao_multipla():
    """
    Prepara dados para regressão múltipla, seguindo mesma lógica do KNN
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

        print("Limpando dados...")
        colunas_necessarias = ['valor_empenhado', 'valor_liquidado', 'valor_pago']
        
        df = df.dropna(subset=colunas_necessarias)
        
        if df.empty:
            print(f"Atenção: Após a limpeza, não restaram dados válidos.")
            return

        df_final = pd.concat([df_final, df], ignore_index=True)
    
    print(f"Preparação concluída! Total de registros: {len(df_final)}")

    cursor.close()
    conexao.close()
    
    return df_final

def regressao_multipla(df):
    """
    Executa regressão múltipla usando valor_empenhado e valor_liquidado para prever valor_pago
    Mesma proporção de treinamento/teste: 30% teste, 70% treinamento
    """
    print("\n" + "="*60)
    print("REGRESSÃO MÚLTIPLA")
    print("="*60)
    
    # Selecionar features e target
    # Para regressão múltipla: usar valor_empenhado e valor_liquidado para prever valor_pago
    X = df[['valor_empenhado', 'valor_liquidado']].values
    y = df['valor_pago'].values

    print(f"\nTotal de amostras: {len(X)}")
    print(f"Features: valor_empenhado, valor_liquidado")
    print(f"Target: valor_pago")

    # Dividir em treino e teste com mesma proporção do KNN (30% teste)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42
    )

    print(f"\nAmostras de treinamento: {len(X_treino)}")
    print(f"Amostras de teste: {len(X_teste)}")

    # Normalizar os dados
    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino)
    X_teste_escalado = scaler.transform(X_teste)

    # Treinar o modelo
    print("\nTreinando modelo de regressão múltipla...")
    modelo = LinearRegression()
    modelo.fit(X_treino_escalado, y_treino)

    # Validação cruzada (cv=5)
    scores = cross_val_score(modelo, X_treino_escalado, y_treino, cv=5, scoring='r2')
    print(f"\nValidação Cruzada (cv=5):")
    print(f"  Scores R²: {scores}")
    print(f"  Média R²: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Fazer previsões
    y_pred_treino = modelo.predict(X_treino_escalado)
    y_pred_teste = modelo.predict(X_teste_escalado)

    # Calcular métricas
    r2_treino = r2_score(y_treino, y_pred_treino)
    r2_teste = r2_score(y_teste, y_pred_teste)
    rmse_treino = np.sqrt(mean_squared_error(y_treino, y_pred_treino))
    rmse_teste = np.sqrt(mean_squared_error(y_teste, y_pred_teste))
    mae_treino = mean_absolute_error(y_treino, y_pred_treino)
    mae_teste = mean_absolute_error(y_teste, y_pred_teste)

    print("\n" + "="*60)
    print("MÉTRICAS DO MODELO")
    print("="*60)
    print(f"\nR² (Coeficiente de Determinação):")
    print(f"  Treinamento: {r2_treino:.4f}")
    print(f"  Teste: {r2_teste:.4f}")
    print(f"\nRMSE (Raiz do Erro Quadrático Médio):")
    print(f"  Treinamento: R$ {rmse_treino:,.2f}")
    print(f"  Teste: R$ {rmse_teste:,.2f}")
    print(f"\nMAE (Erro Absoluto Médio):")
    print(f"  Treinamento: R$ {mae_treino:,.2f}")
    print(f"  Teste: R$ {mae_teste:,.2f}")
    print(f"\nCoeficientes:")
    print(f"  valor_empenhado: {modelo.coef_[0]:.6f}")
    print(f"  valor_liquidado: {modelo.coef_[1]:.6f}")
    print(f"Intercepto: {modelo.intercept_:,.2f}")

    # Gerar gráficos
    gera_graficos_regressao_multipla(df, X_treino, X_teste, y_treino, y_teste, 
                                     y_pred_treino, y_pred_teste, scaler, modelo)

def gera_graficos_regressao_multipla(df, X_treino, X_teste, y_treino, y_teste, 
                                     y_pred_treino, y_pred_teste, scaler, modelo):
    """
    Gera gráficos para visualização do modelo de regressão múltipla
    """
    print("\nGerando gráficos...")

    # Desfazer normalização para plotting
    X_treino_original = scaler.inverse_transform(X_treino)
    X_teste_original = scaler.inverse_transform(X_teste)

    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Predições vs Valores Reais (Treinamento)",
            "Predições vs Valores Reais (Teste)",
            "Resíduos do Treinamento",
            "Resíduos do Teste"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Subplot 1: Predições vs Valores Reais (Treinamento)
    fig.add_trace(
        go.Scatter(
            x=y_treino,
            y=y_pred_treino,
            mode='markers',
            marker=dict(color='#3498db', size=6, opacity=0.6),
            name='Treinamento'
        ),
        row=1, col=1
    )
    
    # Linha perfeita de predição
    min_val = min(y_treino.min(), y_pred_treino.min())
    max_val = max(y_treino.max(), y_pred_treino.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            name='Predição Perfeita'
        ),
        row=1, col=1
    )

    # Subplot 2: Predições vs Valores Reais (Teste)
    fig.add_trace(
        go.Scatter(
            x=y_teste,
            y=y_pred_teste,
            mode='markers',
            marker=dict(color='#2ecc71', size=6, opacity=0.6),
            name='Teste'
        ),
        row=1, col=2
    )
    
    # Linha perfeita de predição
    min_val = min(y_teste.min(), y_pred_teste.min())
    max_val = max(y_teste.max(), y_pred_teste.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            name='Predição Perfeita'
        ),
        row=1, col=2
    )

    # Subplot 3: Resíduos Treinamento
    residuos_treino = y_treino - y_pred_treino
    fig.add_trace(
        go.Scatter(
            x=y_pred_treino,
            y=residuos_treino,
            mode='markers',
            marker=dict(color='#3498db', size=6, opacity=0.6),
            name='Resíduos (Treino)'
        ),
        row=2, col=1
    )
    
    # Linha de referência (y=0)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        row=2, col=1
    )

    # Subplot 4: Resíduos Teste
    residuos_teste = y_teste - y_pred_teste
    fig.add_trace(
        go.Scatter(
            x=y_pred_teste,
            y=residuos_teste,
            mode='markers',
            marker=dict(color='#2ecc71', size=6, opacity=0.6),
            name='Resíduos (Teste)'
        ),
        row=2, col=2
    )
    
    # Linha de referência (y=0)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        row=2, col=2
    )

    # Atualizar layout
    fig.update_xaxes(title_text="Valores Reais (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Predições (R$)", row=1, col=1)
    
    fig.update_xaxes(title_text="Valores Reais (R$)", row=1, col=2)
    fig.update_yaxes(title_text="Predições (R$)", row=1, col=2)
    
    fig.update_xaxes(title_text="Predições (R$)", row=2, col=1)
    fig.update_yaxes(title_text="Resíduos (R$)", row=2, col=1)
    
    fig.update_xaxes(title_text="Predições (R$)", row=2, col=2)
    fig.update_yaxes(title_text="Resíduos (R$)", row=2, col=2)

    fig.update_layout(
        title_text="Regressão Múltipla: Valor Empenhado + Valor Liquidado → Valor Pago",
        template="plotly_white",
        height=900,
        showlegend=True
    )

    print("Salvando gráfico...")
    fig.write_html("regressao_multipla_graficos.html")
    #fig.show()

if __name__ == "__main__":
    df = prepara_df_para_regressao_multipla()
    regressao_multipla(df)
