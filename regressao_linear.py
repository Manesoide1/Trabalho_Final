from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import time
import controla_bd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# IMPORTAÇÕES NECESSÁRIAS PARA REGRESSÃO LINEAR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def prepara_df_para_regressao_linear():
    """
    Prepara dados para regressão linear, seguindo mesma lógica do KNN
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

        ''' --- REMOVIDO: Filtragem de outliers com base no valor empenhado (X% superior e inferior) ---

        # Remove o X% superior e inferior dos dados com base no valor original
        porcentagem_filtrada = 0.25  # X% superior e X% inferior

        # --- Superior: Remove os X% maiores valores de valor_empenhado ---
        limite_superior = df['valor_empenhado'].quantile(1 - porcentagem_filtrada)
        df_filtrado_maiores = df[df['valor_empenhado'] <= limite_superior]

        # --- Inferior: Remove os X% menores valores de valor_empenhado ---
        limite_inferior = df['valor_empenhado'].quantile(porcentagem_filtrada)
        df_filtrado_menores = df[df['valor_empenhado'] >= limite_inferior]

        # Combinar os dois DataFrames filtrados
        df = pd.concat([df_filtrado_maiores, df_filtrado_menores], ignore_index=True)

        '''

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

def regressao_linear(df):
    """
    Executa regressão linear simples com transformação logarítmica
    Usa valor_empenhado para prever valor_pago com log transformation
    Mesma proporção de treinamento/teste: 30% teste, 70% treinamento
    """
    print("\n" + "="*60)
    print("REGRESSÃO LINEAR SIMPLES COM TRANSFORMAÇÃO LOGARÍTMICA")
    print("="*60)
    
    # Aplicar transformação logarítmica aos dados
    # Usar log(1 + valor) para evitar log(0)
    X_log = np.sign(df[['valor_empenhado']]) * np.log1p(np.abs(df[['valor_empenhado']]))
    y_log = np.sign(df['valor_pago']) * np.log1p(np.abs(df['valor_pago']))
    
    # Manter referências dos dados originais também
    X_original = df[['valor_empenhado']]
    y_original = df['valor_pago']

    print(f"\nTotal de amostras: {len(X_original)}")
    print(f"Feature: valor_empenhado (com transformação logarítmica)")
    print(f"Target: valor_pago (com transformação logarítmica)")
    print(f"\nTransformação: log(1 + valor) para evitar log(0)")

    # Dividir em treino e teste com mesma proporção do KNN (30% teste)
    X_treino_log, X_teste_log, y_treino_log, y_teste_log, X_treino_orig, X_teste_orig, y_treino_orig, y_teste_orig = train_test_split(
        X_log, y_log, X_original, y_original,
        test_size=0.30,
        random_state=42
    )

    print(f"\nAmostras de treinamento: {len(X_treino_log)}")
    print(f"Amostras de teste: {len(X_teste_log)}")

    # Normalizar os dados log-transformados
    scaler = StandardScaler()
    X_treino_escalado = scaler.fit_transform(X_treino_log)
    X_teste_escalado = scaler.transform(X_teste_log)

    # Treinar o modelo nos dados log-transformados
    print("\nTreinando modelo de regressão linear (dados log-transformados)...")
    modelo = LinearRegression()
    modelo.fit(X_treino_escalado, y_treino_log)

    # Validação cruzada (cv=5)
    scores = cross_val_score(modelo, X_treino_escalado, y_treino_log, cv=5, scoring='r2')
    print(f"\nValidação Cruzada (cv=5):")
    print(f"  Scores R²: {scores}")
    print(f"  Média R²: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Fazer previsões em escala log
    y_pred_treino_log = modelo.predict(X_treino_escalado)
    y_pred_teste_log = modelo.predict(X_teste_escalado)
    
    # Inverter transformação logarítmica para escala original
    y_pred_treino = np.expm1(y_pred_treino_log)
    y_pred_teste = np.expm1(y_pred_teste_log)

    # Calcular métricas na escala original (para comparabilidade)
    r2_treino = r2_score(y_treino_orig, y_pred_treino)
    r2_teste = r2_score(y_teste_orig, y_pred_teste)
    rmse_treino = np.sqrt(mean_squared_error(y_treino_orig, y_pred_treino))
    rmse_teste = np.sqrt(mean_squared_error(y_teste_orig, y_pred_teste))
    mae_treino = mean_absolute_error(y_treino_orig, y_pred_treino)
    mae_teste = mean_absolute_error(y_teste_orig, y_pred_teste)
    
    # Também calcular R² na escala log para análise
    r2_treino_log = r2_score(y_treino_log, y_pred_treino_log)
    r2_teste_log = r2_score(y_teste_log, y_pred_teste_log)

    print("\n" + "="*60)
    print("MÉTRICAS DO MODELO (Escala Original - para Comparabilidade)")
    print("="*60)
    print(f"\nR² (Coeficiente de Determinação - Escala Original):")
    print(f"  Treinamento: {r2_treino:.4f}")
    print(f"  Teste: {r2_teste:.4f}")
    print(f"\nR² (Coeficiente de Determinação - Escala Log):")
    print(f"  Treinamento: {r2_treino_log:.4f}")
    print(f"  Teste: {r2_teste_log:.4f}")
    print(f"\nRMSE (Raiz do Erro Quadrático Médio - Escala Original):")
    print(f"  Treinamento: R$ {rmse_treino:,.2f}")
    print(f"  Teste: R$ {rmse_teste:,.2f}")
    print(f"\nMAE (Erro Absoluto Médio - Escala Original):")
    print(f"  Treinamento: R$ {mae_treino:,.2f}")
    print(f"  Teste: R$ {mae_teste:,.2f}")
    
    print("\n" + "="*60)
    print("ANÁLISE DA TRANSFORMAÇÃO LOGARÍTMICA")
    print("="*60)
    print(f"\nCoeficiente Angular (Escala Log): {modelo.coef_[0]:.6f}")
    print(f"  Interpretação: A cada aumento de 1% em valor_empenhado,")
    print(f"  valor_pago aumenta aproximadamente {modelo.coef_[0]:.4f}% (semielasticidade)")
    print(f"\nIntercepto (Escala Log): {modelo.intercept_:.4f}")

    # Gerar gráficos
    gera_graficos_regressao_linear(df, X_treino_orig, X_teste_orig, y_treino_orig, y_teste_orig, 
                                   y_pred_treino, y_pred_teste, scaler, modelo, X_treino_log, X_teste_log)

def gera_graficos_regressao_linear(df, X_treino, X_teste, y_treino, y_teste, 
                                   y_pred_treino, y_pred_teste, scaler, modelo, 
                                   X_treino_log=None, X_teste_log=None):
    """
    Gera gráficos para visualização do modelo de regressão linear com transformação logarítmica
    """
    print("\nGerando gráficos...")

    # Se dados log-transformados foram fornecidos, usar os dados originais
    # Caso contrário, usar os dados como estão (compatibilidade com versão anterior)
    X_treino_original = X_treino
    X_teste_original = X_teste

    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Dados de Treinamento com Linha de Regressão (Escala Original)",
            "Dados de Teste com Linha de Regressão (Escala Original)",
            "Resíduos do Treinamento (Escala Original)",
            "Resíduos do Teste (Escala Original)"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Subplot 1: Treinamento (Escala Original)
    fig.add_trace(
        go.Scatter(
            x=X_treino_original.iloc[:, 0].values,
            y=y_treino,
            mode='markers',
            marker=dict(color='#3498db', size=6, opacity=0.6),
            name='Dados de Treinamento'
        ),
        row=1, col=1
    )
    
    # Linha de regressão para treinamento (Escala Original)
    if X_treino_log is not None:
        # Usar dados log-transformados para calcular a linha
        x_linha_original = np.array([X_treino_original.min().values[0], X_treino_original.max().values[0]])
        
        # --- CORREÇÃO: Aplicando a transformação logarítmica simétrica com segurança ---
        x_linha_log = np.sign(x_linha_original) * np.log1p(np.abs(x_linha_original))
        
        # Criando DataFrame temporário com o nome da feature para o StandardScaler não reclamar do Warning
        x_linha_df = pd.DataFrame(x_linha_log.reshape(-1, 1), columns=X_treino_log.columns)
        x_linha_escalado = scaler.transform(x_linha_df)
        
        y_linha_log = modelo.predict(x_linha_escalado)
        
        # Inversão simétrica para retornar à escala original com segurança
        y_linha = np.sign(y_linha_log) * np.expm1(np.abs(y_linha_log))
    else:
        x_linha_original = np.array([X_treino_original.min().values[0], X_treino_original.max().values[0]])
        x_linha_df = pd.DataFrame(x_linha_original.reshape(-1, 1), columns=X_treino.columns)
        x_linha_escalado = scaler.transform(x_linha_df)
        y_linha = modelo.predict(x_linha_escalado)
        
    fig.add_trace(
        go.Scatter(
            x=x_linha_original,
            y=y_linha,
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            name='Linha de Regressão'
        ),
        row=1, col=1
    )

    # Subplot 2: Teste (Escala Original)
    fig.add_trace(
        go.Scatter(
            x=X_teste_original.iloc[:, 0].values,
            y=y_teste,
            mode='markers',
            marker=dict(color='#2ecc71', size=6, opacity=0.6),
            name='Dados de Teste'
        ),
        row=1, col=2
    )
    
    # Linha de regressão para teste (Escala Original)
    fig.add_trace(
        go.Scatter(
            x=x_linha_original,
            y=y_linha,
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            name='Linha de Regressão'
        ),
        row=1, col=2
    )

    # Subplot 3: Resíduos Treinamento (Escala Original)
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

    # Subplot 4: Resíduos Teste (Escala Original)
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
    fig.update_xaxes(title_text="Valor Empenhado (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Valor Pago (R$)", row=1, col=1)
    
    fig.update_xaxes(title_text="Valor Empenhado (R$)", row=1, col=2)
    fig.update_yaxes(title_text="Valor Pago (R$)", row=1, col=2)
    
    fig.update_xaxes(title_text="Predições (R$)", row=2, col=1)
    fig.update_yaxes(title_text="Resíduos (R$)", row=2, col=1)
    
    fig.update_xaxes(title_text="Predições (R$)", row=2, col=2)
    fig.update_yaxes(title_text="Resíduos (R$)", row=2, col=2)

    fig.update_layout(
        title_text="Regressão Linear com Transformação Logarítmica: Valor Empenhado → Valor Pago",
        template="plotly_white",
        height=900,
        showlegend=True
    )

    print("Salvando gráfico...")
    fig.write_html("regressao_linear_graficos.html")
    #fig.show()

if __name__ == "__main__":
    df = prepara_df_para_regressao_linear()
    regressao_linear(df)