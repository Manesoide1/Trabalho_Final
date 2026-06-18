from datetime import datetime
import mysql.connector
import pandas as pd
import numpy as np
import time
import controla_bd
import plotly.graph_objects as go

# IMPORTAÇÕES NECESSÁRIAS PARA O CÁLCULO DE DISTÂNCIA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def gera_grafico_dispersão(ano):
    conexao, cursor = controla_bd.conecta_banco_de_dados()
    controla_bd.cria_banco_e_tabelas(cursor)

    # Pegando os dados do ano específico
    query = f"SELECT * FROM despesas_governo_federal WHERE ano = {ano}"
    
    print(f"\nBuscando dados de {ano} no MySQL...")
    df = pd.read_sql(query, conexao)
    print(f"Dados carregados! Total de registros em {ano}: {len(df)}")

    cursor.close()
    conexao.close()

    if df.empty:
        print(f"Atenção: A tabela para o ano {ano} está vazia.")
        return

    print("Limpando dados e calculando distâncias geométricas...")
    colunas_gasto = ['valor_empenhado', 'valor_liquidado', 'valor_pago']
    
    df = df.dropna(subset=colunas_gasto)
    #df = df[(df['valor_empenhado'] > 0) & (df['valor_liquidado'] > 0) & (df['valor_pago'] > 0)]
    
    if df.empty:
        print(f"Atenção: Após a limpeza, não restaram dados válidos para o ano {ano}.")
        return

    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(df[colunas_gasto])
    
    n_vizinhos = min(5, len(df))
    modelo_knn = NearestNeighbors(n_neighbors=n_vizinhos)
    modelo_knn.fit(dados_normalizados)
    
    distancias, _ = modelo_knn.kneighbors(dados_normalizados)
    df['distancia_media'] = np.mean(distancias, axis=1)
    df = df.dropna(subset=['distancia_media'])

    print("Classificando os pontos em duas categorias...")
    
    # DEFINE SEU VALOR X DE CORTE AQUI
    limiar_anomalia_critica = 10
    
    df_normal = df[df['distancia_media'] <= limiar_anomalia_critica]
    df_critico = df[df['distancia_media'] > limiar_anomalia_critica]

    fig = go.Figure()

    # 1. Camada dos Dados Regulares: Cor fixa (Verde Esmeralda ou Cinza para dar contraste)
    fig.add_trace(go.Scatter3d(
        x=df_normal['valor_empenhado'],
        y=df_normal['valor_liquidado'],
        z=df_normal['valor_pago'],
        mode='markers',
        marker=dict(
            size=3,
            color='#2ecc71', # <--- Cor verde sólida para a grande massa comum
            opacity=0.5      # Opacidade um pouco menor ajuda a ver o acúmulo de pontos
        ),
        name='Dados Regulares'
    ))

    # 2. Camada das Anomalias Críticas: Cor fixa (Vermelho Vivo)
    if not df_critico.empty:
        fig.add_trace(go.Scatter3d(
            x=df_critico['valor_empenhado'],
            y=df_critico['valor_liquidado'],
            z=df_critico['valor_pago'],
            mode='markers',
            marker=dict(
                size=5,       # Pontos maiores para destacar bem onde estão os erros
                color='#ff0000', # <--- Vermelho Vivo sólido
                opacity=1.0   # Totalmente opaco para destacar na tela
            ),
            name=f'Anomalia Crítica (>{limiar_anomalia_critica})'
        ))

    # --- 3. FORÇANDO AS LINHAS DE MEDIÇÃO E AJUSTES DE LAYOUT ---
    fig.update_layout(
        template="plotly_white", 
        title=dict(text=f"Detecção de Anomalias nos Gastos Públicos - Ano {ano}"),
        margin=dict(l=0, r=0, b=0, t=40), 
        showlegend=True, # Mantém a legenda no canto indicando o que é cada cor
        scene=dict(
            xaxis=dict(title='Empenhado (X)', showgrid=True, gridcolor="lightgray", showspikes=True, spikecolor="black"),
            yaxis=dict(title='Liquidado (Y)', showgrid=True, gridcolor="lightgray", showspikes=True, spikecolor="black"),
            zaxis=dict(title='Pago (Z)', showgrid=True, gridcolor="lightgray", showspikes=True, spikecolor="black")
        )
    )

    print(f"Renderizando gráfico bicolor de {ano} no navegador...")
    fig.show()

def teste_grafico_dispersao():
    ano_atual = datetime.now().year
    for ano in range(2014, ano_atual + 1):
        gera_grafico_dispersão(ano)
        
        if ano < ano_atual:
            input(f"\n[Terminado {ano}] Pressione 'ENTER' no terminal para calcular e abrir o ano de {ano + 1}...")
        else:
            print("\nTodos os anos foram renderizados!")

def prepara_df_para_KNN():
    conexao, cursor = controla_bd.conecta_banco_de_dados()
    controla_bd.cria_banco_e_tabelas(cursor)

    df_final = pd.DataFrame()

    for codigo_orgao in pd.read_sql("SELECT codigo_orgao_superior FROM despesas_governo_federal", conexao)['codigo_orgao_superior'].unique():
        # Pegando os dados do orgao específico
        query = f"SELECT * FROM despesas_governo_federal WHERE codigo_orgao_superior = {codigo_orgao}"
        
        print(f"\nBuscando dados de no MySQL...")
        df = pd.read_sql(query, conexao)
        print(f"Dados carregados! Total de registros: {len(df)}")

        df = df.sample(n=max(int(df.shape[0] * 0.05), 1), random_state=42) # Reduzindo a amostra para 10% para acelerar o processo

        print(f"Dados selecionados! Total de registros: {len(df)}")

        if df.empty:
            print(f"Atenção: A tabela está vazia.")
            return

        print("Limpando dados e calculando distâncias geométricas...")
        colunas_gasto = ['valor_empenhado', 'valor_liquidado', 'valor_pago', 'codigo_orgao_superior']
        
        df = df.dropna(subset=colunas_gasto)
        
        if df.empty:
            print(f"Atenção: Após a limpeza, não restaram dados válidos.")
            return
        
        media_empenhado_liquidado = (df['valor_empenhado'] - df['valor_liquidado']).mean()
        media_liquidado_pago = (df['valor_liquidado'] - df['valor_pago']).mean()

        desvio_padrao_empenhado_liquidado = (df['valor_empenhado'] - df['valor_liquidado']).std()
        desvio_padrao_liquidado_pago = (df['valor_liquidado'] - df['valor_pago']).std()

        print(f'media_empenhado_liquidado = {media_empenhado_liquidado}')
        print(f'media_liquidado_pago = {media_liquidado_pago}')
        print(f'desvio_padrao_empenhado_liquidado = {desvio_padrao_empenhado_liquidado}')
        print(f'desvio_padrao_liquidado_pago = {desvio_padrao_liquidado_pago}')

        # --- ESTRATÉGIA ROBUSTA (MEDIANA E MAD) PARA EVITAR OVERFITTING ---
        # 1. Calcula as diferenças reais de fluxo
        dif_emp_liq = df['valor_empenhado'] - df['valor_liquidado']
        dif_liq_pag = df['valor_liquidado'] - df['valor_pago']

        # 2. Encontra as medianas (muito mais realistas que as médias infladas)
        mediana_emp_liq = dif_emp_liq.median()
        mediana_liq_pag = dif_liq_pag.median()

        # 3. Calcula o MAD (Median Absolute Deviation) para cada par
        mad_emp_liq = (dif_emp_liq - mediana_emp_liq).abs().median()
        mad_liq_pag = (dif_liq_pag - mediana_liq_pag).abs().median()

        # 4. Transforma o MAD em um "Desvio Padrão Robusto" (Fator estatístico padrão: 1.4826)
        # E multiplicamos por 3 para aplicar a regra dos 3 sigmas (Rigor Acadêmico)
        desvio_robusto_emp_liq = 1.4826 * mad_emp_liq
        desvio_robusto_liq_pag = 1.4826 * mad_liq_pag
        
        multiplicador_sigma = 3  # Regra dos 3 desvios padrões para outliers reais

        # 5. Filtro de Relevância Material (Evita considerar variações pequenas/centavos como anomalia)
        limite_financeiro_minimo = 5000  # R$ 5.000,00

        df['anomalia'] = np.where(
            
            # 1. Anomalia Dimensional Empenhado vs Liquidado (Usando Estatística Robusta + Relevância Material)
            #((abs(dif_emp_liq - mediana_emp_liq) > desvio_robusto_emp_liq * multiplicador_sigma) & 
            #(abs(dif_emp_liq) > limite_financeiro_minimo)) |
            
            # 2. Anomalia Dimensional Liquidado vs Pago (Usando Estatística Robusta + Relevância Material)
            #((abs(dif_liq_pag - mediana_liq_pag) > desvio_robusto_liq_pag * multiplicador_sigma) & 
            #(abs(dif_liq_pag) > limite_financeiro_minimo)) |
            
            # 3. Valor negativo enquanto o anterior é positivo (Regras lógicas mantidas)
            #((df['valor_pago'] < 0) & (df['valor_liquidado'] >= 0)) |
            #((df['valor_liquidado'] < 0) & (df['valor_empenhado'] >= 0)) |
            
            # 4. Valor negativo enquanto os demais são 0
            #((df['valor_empenhado'] < 0) & (df['valor_liquidado'] == 0) & (df['valor_pago'] == 0)) |
            #((df['valor_liquidado'] < 0) & (df['valor_empenhado'] == 0) & (df['valor_pago'] == 0)) |
            #((df['valor_pago'] < 0) & (df['valor_liquidado'] == 0) & (df['valor_empenhado'] == 0))

            (df['valor_empenhado'] < df['valor_liquidado']) |
            (df['valor_liquidado'] < df['valor_pago']) |
            (df['valor_empenhado'] < df['valor_pago'])
            ,
            1,
            0
        )


        df_final = pd.concat([df_final, df], ignore_index=True)
    
    #=========================================================================

    print(f"Classificação concluída! Total de anomalias encontradas: {df_final['anomalia'].sum()}")
    print(f"Total de não-anomalias encontradas: {len(df_final['anomalia']) - df_final['anomalia'].sum()}")

    cursor.close()
    conexao.close()
    
    # Agora você pode retornar o DataFrame modificado para usar no resto do seu projeto
    return df_final

def KNN(df):
    X = df[['valor_empenhado', 'valor_liquidado', 'valor_pago']]
    Y = df['anomalia']

    X_treino, X_teste, Y_treino, Y_teste = train_test_split(
        X, Y,
        test_size=0.30,
        #random_state=42,
        stratify=Y
    )

    scaler = StandardScaler()

    X_treino_escalado = scaler.fit_transform(X_treino)
    X_teste_escalado = scaler.transform(X_teste)

    melhor_acuracia = 0
    melhor_k = None
    melhor_peso = None

    print("Iniciando a busca pelo melhor modelo (Validação Cruzada cv=5)...")
    print("-" * 60)

    # Iterando apenas por valores ímpares de k (de 1 a 15) e os tipos de pesos
    for k in range(1, 16, 2):
        for peso in ['uniform', 'distance']:
            
            # Inicializa o classificador KNN para este teste
            knn = KNeighborsClassifier(n_neighbors=k, weights=peso)
            
            # Executa a validação cruzada em 5 partes usando APENAS o conjunto de treino
            scores = cross_val_score(knn, X_treino_escalado, Y_treino, cv=5, scoring='accuracy')
            acuracia_media = np.mean(scores)
            
            print(f"Configuração: k={k:2d} | Pesos: {peso:8s} -> Acurácia Média: {acuracia_media:.4f}")
            
            # Salva a melhor configuração encontrada pela validação cruzada
            if acuracia_media > melhor_acuracia:
                melhor_acuracia = acuracia_media
                melhor_k = k
                melhor_peso = peso

    print("-" * 60)
    print(f"-> Melhor configuração na Validação Cruzada: k={melhor_k} com weights='{melhor_peso}' (Acurácia: {melhor_acuracia:.4f})")

    modelo_final = KNeighborsClassifier(n_neighbors=melhor_k, weights=melhor_peso)
    modelo_final.fit(X_treino_escalado, Y_treino)

    # 2. Faz as previsões usando o conjunto de teste (dados que o modelo nunca viu)
    previsoes = modelo_final.predict(X_teste_escalado)

    # 3. Calcula a acurácia final real
    acuracia_final = accuracy_score(Y_teste, previsoes)

    print("\n================ AVALIAÇÃO DO MODELO FINAL ================")
    print(f"Acurácia real alcançada no conjunto de teste: {acuracia_final:.4f}")
    print("===========================================================")

    gera_fronteira_decisao_2d(df, k=melhor_k, peso=melhor_peso)

def gera_fronteira_decisao_2d(df, k=3, peso='distance'):
    print("\nPreparando dados para o gráfico de fronteira 2D...")
    
    # 1. Selecionamos apenas 2 dimensões para viabilizar o gráfico de área
    # Vamos usar Empenhado e Liquidado como eixos X e Y
    X = df[['valor_empenhado', 'valor_liquidado']].values
    Y = df['anomalia'].values

    # 2. Padronização (Essencial para o KNN)
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    # 3. Treina o modelo KNN especificamente para essas duas dimensões do gráfico
    knn_2d = KNeighborsClassifier(n_neighbors=k, weights=peso)
    knn_2d.fit(X_escalado, Y)

    # 4. Criando a grade (Meshgrid) que vai cobrir o fundo do gráfico
    # Define os limites do gráfico com uma folga
    x_min, x_max = X_escalado[:, 0].min() - 0.5, X_escalado[:, 0].max() + 0.5
    y_min, y_max = X_escalado[:, 1].min() - 0.5, X_escalado[:, 1].max() + 0.5
    
    # Definimos uma resolução fixa (ex: 200x200 pontos = 40.000 pontos no total)
    # Isso gera curvas bonitas e consome pouquíssima memória
    resolucao = 200 
    
    eixo_x = np.linspace(x_min, x_max, resolucao)
    eixo_y = np.linspace(y_min, y_max, resolucao)
    xx, yy = np.meshgrid(eixo_x, eixo_y)

    # 5. Prediz a classe de cada ponto da grade de fundo
    pontos_grade = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_2d.predict(pontos_grade)
    Z = Z.reshape(xx.shape)

    # Voltando os valores da grade para a escala em Reais (para os eixos ficarem corretos)
    # Criamos um array temporário com o formato original do scaler para desfazermos a escala
    grade_original = scaler.inverse_transform(pontos_grade)
    xx_original = grade_original[:, 0].reshape(xx.shape)
    yy_original = grade_original[:, 1].reshape(xx.shape)

    # Desfazendo a escala dos pontos reais para plotagem perfeita
    X_original = scaler.inverse_transform(X_escalado)

    # 6. Construindo o Gráfico com Plotly
    fig = go.Figure()

    # Camada 1: A Área de Decisão (Fundo Colorido)
    fig.add_trace(go.Contour(
        x=xx_original[0, :],
        y=yy_original[:, 0],
        z=Z,
        showscale=False,
        opacity=0.3,
        colorscale=[[0, '#2ecc71'], [1, '#ff0000']], # Verde para 0 (Regular), Vermelho para 1 (Anomalia)
        line=dict(width=0), # Remove as linhas de contorno internas
        hoverinfo='skip'
    ))

    # Camada 2: Seus dados reais (Pontos Regulares)
    df_normal = X_original[Y == 0]
    fig.add_trace(go.Scatter(
        x=df_normal[:, 0],
        y=df_normal[:, 1],
        mode='markers',
        marker=dict(size=4, color='#27ae60'),
        name='Dados Regulares'
    ))

    # Camada 3: Seus dados reais (Anomalias Críticas)
    df_critico = X_original[Y == 1]
    if len(df_critico) > 0:
        fig.add_trace(go.Scatter(
            x=df_critico[:, 0],
            y=df_critico[:, 1],
            mode='markers',
            marker=dict(size=6, color='#c0392b', symbol='x'),
            name='Anomalias Reais'
        ))

    # Ajustes de Layout
    fig.update_layout(
        template="plotly_white",
        title=f"Fronteira de Decisão KNN (k={k}, weights='{peso}')",
        xaxis=dict(title='Valor Empenhado (R$)', showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title='Valor Liquidado (R$)', showgrid=True, gridcolor="lightgray"),
        showlegend=True
    )

    print("Renderizando fronteira de decisão 2D no navegador...")
    #fig.show()
    fig.write_html("grafico_interativo.html")

KNN(prepara_df_para_KNN())