import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import os

from pso import SearchSpace
from pso.util.benchmarks import dejong_sphere as function

if __name__ == "__main__":
    MAX_REPETITION = 50

    NRO_PARTICULAS = 50
    NRO_ITERACOES = 10000
    D_DIMENSOES = 30
    LIMITES = [-5.12,5.12]

    diretorio_imagens = "img"
    diretorio_dados = "data"

    plt.figure()

    ## PSO PADRÃO
    print("PSO Padrão")
    pso = SearchSpace(costFunction=function,
                    nroParticles=NRO_PARTICULAS,
                    maxIteration=NRO_ITERACOES,
                    dimensions=D_DIMENSOES,
                    bounds=LIMITES,
                    velocityStrategy="default",
                    positionStrategy="default"
                )

    i = 0
    DADOS_pso_gbest_padrao = list()
    while i < MAX_REPETITION:
        i += 1 # incrementar
        # Executar PSO
        X, Y, Z = 1,2,0.5
        pso.set_updateStrategiesParams(c1=X,c2=Y,w=Z)
        pso.setup()
        pso.initialize_particles()
        start = time.time()
        pso.run()
        end = time.time()
        print(f'Iteração: {i} - Tempo {(end-start)}s')

        # Coletar dados
        DADOS_pso_gbest_padrao.append(pso.gbest.fitness)

    ## PSO LINEAR
    print("PSO Linear")
    pso = SearchSpace(costFunction=function,
                    nroParticles=NRO_PARTICULAS,
                    maxIteration=NRO_ITERACOES,
                    dimensions=D_DIMENSOES,
                    bounds=LIMITES,
                    velocityStrategy="LINEAR",
                    positionStrategy="default"
                )

    i = 0
    DADOS_pso_gbest_linear = list()
    while i < MAX_REPETITION:
        i += 1 # incrementar
        # Executar PSO
        X, Y, MIN,MAX = 1,2,-1,1
        pso.set_updateStrategiesParams(c1=X,c2=Y,w_min=MIN,w_max=MAX)
        pso.setup()
        pso.initialize_particles()
        start = time.time()
        pso.run()
        end = time.time()
        print(f'Iteração: {i} - Tempo {(end-start)}s')

        # Coletar dados
        DADOS_pso_gbest_linear.append(pso.gbest.fitness)


    ## PSO CONSTRICTION
    print("PSO Constriction")
    pso = SearchSpace(costFunction=function,
                    nroParticles=NRO_PARTICULAS,
                    maxIteration=NRO_ITERACOES,
                    dimensions=D_DIMENSOES,
                    bounds=LIMITES,
                    velocityStrategy="CONSTRICTION",
                    positionStrategy="default"
                )

    i = 0
    DADOS_pso_gbest_constriction = list()
    while i < MAX_REPETITION:
        i += 1 # incrementar
        # Executar PSO
        X, Y, kappa = 2.05,2.05,1
        pso.set_updateStrategiesParams(c1=X,c2=Y,kappa=kappa)
        pso.setup()
        pso.initialize_particles()
        start = time.time()
        pso.run()
        end = time.time()
        print(f'Iteração: {i} - Tempo {(end-start)}s')

        # Coletar dados
        DADOS_pso_gbest_constriction.append(pso.gbest.fitness)

    ## PSO AVG_VELOCITY
    print("PSO Average Velocity")
    pso = SearchSpace(costFunction=function,
                    nroParticles=NRO_PARTICULAS,
                    maxIteration=NRO_ITERACOES,
                    dimensions=D_DIMENSOES,
                    bounds=LIMITES,
                    velocityStrategy="default",
                    positionStrategy="AVG_VELOCITY"
                )

    i = 0
    DADOS_pso_gbest_avgvelocity = list()
    while i < MAX_REPETITION:
        i += 1 # incrementar
        # Executar PSO
        X, Y, Z, K = 1,2,0.5,-0.4
        pso.set_updateStrategiesParams(c1=X,c2=Y,w=Z,c3=K)
        pso.setup()
        pso.initialize_particles()
        start = time.time()
        pso.run()
        end = time.time()
        print(f'Iteração: {i} - Tempo {(end-start)}s')

        # Coletar dados
        DADOS_pso_gbest_avgvelocity.append(pso.gbest.fitness)

    x = list(range(1,(MAX_REPETITION+1)))
    plt.plot(x,DADOS_pso_gbest_padrao) 
    plt.plot(x,DADOS_pso_gbest_linear)
    plt.plot(x,DADOS_pso_gbest_constriction)
    plt.plot(x,DADOS_pso_gbest_avgvelocity)
    plt.title("Comparação gbest simulações: Função f_3 De Jong")
    plt.xlabel("Nro. Simulação")
    plt.ylabel('Fitness gbest')
    plt.grid(True)

    legendas = ['PSO-padrao','PSO-linear','PSO-constriction','PSO-avg']
    plt.legend(legendas,loc='upper right')

    salvar_imagem_em = os.path.join(diretorio_imagens,f'comparativo_{function.__name__}_simulacao.png')
    plt.savefig(salvar_imagem_em)

    data ={
        'PSO-padrao' : DADOS_pso_gbest_padrao,
        'PSO-linear' : DADOS_pso_gbest_linear,
        'PSO-constriction' : DADOS_pso_gbest_constriction,
        'PSO-avg' : DADOS_pso_gbest_avgvelocity
    }
    dfData = pd.DataFrame(data=data)

    dfData.to_csv(os.path.join("data",f'comparativo_{function.__name__}_gbest.csv'))

