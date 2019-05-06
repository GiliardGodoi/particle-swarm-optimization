import numpy as np
from pso import SearchSpace
# from pso.util.benchmarks import dejong_sphere as costFitness
# from pso.util.benchmarks import quadratic_noise as costFitness
from pso.util.benchmarks import rastrigin as costFitness
# from pso.util.benchmarks import griewank_function as costFitness
# import time

if __name__ == "__main__":
    
    pso = SearchSpace(costFunction=costFitness,
                        nroParticles=50,
                        maxIteration=10000,
                        dimensions=3,
                        bounds=[-5.12,5.12],
                        velocityStrategy="CONSTRICTION",
                        positionStrategy="default"
                    )

    # pso.set_updateStrategiesParams(c1=1,c2=2,w=0.5)
    # X, Y, MIN,MAX = 1,2,-1,1
    print(f'Funtion test: {costFitness.__name__}')
    # print("Parâmetros:", X, Y, MIN, MAX)
    # pso.set_updateStrategiesParams(c1=X,c2=Y,w_min=MIN,w_max=MAX)
    X, Y, kappa = 2.05,2.05,1
    pso.set_updateStrategiesParams(c1=X,c2=Y,kappa=kappa)
    np.random.seed()
    pso.initialize_particles()
    
    print(pso.gbest,pso.gbest.fitness)

    pso.run()

    print(pso.gbest,pso.gbest.fitness)


