import numpy as np
from pso.particle import Particle
from pso.updateposition import DefaultPositionUpdate, AverageVelocityBased
from pso.updatevelocity import DefaultVelocityUpdate, LinearReduction, ConstrictionFactor

class SearchSpace():

    def __init__(self,
            costFunction,
            nroParticles,
            maxIteration,
            dimensions,
            bounds,
            **kwargs
        ):
        # assegurando o passagem do tipo de dado certo
        assert type(nroParticles) is int
        assert type(maxIteration) is int
        assert type(dimensions) is int
        assert callable(costFunction), "costFunction must be a function"
        assert type(bounds) is list

        # somente atribuição de variáveis passadas para o construtor
        self.costFunction = costFunction
        self.NroParticles = nroParticles
        self.MaxIteration = maxIteration
        self.dimensions = dimensions
        self.bounds = bounds

        self.velocityStrategyName = kwargs.get("velocityStrategy","default")
        self.positionStrategyName = kwargs.get("positionStrategy","default")
        
        # somente declaração de variáveis
        self.is_setup = False
        self.gbest = None
        self.particles = None
        self.velocityUpdate = None
        self.positionUpdate = None

    def __define_velocityUpdadeStrategy(self,strategy="default"):
        strategy = strategy.upper()
        
        if strategy == "CONSTRICTION":
            # print(f'Velocity Update Strategy: {strategy}',end='\r')
            return ConstrictionFactor(c1=self._C1,c2=self._C2,kappa=self._KAPPA)
        elif strategy == "LINEAR":
            # print(f'Velocity Update Strategy: {strategy}',end='\r')
            return LinearReduction(w_min=self._W_MIN,w_max=self._W_MAX,c1=self._C1,c2=self._C2,max_iteration=self.MaxIteration)
        else :
            # print(f'Velocity Update Strategy: DEFAULT',end='\r')
            return DefaultVelocityUpdate(c1=self._C1,c2=self._C2,w=self._W)

    def __define_positionUpdateStrategy(self,strategy="default"):
        strategy = strategy.upper()
        
        if strategy == "AVG_VELOCITY":
            # print(f'Position Update Strategy: {strategy}',end='\r')
            return AverageVelocityBased(c3=self._C3)
        else:
            # print(f'Position Update Strategy: DEFAULT',end='\r')
            return DefaultPositionUpdate()


    def setup(self):
        self.velocityUpdate = self.__define_velocityUpdadeStrategy(self.velocityStrategyName)
        self.positionUpdate = self.__define_positionUpdateStrategy(self.positionStrategyName)
        self.is_setup = True
    
    def set_updateStrategiesParams(self,**kwargs):
        if kwargs.get('c1'): 
            self._C1 = kwargs.get('c1')
        if kwargs.get('c2'):
            self._C2 = kwargs.get('c2')
        if kwargs.get('c3'):
             self._C3 = kwargs.get('c3')
        if kwargs.get('w'):
            self._W = kwargs.get('w')
        if kwargs.get('w_min'):
            self._W_MIN = kwargs.get('w_min')
        if kwargs.get('w_max'):
            self._W_MAX = kwargs.get('w_max')
        if kwargs.get('kappa'):
            self._KAPPA = kwargs.get('kappa')

        if self.is_setup :
            self.is_setup = False

    def initialize_particles(self):
        
        bounds = self.bounds
        del self.particles
        self.particles = list()
        self.gbest = None

        for _ in range(0,self.NroParticles):
            position = np.random.uniform(*bounds,size=self.dimensions)
            velocity = np.random.uniform(*bounds,size=self.dimensions)
            fitness = self.costFunction(position)

            particle = Particle(position,velocity,fitness=fitness)
            
            if self.gbest is None :
                self.gbest = Particle(position,velocity,fitness=fitness)
            elif fitness < self.gbest.fitness :
                self.gbest = Particle(position,velocity,fitness=fitness)

            self.particles.append(particle)

    def fitness(self,particle):
        score = self.costFunction(particle.position)
        # atribuir 'score' a 'particle'
        particle.fitness = score
        return score

    def run(self):
        if not self.is_setup:
            self.setup()

        iteration = 0
        while iteration < self.MaxIteration:
            # print(f'Iteration: {iteration}',end='\r')
            for p in self.particles:
                self.velocityUpdate.update(p,self.gbest.position,iteration=iteration)
                self.positionUpdate.update(p)
                # print(p,p.fitness,p.pbest_fitness)
                self.fitness(p)

            for p in self.particles:
                if p.fitness < p.pbest_fitness :
                    p.pbest_position = p.position
                    p.pbest_fitness = p.fitness
                    # print(f'Atualizando pbest: {p} -> {p.pbest_fitness}')

                    if p.fitness < self.gbest.fitness :
                        self.gbest = Particle(position=p.position,velocity=p.velocity,fitness=p.fitness)
                        # print(f'Atualizando gbest: {self.gbest} -> {self.gbest.fitness}')
        
            iteration += 1

        return self.gbest

    # def updatePosition(self,particle):
    #     particle.position =  np.add(particle.velocity,particle.position)

    # def updateVelocity(self,particle,gbest):
    #     position = particle.position
    #     velocity = particle.velocity
    #     pbest = particle.pbest_position

    #     r1 = np.random.uniform(0,1,size=len(position))
    #     r2 = np.random.uniform(0,1,size=len(position))

    #     w = self._W * np.ones(len(velocity))
    #     c1 = self._C1 * np.ones(len(velocity))
    #     c2 = self._C2 * np.ones(len(velocity))

    #     cognitive = c1 * r1 * (pbest - position)
    #     # cognitive = np.multiply(c1,r1,np.subtract(pbest,position),dtype=np.float64)
    #     social = c2 * r2 * (gbest - position)
    #     # social = np.multiply(c2,r2,np.subtract(gbest,position),dtype=np.float64)

    #     # assert any(social), "social não pode ser igual a zero"
    #     # assert all(cognitive), "cognitive não pode ser igual a zero"

    #     particle.velocity = (w * velocity) + cognitive + social
    #     # particle.velocity = np.add(np.multiply(w,velocity),cognitive,social,dtype=np.float64)