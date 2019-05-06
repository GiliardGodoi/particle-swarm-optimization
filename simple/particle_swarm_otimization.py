import random
import numpy as np

class Particle:

    def __init__(self,position,velocity):
        self.position = position
        self.velocity = velocity
        self.pbest_position = self.position
        self.best_error = -1
        self.error = -1

    def __str__(self):
        return f"({self.position}): pbest ({self.velocity})"

    def __repr__(self):
        return self.__str__()

    def update_velocity(self,gbest_position,w=0.5,c1=1,c2=2):
        for i in range(0, len(self.velocity)):
            r1 = random.random()
            r2 = random.random()

            cognitive_comp = c1 * r1 * (self.pbest_position[i] - self.position[i])
            social_comp = c2 * r2 * (gbest_position[i] - self.position[i])
            self.velocity[i] = (w * self.velocity[i]) + cognitive_comp + social_comp

    def update_position(self):
        self.position = [ p + v for p, v in zip(self.position, self.velocity)]

    def update_position_with_bounds(self,bounds):
        new_position = []
        for p, v in zip(self.position, self.velocity):
            t = p + v
            if t > bounds[1]:
                t = bounds[1]
            elif t < bounds[0]:
                t = bounds[0]
            
            new_position.append(t)
        
        self.position = new_position


# =======================================================

def function_sphere(variables):
    return sum([pow(x,2) for x in variables])

if __name__ == "__main__":
    
    error = 1.0e-05
    nro_dimensions = 20

    iteration = 0
    MAX_ITERATION = 1000
    NRO_PARTICLES = 50

    bounds = [-20,20]


    gbest = [ 5 for _ in range(0,nro_dimensions)]

    particles = []

    def random_population():

        for i in range(0, NRO_PARTICLES):
            position = [random.uniform(*bounds) for _ in range(0,nro_dimensions)]
            velocity = [0 for _ in range(0,nro_dimensions)]
            particles.append(Particle(position,velocity))

    def fitness(position):
        return function_sphere(position)

    def evaluate(particle):
        if fitness(particle) < error:
            return False
        else :
            return True


    random_population()

    while evaluate(gbest):

        for p in particles:
            if fitness(p.position) < fitness(p.pbest_position):
                p.pbest_position = p.position

                if fitness(p.position) < fitness(gbest):
                    gbest = p.position
        
        for p in particles:
            p.update_velocity(gbest)
            p.update_position_with_bounds(bounds)

        iteration += 1

    
    # for p in particles:
    #     print(p)

    for g in gbest:
        print(g)

    print('\n\n\n',fitness(gbest))
