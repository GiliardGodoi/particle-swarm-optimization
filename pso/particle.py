import numpy as np
import copy

class Particle(object):

    def __init__(self,position,velocity,**kwargs):
        assert len(position) == len(velocity), "Posição e velocidade devem ter a mesma dimensões (tamanho)"
        self.__velocity = np.array(velocity,dtype=np.float64)
        self.__position = np.array(position,dtype=np.float64)
        self.__fitness = kwargs.get("fitness",None)
        
        self.__pbest_position = np.array(self.position,dtype=np.float64)
        self.__pbest_fitness = copy.copy(self.__fitness)

    def __str__(self):
        return f'Particle: {self.__position}'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.__position)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self,value):
        assert len(value) == len(self.__position), f'dimensões necessária: {len(self.__position)}'
        if not type(value) is np.ndarray:
            value = np.array(value,dtype=np.float64,copy=True)
        self.__position = value

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self,value):
        assert len(value) == len(self.__velocity), f'dimensões necessária: {len(self.__velocity)}'
        if not type(value) is np.ndarray:
            value = np.array(value,dtype=np.float64,copy=True)
        self.__velocity = value

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self,value):
        if value < self.__pbest_fitness:
            pass
            # print("Update pbest")
        self.__fitness = value

    @property
    def pbest_position(self):
        return self.__pbest_position

    @pbest_position.setter
    def pbest_position(self,value):
        if type(value) is np.ndarray:
            self.__pbest_position = np.array(value,copy=True)
        elif type(value) is Particle:
            raise TypeError("Atribuindo Particle para pbest")

    @property
    def pbest_fitness(self):
        return self.__pbest_fitness

    @pbest_fitness.setter
    def pbest_fitness(self,value):
        if value <= self.__pbest_fitness :
            self.__pbest_fitness = copy.copy(value)
        else:
            raise TypeError("value needs to be smaller than current pbest_fitness")

    def checkPBest(self):
        return self.__pbest_fitness < self.__fitness

    def updatePBest(self):
        self.__pbest_position = np.array(self.__position)
        self.__pbest_fitness = copy.copy(self.__fitness)

        return True