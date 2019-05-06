import numpy as np

class DefaultPositionUpdate():

    def update(self,particle,**kwargs):
        particle.position = particle.velocity + particle.position

class AverageVelocityBased():

    def __init__(self,c3):
        self.C3 = c3

    def update(self,particle,**kwargs):
        position = particle.position
        velocity = particle.velocity
        # mean_velocity = particle.mean_velocity if hasattr(particle,'mean_velocity') else self.particle_mean_velocity(particle)
        mean_velocity,mad_velocity = self.particle_mean_velocity(particle)

        c3 = self.C3
        r1 = np.random.uniform(0,1,size=len(position))

        componete = c3 * r1 * (position - mean_velocity)
        # component = np.multiply(c3,r1,np.subtract(position,mean_velocity,dtype=np.float64),dtype=np.float64)

        particle.position = particle.position + particle.velocity + componete
        # particle.position = np.add(position,velocity,component,dtype=np.float64)

    def particle_mean_velocity(self,particle):
        mean_velocity = np.mean(particle.velocity)
        mad_velocity = np.mean(np.absolute(particle.velocity - mean_velocity))

        return mean_velocity,mad_velocity