import numpy as np

def dejong_sphere(variables):
        return np.sum(np.power(variables,2))

def quadratic_noise(variables):
    random_vector = np.random.uniform(0,1,size=len(variables))
    integer_vector = np.arange(1,len(variables) + 1)
    return np.sum(integer_vector * np.power(variables,4) + random_vector)

def rastrigin(variables):
    return np.sum((np.power(variables,2)) + (-10 * np.cos(2 * np.pi * variables)) + 10)

def rastrigin_function(variables):
    N = len(variables)
    return (10 * N) + np.sum((np.power(variables,2)) - (10 * np.cos(2 * np.pi * variables)))

def _rastring_transf_function(x):
    if np.abs(x) >= 0.5 :
        return np.divide(np.round(2 * x),2)
    else :
        return x

noncontinuous_transformation = np.vectorize(_rastring_transf_function)

def rastrigin_noncontinuous(variables):
    Y = noncontinuous_transformation(variables)
    return rastrigin(Y)

def griewank_function(variables):
    const = np.divide(1,4000)
    indices = np.arange(1,len(variables)+1,1,dtype=np.int16)

    somatorio = const * np.sum(np.power(variables,2))
    produtorio = np.prod(np.cos(np.divide(variables,np.sqrt(indices))))

    return 1 + somatorio - produtorio