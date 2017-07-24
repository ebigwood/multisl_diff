import numpy
from numba import jit



@jit(nopython=True, parallel=True)              # this tag allows numba to optimize for parallel (multicore or GPU) computations automatically.
def lambda_from_eV(energy):
    # returns relativistic electron wavelength from kinetic energy in eV. see Subsection 1.
    return numpy.power( numpy.power(6.626069934e-34,2)*numpy.power(3e8,2)/(energy*(1.6021e-19)*(2*(9.109e-31)*numpy.power(3e8,2)+energy*(1.6021e-19))  ) ,2)
    




@jit(nopython=True, parallel=True)
def rotate_vec_array(array, tx, ty, tz):
    # iterates over each vector in the array n x 3 array and rotates around x, y, then z by tx, ty, and tz respectively
    for i in range(numpy.size(arr)[0]):
        array[i,:] = numpy.dot(rotation_mat(tx,ty,tz),array[i,:])               #multiplies by rotation matrices with thetas given


    @jit(nopython=True)
    def rotation_mat(tx,ty,tz):
        return numpy.dot(rotation_mat_z(tx), numpy.dot(rotation_mat_x(th)))


    @jit(nopython=True)
    def rotation_mat_x(th):
        return numpy.array( [(1,0,0),(0,numpy.cos(th),-numpy.sin(th)),(0,numpy.sin(th),numpy.cos(th))] )


    @jit(nopython=True)
    def rotation_mat_y(th):
        return numpy.array( [(numpy.cos(th),0,numpy.sin(th)),(0,1,0),(-numpy.sin(th),0,numpy.cos(th))] )


    @jit(nopython=True)
    def rotation_mat_z(th):
        return numpy.array( [(numpy.cos(th),-numpy.sin(th),0),(numpy.sin(th),numpy.cos(th),0),(0,0,1)] )
    
    
    
    

