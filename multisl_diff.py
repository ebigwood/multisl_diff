import numpy
from numba import jit

## physical constants
electron_m0 = 9.10956e-13
c = 3e8
J_per_eV =1.60217e-19
Au_param=4.0782e-10



## relativistic functions
@jit(nopython=True)              # this tag allows numba to optimize for parallel (multicore or GPU) computations automatically.
def lambda_from_eV(energy):
    # returns relativistic electron wavelength from kinetic energy in eV. see Subsection 1.
    return numpy.power( numpy.power(6.626069934e-34,2)*numpy.power(3e8,2)/(energy*(J_per_eV)*(2*(electron_m0)*numpy.power(c,2)+energy*(1.6021e-19))  ) ,2)
    
@jit(nopython=True)
def mass_from_eV(energy):
    # returns relativistic electron mass from kinetic energy in eV.
    return numpy.power( (numpy.power((energy)/(J_per_eV),2)+2*energy*(electron_m0)*numpy.power(c,2)*numpy.power(J_per_eV,-1))*numpy.power(c,-4) + numpy.power(electron_m0,2), 0.5 )




## geometric functions
@jit(nopython=True, parallel=False )              
def rotate_vec_array(array, tx, ty, tz):               #rotation functions
    # iterates over each vector in the array n x 3 array and rotates around x, y, then z by tx, ty, and tz respectively
    for i in range(numpy.shape(arr)[0]):
        array[i,:] = numpy.dot(rotation_mat(tx,ty,tz),array[i,:]-mean_vector_array(array))   + mean_vector_array(array)             #multiplies by rotation matrices with thetas given

@jit(nopython=True)
def rotation_mat(tx,ty,tz):
    return rotation_mat_z(tz) @ rotation_mat_y(ty) @ rotation_mat_x(tx)

@jit(nopython=True)
def rotation_mat_x(th):
    return numpy.array( [(1,0,0),(0,numpy.cos(th),-numpy.sin(th)),(0,numpy.sin(th),numpy.cos(th))] )

@jit(nopython=True)
def rotation_mat_y(th):
    return numpy.array( [(numpy.cos(th),0,numpy.sin(th)),(0,1,0),(-numpy.sin(th),0,numpy.cos(th))] )

@jit(nopython=True)
def rotation_mat_z(th):
    return numpy.array( [(numpy.cos(th),-numpy.sin(th),0),(numpy.sin(th),numpy.cos(th),0),(0,0,1)] )

#@jit(nopython=True)
def translate_vec_array(arr,trans):
    # translates an n x 3 array of vectors by an input vector trans
    for i in range(numpy.shape(arr)[0]):
        arr[i,:] = arr[i,:]+trans
    return arr

def mean_vector_array(arr):
    # returns the mean of all of the vectors in a n x 3 array of vectors arr
    mean = numpy.array([0,0,0])
    for i in range(numpy.shape(arr)[0]):
        mean = mean + arr[i,:]/numpy.shape(arr)[0]
    return mean

def rotate_tranlate_array(arr, tx,ty,tz, t):
    # returns the array first rotated and then translated.
    return translate_vec_array(rotate_vec_array(arr,tx,ty,tz),t)

def rand_vec_len(lent):
    # outputs a random vector of length lent
    r = numpy.random.rand(3)
    return ((r/numpy.power(r,2) @ numpy.ones(3))**0.5)*lent


## lattice functions
#@jit(nopython=True)
def lattice_populate_single(n,clen,latt_center):
    # returns a simple cubic lattice centered at latt_center (numpy.array) with side number n and lattice parameter clen
    latt = numpy.zeros([numpy.power(n,3),3])                # template lattice init
    for i in range(n):                          # template cubic populate
        for j in range(n):
            for k in range(n):
                latt[i*n**2+j*n+k,:]= numpy.array([(clen*(i-n/2),clen*(j-n/2),clen*(k-n/2))])+latt_center
    latt=latt[~(latt==0).all(1)]                #\template cubic populate
    return latt

#@jit(nopython=True)
def lattice_populate_fcc(clen,latt_center,radius):
    # returns a fcc nanoparticle with radius, lattice parameter clen, and centered at latt_center
    fudge = 2           # overestimation factor
    estimated_n = int(numpy.floor((2*radius)*fudge/clen))
    latt = numpy.zeros([numpy.power(estimated_n,3),3])                # template lattice init
    
    
    # fcc primitive vectors
    v0 = numpy.array([0,   0.5, 0.5])
    v1 = numpy.array([0.5, 0,   0.5])
    v2 = numpy.array([0.5, 0.5, 0  ])
    
    
    for i in range(estimated_n):                # template cubic populate
        for j in range(estimated_n):
            for k in range(estimated_n):
                latt[i*estimated_n**2+j*estimated_n+k,:]= (i-estimated_n/2)*clen*v0+(j-estimated_n/2)*clen*v1+(k-estimated_n/2)*clen*v2+latt_center
                                                #\template cubic populate
    
    
    for i in range(latt.shape[0]):              # template sphere prune
        if numpy.power(latt[i,:]-latt_center,2) @ numpy.ones(3) > radius**2:             # if point outside radius, mark and remove
            latt[i,:] = numpy.array([0,0,0])
    latt=latt[~(latt==0).all(1)]                #\template sphere prune
    
    return latt

def detector_populate(width, ndet, d_center):
    # returns a square detector with width, n^2 elements, and centered at d_center
    det = numpy.zeros([numpy.power(ndet,2),3])
    for i in range(ndet):                          # template detector populate
        for j in range(ndet):
            det[i*ndet+j,:]= width*numpy.array([(i-ndet/2)/ndet,(j-ndet/2)/ndet,0])+d_center
                                                #\template detector populate



## geometric diffraction functions
#@numba.vectorize('[float64(float64, float64, float64)','float64(float64, float64, float64)','float64(float64)'],target='cuda')             eventually vectorize this expression for speedup
#@jit(nopython=True)
def phase_point(latt,d_point,lam):
    # returns the geometric phase from latt at d_point with wavelength lambda
    return numpy.sum(  numpy.cos(  (2*numpy.pi/lam)*numpy.power(( numpy.power( latt @ numpy.array([1,0,0]).T - d_point @ numpy.array([1,0,0]),2) +numpy.power( latt @ numpy.array([0,1,0]).T - d_point @ numpy.array([0,1,0]),2) + numpy.power( latt @ numpy.array([0,0,1]).T - d_point @ numpy.array([0,0,1]),2)   ),.5)  )  )
    
    
    
