import numpy

def lambda_from_eV(energy):
    return numpy.power( numpy.power(6.626069934e-34,2)*numpy.power(3e8,2)/(energy*(1.6021e-19)*(2*(9.109e-31)*numpy.power(3e8,2)+energy*(1.6021e-19))  ) ,2)