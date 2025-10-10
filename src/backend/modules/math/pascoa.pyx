from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)

def fibonacci_serie(int number=10,bint single=False):
    cdef unsigned long long a = 0
    cdef unsigned long long b = 1
    cdef unsigned char result
    cdef unsigned int i 
    cdef list lista = []

    if number < 0:
        msg = f"The number entered is negative, invalid number {number !r}"
        raise Exception(msg)

    if number == 0:

        return None

    if not single:

        for i in range(number):

            lista.append(a)

            a, b = a + b, a

        return lista

    for i in range(number):

        result = "{:.2e}".format(int(a))

        a, b = a + b, a

    return result