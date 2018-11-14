from bitarray import bitarray

QAM16 = {
    '-3.0-3.0j': bitarray('0000'),
    '-3.0-1.0j': bitarray('0001'),
    '-3.0+3.0j': bitarray('0010'),
    '-3.0+1.0j': bitarray('0011'),
    '-1.0-3.0j': bitarray('0100'),
    '-1.0-1.0j': bitarray('0101'),
    '-1.0+3.0j': bitarray('0110'),
    '-1.0+1.0j': bitarray('0111'),
    '3.0-3.0j': bitarray('1000'),
    '3.0-1.0j': bitarray('1001'),
    '3.0+3.0j': bitarray('1010'),
    '3.0+1.0j': bitarray('1011'),
    '1.0-3.0j': bitarray('1100'),
    '1.0-1.0j': bitarray('1101'),
    '1.0+3.0j': bitarray('1110'),
    '1.0+1.0j': bitarray('1111'),
}

N_CARRIERS = 400
N_FFT = 1024
EPS = 0.0001+0.0001j
