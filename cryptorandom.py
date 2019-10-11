# Author: IndependantPenguin
# Date-created: 11-10-2019
# Date-last-modified: 12-10-2019
# Version: 1.0

# Usage:
# cryptorandom.randint(0, 100)
# >>>53

from os import urandom as _urandom
try:
    from math import log as _log, exp as _exp, pi as _pi, e as _e, acos as _acos, cos as _cos, sin as _sin
except:
    _log = lambda x: 0
    _exp = lambda x: 0
    _pi = lambda x: 0
    _e = lambda x: 0
    _acos = lambda x: 0
    _cos = lambda x: 0
    _sin = lambda x: 0

class cryptorandom:
    def randrange(start, stop=None, step=1):
        """Choose a random item from range(start, stop[, step]).

        This fixes the problem with randint() which includes the
        endpoint; in Python this is usually not what you want.

        """
        if stop == None:
            stop = start
            start = 0
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)
        
        length = stop-start
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)
        
        random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
        random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        
        while (random_int+start)>stop or (random_int+start)%step!=0:
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        return random_int + start

    def randint(a, b):
        """Return random integer in range [a, b], including both end points.
        """
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)
        
        length = b-a
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)
        
        random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
        try:
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        except ValueError as e:
            print(e, random_int, binary_length, length)
        
        while (random_int+a)>b:
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        return random_int + a


    def choice(seq):
        """Choose a random element from a non-empty sequence."""
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)
        
        length = len(seq)-1
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)
        
        random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
        random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        
        while (random_int)>length:
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
        return seq[random_int]

    def choices(population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population elements chosen with replacement.

        If the relative weights or cumulative weights are not specified,
        the selections are made with equal probability.

        """
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)

        if weights == None and cum_weights == None:
            pop = list(population)
        elif cum_weights == None:
            pop = []
            for i in range(len(weights)):
                for n in range(weights[i]):
                    pop.append(population[i])
        else:
            pop = []
            cw = 0
            for i in range(len(weights)):
                cw += weights[i]
                for n in range(cw):
                    pop.append(population[i])

        length = len(pop)-1
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)
        output = []
        while len(output) < k:
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            
            while (random_int)>length:
                random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
                random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            if random_int not in output:
                output.append(pop[random_int])
        return output

    def shuffle(x, random=None):
        """Shuffle list x in place, and return None.

        Optional argument random is disabled.

        """
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)

        
        length = len(x)-1
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)

        for i in range(len(x)):
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            
            while (random_int)>length:
                random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
                random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            x.insert(random_int, x.pop(i))
        return None
            

    def sample(population, k):
        """Chooses k unique random elements from a population sequence or set.

        Returns a new list containing elements from the population while
        leaving the original population unchanged.  The resulting list is
        in selection order so that all sub-slices will also be valid random
        samples.  This allows raffle winners (the sample) to be partitioned
        into grand prize and second place winners (the subslices).

        Members of the population need not be hashable or unique.  If the
        population contains repeats, then each occurrence is a possible
        selection in the sample.

        To choose a sample in a range of integers, use range as an argument.
        This is especially fast and space efficient for sampling from a
        large population:   sample(range(10000000), 60)
        """
        ceil = lambda n: int(n if (n == n//1) else (n//1)+1)
        bytebin = lambda i: '0b'+'0'*( 8*ceil((len(bin(i))-2)/8) - (len(bin(i))-2) )  +  bin(i)[2:]
        positive = lambda v: (0 if abs(v)!=v else v)

        
        length = len(population)-1
        binary_length = len(bin(length))

        byte_count = ceil((binary_length-2)/8)

        output = []
        while len(output) < k:
            random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
            random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            
            while (random_int)>length:
                random_int = int.from_bytes(_urandom(byte_count), byteorder='big', signed=False)
                random_int = random_int >> positive(len(bytebin(random_int))-binary_length)
            if random_int not in output:
                output.append(population[random_int])
        return output

    def uniform(a, b):
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        
        random_float = float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3)) # 40% faster than dividing by 10**16
        return a + (b-a) * random_float

    def triangular(low=0.0, high=1.0, mode=None):
        """Triangular distribution.

        Continuous distribution bounded by given lower and upper limits,
        and having a given mode value in-between.

        http://en.wikipedia.org/wiki/Triangular_distribution

        """
        sqrt = lambda s: s**0.5
        try:
            c = 0.5 if mode is None else (mode - low) / (high - low)
        except ZeroDivisionError:
            return low

        random_float = float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3)) # 40% faster than dividing by 10**16
        
        if random_float > c:
            random_float = 1.0 - random_float
            c = 1.0 - c
            low = high
            high = low
        return low + (high - low) * sqrt(random_float * c)

    def normalvariate(mu, sigma):
        """Normal distribution.

        mu is the mean, and sigma is the standard deviation.

        """
        # mu = mean, sigma = standard deviation

        # Uses Kinderman and Monahan method. Reference: Kinderman,
        # A.J. and Monahan, J.F., "Computer generation of random
        # variables using the ratio of uniform deviates", ACM Trans
        # Math Software, 3, (1977), pp257-260.

        while True:
            u1 = float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3))
            u2 = 1.0 - float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3))
            z = (4 * _exp(-0.5)/(2.0**0.5))*(u1-0.5)/u2
            zz = z*z/4.0
            if zz <= -_log(u2):
                break
        return mu + z*sigma

    def lognormvariate(mu, sigma):
        """Log normal distribution.

        If you take the natural logarithm of this distribution, you'll get a
        normal distribution with mean mu and standard deviation sigma.
        mu can have any value, and sigma must be greater than zero.

        """
        while True:
            u1 = float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3))
            u2 = 1.0 - float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3))
            z = (4 * _exp(-0.5)/(2.0**0.5))*(u1-0.5)/u2
            zz = z*z/4.0
            if zz <= -_log(u2):
                break
        return _exp(nmu + z*sigma)

    def expovariate(self, lambd):
        """Exponential distribution.

        lambd is 1.0 divided by the desired mean.  It should be
        nonzero.  (The parameter would be called "lambda", but that is
        a reserved word in Python.)  Returned values range from 0 to
        positive infinity if lambd is positive, and from negative
        infinity to 0 if lambd is negative.

        """
        # lambd: rate lambd = 1/mean
        # ('lambda' is a Python reserved word)

        # we use 1-random() instead of random() to preclude the
        # possibility of taking the log of zero.
        return -_log(1.0 - float('0.'+str(int.from_bytes(_urandom(7), byteorder='big', signed=False) >> 3)))/lambd
    

if __name__ == "__main__":
    import random, time
    print ("Benchmarking ...")
    
    a = time.process_time()
    y = [random.randint(0, 1000000) for x in range(1000000)]
    a = time.process_time() - a
    b = time.process_time()
    y = [cryptorandom.randint(0, 1000000) for x in range(1000000)]
    b = time.process_time() - b
    print ("\n random.randint: %fs" % a)
    print (" cryptrandom.randint: %fs" % b)

    # Red is faulty
    
    c = time.process_time()
    y = [random.randrange(0, 2000000, 2) for x in range(500000)]
    c = time.process_time() - c
    d = time.process_time()
    y = [cryptorandom.randrange(0, 2000000, 2) for x in range(500000)]
    d = time.process_time() - d
    print ("\n random.randrange: %fs" % c)
    print (" cryptrandom.randrange: %fs" % d)
    
    e = time.process_time()
    y = [random.choice(list(range(10000))) for x in range(20000)]
    e = time.process_time() - e
    f = time.process_time()
    y = [cryptorandom.choice(list(range(10000))) for x in range(20000)]
    f = time.process_time() - f
    print ("\n random.choice: %fs" % e)
    print (" cryptrandom.choice: %fs" % f)
    
    g = time.process_time()
    m = list(range(10000))
    y = [random.shuffle(m) for x in range(50)]
    g = time.process_time() - g
    h = time.process_time()
    m = list(range(10000))
    y = [cryptorandom.shuffle(m) for x in range(50)]
    h = time.process_time() - h
    print ("\n random.shuffle: %fs" % g)
    print (" cryptrandom.shuffle: %fs" % h)

    i = time.process_time()
    y = [random.sample(range(1000000), 100) for x in range(10000)]
    i = time.process_time() - i
    j = time.process_time()
    y = [cryptorandom.sample(range(1000000), 100) for x in range(10000)]
    j = time.process_time() - j
    print ("\n random.sample: %fs" % i)
    print (" cryptrandom.sample: %fs" % j)

    p = (100-(25*((a/b)+(c/d)+(e/f)+(g/h)+(i/j))))
    print("\nSpeed Penalty: %s%%" % round(p,2))
    
