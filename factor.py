import time
import math
import random
import utils

small_primes = utils.prime_sieve(25000)

def compute_bounds(n):
	log_n = len(str(n))
	if log_n <= 30: 
		B1, B2 = 2000, 147396
	elif log_n <= 40: 
		B1, B2 = 11000, 1873422
	elif log_n <= 50: 
		B1, B2 = 50000, 12746592
	elif log_n <= 60: 
		B1, B2 = 250000, 128992510
	elif log_n <= 70: 
		B1, B2 = 1000000, 1045563762
	elif log_n <= 80:
		B1, B2 = 3000000, 5706890290
	else: 
		B1, B2 = 43 * 10**7, 2 * 10**10
	return B1, B2


def point_add(px, pz, qx, qz, rx, rz, n):
	u = (px-pz) * (qx+qz)
	v = (px+pz) * (qx-qz)
	upv, umv = u+v, u-v
	x = (rz * upv * upv)
	if x >= n:
		x %= n
	z = rx * umv * umv
	if z >= n:
		z %= n
	return x, z


def point_double(px, pz, n, a24):
	u, v = px+pz, px-pz
	u2, v2 = u*u, v*v
	t = u2 - v2
	x = (u2 * v2) 
	if x >= n:
		x %= n
	z = (t * (v2 + a24*t))
	if z >= n:
		z %= n
	return x, z


def scalar_multiply(k, px, pz, n, a24):
	sk = bin(k)
	lk = len(sk)
	qx, qz = px, pz
	rx, rz = point_double(px, pz, n, a24)

	for i in range(3, lk):
		if sk[i] == '1':
			qx, qz = point_add(rx, rz, qx, qz, px, pz, n)
			rx, rz = point_double(rx, rz, n, a24)
		else:
			rx, rz = point_add(qx, qz, rx, rz, px, pz, n)
			qx, qz = point_double(qx, qz, n, a24)	

	return qx, qz

def factorize_ecm(n, verbose = False):
	if n == 1 or utils.is_prime(n):
		return n
        
	B1, B2 = compute_bounds(n)

	D = int(math.sqrt(B2))
	beta = [0] * (D+1)
	S = [0] * (2*D + 2)
	curves, log_B1 = 0, math.log(B1)
	primes = utils.prime_sieve(B2)
	num_primes = len(primes)
	idx_B1 = utils.binary_search(B1, primes)
	
	k = 1
	for i in range(idx_B1):
		p = primes[i]
		k = k * pow(p, int(log_B1/math.log(p)))

	g = 1
	while (g == 1 or g == n) and curves <= 10000:
		curves += 1
		sigma = random.randint(6, 2**63)
		u = ((sigma * sigma) - 5) % n
		v = (4 * sigma) % n
		vmu = v - u
		A = ((vmu*vmu*vmu) * (3*u + v) // (4*u*u*u*v) - 2) % n
		a24 = (A+2) // 4

		px, pz = ((u*u*u) // (v*v*v)) % n, 1
		qx, qz = scalar_multiply(k, px, pz, n, a24)
		g = utils.gcd(n, qz)

		if g != 1 and g != n:
			return g

		S[1], S[2] = point_double(qx, qz, n, a24)
		S[3], S[4] = point_double(S[1], S[2], n, a24)
		beta[1] = (S[1] * S[2]) % n
		beta[2] = (S[3] * S[4]) % n
		for d in range(3, D+1):
			d2 = 2 * d
			S[d2-1], S[d2] = point_add(S[d2-3], S[d2-2], S[1], S[2], S[d2-5], S[d2-4], n)
			beta[d] = (S[d2-1] * S[d2]) % n

		g, B = 1, B1 - 1

		rx, rz = scalar_multiply(B, qx, qz, n, a24)
		tx, tz = scalar_multiply(B - 2*D, qx, qz, n, a24)
		q, step = idx_B1, 2*D

		for r in range(B, B2, step):
			alpha, limit = (rx * rz) % n, r + step
			while q < num_primes and primes[q] <= limit:
				d = (primes[q] - r) // 2
				f = (rx - S[2*d-1]) * (rz + S[2*d]) - alpha + beta[d]
				g = (g * f) % n
				q += 1
			trx, trz = rx, rz
			rx, rz = point_add(rx, rz, S[2*D-1], S[2*D], tx, tz, n)
			tx, tz = trx, trz

		g = utils.gcd(n, g)

	if curves > 10000:
		return -1
	else:
		return g

def factorize_rho(n, verbose = False):
    if n == 1 or utils.is_prime(n):
        return n

    for i in range(len(small_primes) - 1, -1, -1):
        r, c, y = 1, small_primes[i], random.randint(1, n-1)
        m, g, q, ys = random.randint(1, n-1), 1, 1, y
        min_val, k = 0, 0
        while g == 1:
            x, k = y, 0
            for j in range(r):
                y = y*y + c
                if y > n: y %= n
            while k < r and g == 1:
                ys, min_val = y, min(m, r-k)
                for j in range(min_val):
                    y = y*y + c
                    if y > n : y %= n
                    q = q * abs(x - y)
                    if q > n: q %= n
                g = utils.gcd(q, n)
                k += m
            r <<= 1
        
        if g == n:
            while True:
               ys = ys*ys + c
               if ys > n: ys %= n
               g = utils.gcd(abs(x-ys), n)
               if g > 1: 
                break
        
        if g != n:
            return g
        else:
            return -1

def factorize_bf(n):
	sn = int(math.sqrt(n))
	f = []
	for p in small_primes:
		if p > sn:
			if n > 1:
				f.append((n, 1))
				n = 1
			break
		i = 0
		while n % p == 0:
			n //= p
			i += 1
		if i > 0:
			f.append((p, i))
			sn = int(math.sqrt(n))

	return f, n

def factorize(n, level = 3):
	if level > 1 and n <= 10**20 and n > 1:			
		g = factorize_rho(n)
		if g != -1:
			p = g
			q = int(n/g)
			return p, q
		
	if level > 0 and n > 10**20 and n > 1:
		g = factorize_ecm(n)
		if g != -1:
			p = g
			q = int(n/g)
			return p, q

if __name__ == "__main__":
	f = open('nlist.txt', 'r')
	nlist = f.readlines()
	cnt = 0
	for i in range(len(nlist)):
		n = int(nlist[i])
		print 'i =', i, 'num of digits =', len(str(n))
		start = time.time()
		p, q = factorize(n)
		end = time.time()
		t = (end-start)
		print 'i = {}, p = {}, q = {}, t = {}'.format(i, p, q, t)
		ver = False
		if p * q == n:
			ver = True
			cnt += 1
		print ver
		print ''
	print "Correct = ", cnt


		