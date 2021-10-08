import math
import random
import fractions

PRIME_THRESHOLD = 100000
MR_THRESHOLD = 10**36

def binary_search(x, arr, include_equal = False):
	if x > arr[-1]:
		return len(arr)
	elif x < arr[0]:
		return 0

	l, r = 0, len(arr) - 1
	while l <= r:
		m = (l + r) >> 1
		if arr[m] == x:
			return m + 1 if not include_equal else m
		elif arr[m] < x:
			l = m + 1
		else:
			r = m - 1

	return l

def gcd(a, b):
	return fractions.gcd(a, b)

def is_prime_bf(n):
	if n < 2: return False
	if n == 2 or n == 3: return True
	if not n & 1: return False
	if not n % 3: return False
	if n < 9: return True
	sqrt_n = int(math.sqrt(n)) + 1
	for i in range(5, sqrt_n, 6):
		if not n % i or not n % (i + 2): return False
	return True


def is_prime_fast(n, use_probabilistic = False, tolerance = 30):
	firstPrime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, \
                  53, 59, 61, 67, 71]
    
	if n >= MR_THRESHOLD: 
		logn = math.log(n)
		if not use_probabilistic: 
			w = range(2, 2 * int(logn*log(logn)/log(2))) 
		else: 
			w = range(tolerance)
	elif n >= 1543267864443420616877677640751301: w = firstPrime[:20]
	elif n >= 564132928021909221014087501701: w = firstPrime[:18]
	elif n >= 59276361075595573263446330101: w = firstPrime[:16]
	elif n >= 6003094289670105800312596501: w = firstPrime[:15]
	elif n >= 3317044064679887385961981: w = firstPrime[:14]
	elif n >= 318665857834031151167461: w = firstPrime[:13]
	elif n >= 3825123056546413051: w = firstPrime[:12]
	elif n >= 341550071728321: w = firstPrime[:9]
	elif n >= 3474749660383: w = firstPrime[:7]
	elif n >= 2152302898749: w = firstPrime[:6]
	elif n >= 4759123141: w = firstPrime[:5]
	elif n >= 9006403: w = [2, 7, 61]
	elif n >= 489997:
		if n&1 and n%3 and n%5 and n%7 and n%11 and n%13 and n%17 and n%19 \
		and n%23 and n%29 and n%31 and n%37 and n%41 and n%43 and n%47 \
		and n%53 and n%59 and n%61 and n%67 and n%71 and n%73 and n%79 \
		and n%83 and n%89 and n%97 and n%101:
			hn, nm1 = n >> 1, n - 1
			p = pow(2, hn, n)
			if p == 1 or p == nm1:
				p = pow(3, hn, n)
				if p == 1 or p == nm1:
					p = pow(5, hn, n)
					return p == 1 or p == nm1
		return False
	elif n >= 42799:
		return n&1 and n%3 and n%5 and n%7 and n%11 and n%13 and n%17 \
		and n%19 and n%23 and n%29 and n%31 and n%37 and n%41 and n%43 \
		and pow(2, n-1, n) == 1 and pow(5, n-1, n) == 1
	elif n >= 841:
		return n&1 and n%3 and n%5 and n%7 and n%11 and n%13 and n%17 \
		and n%19 and n%23 and n%29 and n%31 and n%37 and n%41 and n%43 \
		and n%47 and n%53 and n%59 and n%61 and n%67 and n%71 and n%73 \
		and n%79 and n%83 and n%89 and n%97 and n%101 and n%103 \
		and pow(2, n-1, n) == 1
	elif n >= 25:
		return n&1 and n%3 and n%5 and n%7 \
		and n%11 and n%13 and n%17 and n%19 and n%23
	elif n >= 4:
		return n&1 and n%3
	else:
		return n > 1
    
	if not (n&1 and n%3 and n%5 and n%7 and n%11 and n%13 and n%17 \
		   and n%19 and n%23 and n%29 and n%31 and n%37 and n%41 and n%43 \
		   and n%47 and n%53 and n%59 and n%61 and n%67 and n%71 and n%73 \
		   and n%79 and n%83 and n%89): return False
    
	s = 0
	d = n - 1
	while not d & 1:
		d >>= 1
		s += 1
	for k in w:
		if use_probabilistic: 
			p = random.randint(2, n-2)
		else:
			p = k
		x = pow(p, d, n)
		if x == 1: continue
		for _ in range(s):
			if x+1 == n: break
			x = x*x % n
		else: return False
	return True


def is_prime(n, use_probabilistic = False, tolerance = 30):
	if n < PRIME_THRESHOLD: 
		return is_prime_bf(n)
	else: 
		if use_probabilistic:
			return is_prime_fast(n, use_probabilistic, tolerance)
		else:
			if n < MR_THRESHOLD:
				return is_prime_fast(n)
			else:
				return is_prime_fast(n, True, 40)

segs = [[] for _ in xrange(60)]
under60 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
dAll = [1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59]
DFG1 = [[1, 0, 1], [1, 0, 11], [1, 0, 19], \
			[1, 0, 29], [1, 2, 15], [1, 3, 5], [1, 3, 25], [1, 5, 9], \
			[1, 5, 21], [1, 7, 15], [1, 8, 15], [1, 10, 9], \
			[1, 10, 21], [1, 12, 5], [1, 12, 25], [1, 13, 15], \
			[13, 1, 3], [13, 1, 27], [13, 4, 3], [13, 4, 27], \
			[13, 6, 7], [13, 6, 13], [13, 6, 17], [13, 6, 23], \
			[13, 9, 7], [13, 9, 13], [13, 9, 17], [13, 9, 23], \
			[13, 11, 3], [13, 11, 27], [13, 14, 3], [13, 14, 27], \
			[17, 2, 1], [17, 2, 11], [17, 2, 19], [17, 2, 29], \
			[17, 7, 1], [17, 7, 11], [17, 7, 19], [17, 7, 29], \
			[17, 8, 1], [17, 8, 11], [17, 8, 19], [17, 8, 29], \
			[17, 13, 1], [17, 13, 11], [17, 13, 19], [17, 13, 29], \
			[29, 1, 5], [29, 1, 25], [29, 4, 5], [29, 4, 25], \
			[29, 5, 7], [29, 5, 13], [29, 5, 17], [29, 5, 23], \
			[29, 10, 7], [29, 10, 13], [29, 10, 17], [29, 10, 23], \
			[29, 11, 5], [29, 11, 25], [29, 14, 5], [29, 14, 25], \
			[37, 2, 9], [37, 2, 21], [37, 3, 1], [37, 3, 11], \
			[37, 3, 19], [37, 3, 29], [37, 7, 9], [37, 7, 21], \
			[37, 8, 9], [37, 8, 21], [37, 12, 1], [37, 12, 11], \
			[37, 12, 19], [37, 12, 29], [37, 13, 9], [37, 13, 21], \
			[41, 2, 5], [41, 2, 25], [41, 5, 1], [41, 5, 11], \
			[41, 5, 19], [41, 5, 29], [41, 7, 5], [41, 7, 25], \
			[41, 8, 5], [41, 8, 25], [41, 10, 1], [41, 10, 11], \
			[41, 10, 19], [41, 10, 29], [41, 13, 5], [41, 13, 25], \
			[49, 0, 7], [49, 0, 13], [49, 0, 17], [49, 0, 23], \
			[49, 1, 15], [49, 4, 15], [49, 5, 3], [49, 5, 27], \
			[49, 6, 5], [49, 6, 25], [49, 9, 5], [49, 9, 25], \
			[49, 10, 3], [49, 10, 27], [49, 11, 15], [49, 14, 15], \
			[53, 1, 7], [53, 1, 13], [53, 1, 17], [53, 1, 23], \
			[53, 4, 7], [53, 4, 13], [53, 4, 17], [53, 4, 23], \
			[53, 11, 7], [53, 11, 13], [53, 11, 17], [53, 11, 23], \
			[53, 14, 7], [53, 14, 13], [53, 14, 17], [53, 14, 23]]

DFG2 = [[7, 1, 2], [7, 1, 8], [7, 1, 22], \
			[7, 1, 28], [7, 3, 10], [7, 3, 20], [7, 7, 10], \
			[7, 7, 20], [7, 9, 2], [7, 9, 8], [7, 9, 22], [7, 9, 28], \
			[19, 1, 4], [19, 1, 14], [19, 1, 16], [19, 1, 26], \
			[19, 5, 2], [19, 5, 8], [19, 5, 22], [19, 5, 28], \
			[19, 9, 4], [19, 9, 14], [19, 9, 16], [19, 9, 26], \
			[31, 3, 2], [31, 3, 8], [31, 3, 22], [31, 3, 28], \
			[31, 5, 4], [31, 5, 14], [31, 5, 16], [31, 5, 26], \
			[31, 7, 2], [31, 7, 8], [31, 7, 22], [31, 7, 28], \
			[43, 1, 10], [43, 1, 20], [43, 3, 4], [43, 3, 14], \
			[43, 3, 16], [43, 3, 26], [43, 7, 4], [43, 7, 14], \
			[43, 7, 16], [43, 7, 26], [43, 9, 10], [43, 9, 20]]

DFG3 = [[11, 0, 7], [11, 0, 13], [11, 0, 17], \
			[11, 0, 23], [11, 2, 1], [11, 2, 11], [11, 2, 19], \
			[11, 2, 29], [11, 3, 4], [11, 3, 14], [11, 3, 16], \
			[11, 3, 26], [11, 5, 2], [11, 5, 8], [11, 5, 22], \
			[11, 5, 28], [11, 7, 4], [11, 7, 14], [11, 7, 16], \
			[11, 7, 26], [11, 8, 1], [11, 8, 11], [11, 8, 19], \
			[11, 8, 29], [23, 1, 10], [23, 1, 20], [23, 2, 7], \
			[23, 2, 13], [23, 2, 17], [23, 2, 23], [23, 3, 2], \
			[23, 3, 8], [23, 3, 22], [23, 3, 28], [23, 4, 5], \
			[23, 4, 25], [23, 6, 5], [23, 6, 25], [23, 7, 2], \
			[23, 7, 8], [23, 7, 22], [23, 7, 28], [23, 8, 7], \
			[23, 8, 13], [23, 8, 17], [23, 8, 23], [23, 9, 10], \
			[23, 9, 20], [47, 1, 4], [47, 1, 14], [47, 1, 16], \
			[47, 1, 26], [47, 2, 5], [47, 2, 25], [47, 3, 10], \
			[47, 3, 20], [47, 4, 1], [47, 4, 11], [47, 4, 19], \
			[47, 4, 29], [47, 6, 1], [47, 6, 11], [47, 6, 19], \
			[47, 6, 29], [47, 7, 10], [47, 7, 20], [47, 8, 5], \
			[47, 8, 25], [47, 9, 4], [47, 9, 14], [47, 9, 16], \
			[47, 9, 26], [59, 0, 1], [59, 0, 11], [59, 0, 19], \
			[59, 0, 29], [59, 1, 2], [59, 1, 8], [59, 1, 22], \
			[59, 1, 28], [59, 4, 7], [59, 4, 13], [59, 4, 17], \
			[59, 4, 23], [59, 5, 4], [59, 5, 14], [59, 5, 16], \
			[59, 5, 26], [59, 6, 7], [59, 6, 13], [59, 6, 17], \
			[59, 6, 23], [59, 9, 2], [59, 9, 8], [59, 9, 22], \
			[59, 9, 28]]


def small_sieve(n):
	correction = (n % 6 > 1)
	n = {0: n, 1: n-1, 2: n+4, 3: n+3, 4: n+2, 5: n+1}[n % 6]
	sieve = [True] * (n/3)
	sieve[0] = False
	limit = int(math.sqrt(n))/3 + 1
	# Use a wheel (mod 6)
	for i in range(limit):
		if sieve[i]:
 			k = 3*i + 1 | 1
			sieve[((k*k)/3) :: (k << 1)] = \
					[False] * ((n/6 - (k*k)/6 - 1)/k + 1)
			sieve[(k * k + (k << 2) - \
					(k << 1) * (i & 1)) / 3 :: (k << 1)] = \
					[False] * ((n/6 - (k*k + (k << 2) - \
						2*k * (i & 1))/6 - 1)/k + 1)
	return [2, 3] + [3*i + 1 | 1 for i in xrange(1, n/3 - correction) if sieve[i]]


def enum1(d, f, g, L, B, segs):
	x, y0, temp = f, g, L+B
	k0 = (4*f*f + g*g - d) / 60
	while k0 < temp:
		k0 += x + x + 15
		x += 15

	while True:
		x -= 15
		k0 -= x + x + 15
		if x <= 0: 
			return
		while k0 < L:
			k0 += y0 + 15
			y0 += 30

		k, y = k0, y0
		while k < temp:
			segs[d][(k-L) >> 5] ^= 1 << ((k-L) & 31)
			k += y + 15
			y += 30

def enum2(d, f, g, L, B, segs):
	x, y0, temp = f, g, L+B
	k0 = (3*f*f + g*g - d) / 60
	while k0 < temp:
		k0 += x + 5
		x += 10

	while True:
		x -= 10
		k0 -= x + 5
		if x <= 0: 
			return
		while k0 < L:
			k0 += y0 + 15
			y0 += 30

		k, y = k0, y0
		while k < temp:
			segs[d][(k-L) >> 5] ^= 1 << ((k-L) & 31)

			k += y + 15
			y += 30


def enum3(d, f, g, L, B, segs):
	x, y0, temp = f, g, L+B
	k0 = (3*f*f - g*g - d) / 60

	while True:
		while k0 >= temp:
			if x <= y0: 
				return
			k0 -= y0 + 15
			y0 += 30

		k, y = k0, y0
		while k >= L and y < x:
			segs[d][(k-L) >> 5] ^= 1 << ((k-L) & 31)
			k -= y + 15
			y += 30
		
		k0 += x + 5
		x += 10


def sieve_of_atkin(n):
	sqrt_n, u, r = int(math.sqrt(n)), n + 32, 17
	B, lu = 60 * sqrt_n, math.log(u)
	primes = small_sieve(sqrt_n)
	ret = under60 + [0] * int(u/lu + u/(lu*lu) * 1.5 - r)
	for d in dAll:
		segs[d] = [0] * ((B >> 5) + 1)

	lim = n/60 + 1
	for L in xrange(1, lim, B):
		for d in dAll:
			for k in xrange(len(segs[d])):
				segs[d][k] = 0

		lim2 = 60 * (L+B)
		for d,f,g in DFG1:
			enum1(d, f, g, L, B, segs)
		for d,f,g in DFG2:
			enum2(d, f, g, L, B, segs)
		for d,f,g in DFG3:
			enum3(d, f, g, L, B, segs)

		for p in primes:
			p2 = p * p
			if p2 > lim2: 
				break
			if p >= 7:
				b = -utils.xgcd(p2, 60)
				if b < 0: b += p2
				for d in dAll:
					x = b * (60*L + d) % p2
					while x < B:
						segs[d][x >> 5] &= ~(1 << (x & 31))
						x += p2

		for j in xrange((B >> 5) + 1):
			for x in xrange(32):
				k = 60 * (L + x + (j << 5))
				for d in dAll:
					if k + d > n:
						return ret[:r]
					if ((segs[d][j] << 31 - x) & 0xFFFFFFFF) >= 0x80000000:
						ret[r] = 60*k + d
						r += 1

def prime_sieve(n):
	if n <= 60:
		return under60[:utils.binary_search(n, under60)]
	elif n <= 35 * 10**5:
		return small_sieve(n)
	elif n <= 10**10:
		return sieve_of_atkin(n)
	else:
		return segmented_sieve(2, n)


def segmented_sieve(lo, hi):
	if hi < lo: return []
	max_prime, pos = int(math.sqrt(hi)), 0
	base_primes = prime_sieve(max_prime)
	primes = [0] * int(math.ceil(1.5 * hi/math.log(hi)) - math.floor(1.5 * lo/math.log(lo)))

	if lo < max_prime:
		lo_pos = utils.binary_search(lo, base_primes, include_equal = True)
		for k in xrange(lo_pos, len(base_primes)):
			primes[pos] = base_primes[k]
			pos += 1
		lo = max_prime

	delta = 2097152 if hi - lo >= 2097152 else 65536

	l1, l = len(base_primes), (delta >> 4) + 1
	int_size, sieve = l << 3, bytearray([0x0] * l)
	lo_1, hi_1 = lo, lo + delta
	
	while lo_1 <= hi:
		if lo_1 != lo:
			for i in range(l):
				sieve[i] = 0

		if (lo_1 & 1) == 0: 
			lo_1 += 1

		for i in xrange(1, l1):
			p = base_primes[i]
			k = (p - (lo_1 % p)) % p
			if (k & 1) == 1: 
				k += p
			k >>= 1
			while k < int_size:
				sieve[k >> 3] |= 1 << (k & 7)
				k += p

		end = min(hi_1, hi) + 1
		for n in range(lo_1, end, 2):
			d = n - lo_1
			if ((sieve[d >> 4] >> ((d >> 1) & 0x7)) & 0x1) == 0x0:
				primes[pos] = n
				pos += 1

		lo_1 = hi_1 + 1
		hi_1 = lo_1 + delta

	return primes[:pos]