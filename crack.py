from factor import *
from utils import gcd
import time
import sys
import matplotlib.pyplot as plt

def get_e(phi):
	e = 2
	while e < phi:
		if gcd(e, phi) == 1:
			return e
		e += 1

def get_d(e, phi):
	if e == 0:
		return (phi, 0, 1)
	else:
		g, y, x = get_d(phi % e, e)
		return (g, x - (phi // e) * y, y)

def mod_exp(a, n, m):
    res = 1
    a %= m
    while n:
        if n & 1:
            res = (res * a) % m
        a = (a * a) % m
        n >>= 1
    return res

def key_setup(p, q, n):
	phi = n - p - q + 1
	e = get_e(phi)
	d = get_d(e, phi)[1]
	d = d % phi
	if d < 0:
		d += phi
	return (e, n), (d, n)

def res(text, key):
	res = [mod_exp(i, key[0], key[1]) for i in text]
	return res

def rsa_plot(nlist):
	factor_time = []
	digit_len = []
	for n in nlist:
		start = time.time()
		p, q = factorize(n)
		end = time.time()
		t = (end - start)
		digit_len.append(len(str(n)))
		factor_time.append(t)
		print 'n = {}, p = {}, q = {}, t = {}'.format(n, p, q, t)
	plt.figure()
	plt.plot(digit_len, factor_time)
	plt.title('Number of Digits in N vs Factorization Time')
	plt.xlabel('Number of Digits in N')
	plt.ylabel('Time to Factorize(s)')
	plt.savefig('rsa_plot.png')
	plt.show()
	plt.close()

f = open('plaintext.txt')
plain_text = f.readline()
f = open('nlist.txt')
nlist = f.readlines()
nlist = [int(x) for x in nlist]

print 'Generating Plot for Time to Factorize vs Number of Digits for the numbers given in nlist.txt'
print ''
rsa_plot(nlist)
print ''

n = int(sys.argv[1])
print 'Number input given by TA is: {}'.format(n)
print ''

start = time.time()
p, q = factorize(n)
end = time.time()
t = (end-start)
print 'p = {}, q = {}'.format(p, q)
print 'Time to factorize = ', t

if p == q:
	print 'Since n is square of a prime number, the plain text cannot be decrypted correctly according to RSA conditions. Provide a number with distinct prime factors as input.'

else:
	public_key, private_key = key_setup(p, q, n)
	print 'Public Key: {}'.format(public_key)
	print 'Private Key: {}'.format(private_key)
	print ''

	print 'Plain Text:', plain_text
	print ''
	pt = [ord(c) for c in plain_text]
	print 'Plain Text as Ascii:', pt
	print ''

	et = res(pt, public_key)
	print 'Encrypted as Ascii:', et
	print ''
	dt = res(et, private_key)
	print 'Decrypted as Ascii:', dt
	print ''
	dtext = ''.join([chr(i) for i in dt])
	print 'Decrypted Text:', dtext