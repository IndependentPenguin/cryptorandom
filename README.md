# cryptorandom
Alternative random library.
Cryptorandom uses the operating system CryptoGenRandom to generate close to true randomness.
It does however come with a small performance penalty compared to python native random library.

Version 1.0 is not optimized and contains redundant code that can be improved.
Version 1.1 will be out soon.

This library is based on https://docs.python.org/3.7/library/random.html


Usage:
All 'import random' functions directly correspond to 'import cryptorandom' functions.
Simply change 'import random' to 'import cryptorandom as random' and your code will use hardware
encryption to produce close to true randomness.

Install:
pip install cryptorandom

https://github.com/IndependentPenguin/cryptorandom
