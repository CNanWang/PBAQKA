import netsquid as ns
import hashlib
import random
import math
import datetime
import time
import numpy as np
from math import log
from os import urandom
from fastpbkdf2 import pbkdf2_hmac
from fuzzyextractor import FuzzyExtractor

# registration
class FuzzyExtractor(object):
    """The FuzzyExtractor utilizes the code from github.com/carter-yagemann/python-fuzzy-extractor/__init__.py."""
    def __init__(self, length, ham_err, rep_err=0.001, **locker_args):
        self.parse_locker_args(**locker_args)
        self.length = length
        self.cipher_len = self.length + self.sec_len
        # Calculate the number of helper values needed to be able to reproduce
        # keys given ham_err and rep_err.
        bits = length * 8
        const = float(ham_err) / log(bits)
        num_helpers = (bits ** const) * log(float(2) / rep_err, 2)
        # num_helpers needs to be an integer
        self.num_helpers = int(round(num_helpers))

    def parse_locker_args(self, hash_func='sha256', sec_len=2, nonce_len=16):
        self.hash_func = hash_func
        self.sec_len = sec_len
        self.nonce_len = nonce_len

    def generate(self, value):
        if isinstance(value, (bytes, str)):
            value = np.fromstring(value, dtype=np.uint8)

        key = np.fromstring(urandom(self.length), dtype=np.uint8)
        key_pad = np.concatenate((key, np.zeros(self.sec_len, dtype=np.uint8)))

        nonces = np.zeros((self.num_helpers, self.nonce_len), dtype=np.uint8)
        masks = np.zeros((self.num_helpers, self.length), dtype=np.uint8)
        digests = np.zeros((self.num_helpers, self.cipher_len), dtype=np.uint8)

        for helper in range(self.num_helpers):
            nonces[helper] = np.fromstring(urandom(self.nonce_len), dtype=np.uint8)
            masks[helper] = np.fromstring(urandom(self.length), dtype=np.uint8)

        vectors = np.bitwise_and(masks, value)

        for helper in range(self.num_helpers):
            d_vector = vectors[helper].tobytes()
            d_nonce = nonces[helper].tobytes()
            digest = pbkdf2_hmac(self.hash_func, d_vector, d_nonce, 1, self.cipher_len)
            digests[helper] = np.fromstring(digest, dtype=np.uint8)

        ciphers = np.bitwise_xor(digests, key_pad)

        return (key.tobytes(), (ciphers, masks, nonces))

    def reproduce(self, value, helpers):
        if isinstance(value, (bytes, str)):
            value = np.fromstring(value, dtype=np.uint8)

        if self.length != len(value):
            raise ValueError('Cannot reproduce key for value of different length')

        ciphers = helpers[0]
        masks = helpers[1]
        nonces = helpers[2]

        vectors = np.bitwise_and(masks, value)

        digests = np.zeros((self.num_helpers, self.cipher_len), dtype=np.uint8)
        for helper in range(self.num_helpers):
            d_vector = vectors[helper].tobytes()
            d_nonce = nonces[helper].tobytes()
            digest = pbkdf2_hmac(self.hash_func, d_vector, d_nonce, 1, self.cipher_len)
            digests[helper] = np.fromstring(digest, dtype=np.uint8)

        plains = np.bitwise_xor(digests, ciphers)

        checks = np.sum(plains[:, -self.sec_len:], axis=1)
        for check in range(self.num_helpers):
            if checks[check] == 0:
                return plains[check, :-self.sec_len].tobytes()

        return None

# four single photons, a1=|0>, a2=|1>, a3=|+>, a4=|->
a1, a2, a3, a4 = ns.qubits.create_qubits(4)
ns.qubits.operate(a2, ns.X)
ns.qubits.operate(a3, ns.H)
ns.qubits.operate(a4, ns.X)
ns.qubits.operate(a4, ns.H)

states = {
    'a1': '|0>',
    'a2': '|1>',
    'a3': '|+>',
    'a4': '|->'
}

bell_operators = []
p0, p1 = ns.Z.projectors
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p0 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p0) * (ns.H ^ ns.I) * ns.CNOT)
bell_operators.append(ns.CNOT * (ns.H ^ ns.I) * (p1 ^ p1) * (ns.H ^ ns.I) * ns.CNOT)

def truncate_binary_id(string, factor=2):
    truncated_parts = [string[i:i+factor] for i in range(2, len(string), factor*2)]
    return ''.join(truncated_parts)

def bitwise_XOR(string1, string2):
    max_length = max(len(string1), len(string2))
    string1_add, string2_add = [bin_str.zfill(max_length) for bin_str in [string1, string2]]
    xor_result = ''.join('1' if bit1 != bit2 else '0' for bit1, bit2 in zip(string1_add, string2_add))
    return xor_result

def hash_fuction(value, N):
    hash_digest = hashlib.sha256(value.encode('utf-8')).digest()
    random.seed(int.from_bytes(hash_digest, byteorder='big'))
    random_bits = ''.join(['1' if random.randint(0, 1) == 1 else '0' for _ in range(4 * N)])
    return random_bits

def create_message(data):
    timestamp = datetime.datetime.utcnow()
    return data, timestamp

def validate_timestamp(received_timestamp, tolerance=300):
    current_time = datetime.datetime.utcnow()
    delta = current_time - received_timestamp
    return abs(delta.total_seconds()) <= tolerance

def alter_biometric_data(data, max_changes=1):
    if not data:
        return data
    data_bytearray = bytearray(data)
    changes_made = 0
    for _ in range(max_changes):
        if changes_made >= max_changes:
            break
        position = urandom(1)[0] % len(data_bytearray)
        new_char = urandom(1)
        while new_char >= b'\x20':
            new_char = urandom(1)
        data_bytearray[position] = new_char[0]
        changes_made += 1
    return bytes(data_bytearray)

def generate_bell_states(binary_string):
    state = [] # store the state vector of the qubit
    bell_states_1 = [] # store the first group of quantum states
    bell_states_2 = []  # store the second group of quantum states
    for i in range(0, len(binary_string), 2):
        if binary_string[i:i + 2] == '00':
            a1, a2 = ns.qubits.create_qubits(2)
            ns.qubits.operate(a1, ns.H)
            ns.qubits.operate([a1, a2], ns.CNOT)
            state.append(ns.qubits.reduced_dm([a1, a2]))
            bell_states_1.append(a1)
            bell_states_2.append(a2)
        elif binary_string[i:i + 2] == '01':
            a1, a2 = ns.qubits.create_qubits(2)
            ns.qubits.operate(a2, ns.X)
            ns.qubits.operate(a1, ns.H)
            ns.qubits.operate([a1, a2], ns.CNOT)
            state.append(ns.qubits.reduced_dm([a1, a2]))
            bell_states_1.append(a1)
            bell_states_2.append(a2)
        elif binary_string[i:i + 2] == '10':
            a1, a2 = ns.qubits.create_qubits(2)
            ns.qubits.operate(a1, ns.X)
            ns.qubits.operate(a1, ns.H)
            ns.qubits.operate([a1, a2], ns.CNOT)
            state.append(ns.qubits.reduced_dm([a1, a2]))
            bell_states_1.append(a1)
            bell_states_2.append(a2)
        elif binary_string[i:i + 2] == '11':
            a1, a2 = ns.qubits.create_qubits(2)
            ns.qubits.operate(a1, ns.X)
            ns.qubits.operate(a2, ns.X)
            ns.qubits.operate(a1, ns.H)
            ns.qubits.operate([a1, a2], ns.CNOT)
            state.append(ns.qubits.reduced_dm([a1, a2]))
            bell_states_1.append(a1)
            bell_states_2.append(a2)
        else:
            print("Invalid binary string")
    return state, bell_states_1, bell_states_2

def generate_single_photon(binary_string):
    DS = []  # store decoy states
    Photon_states = []  # store decoy binary states
    BS = []  # store measurement bases
    for i in range(0, len(binary_string)):
        if binary_string[i] == '0':
            DS.append(random.choice([a1, a2]))
            BS.append('Z')
            if DS[i] == a1:
                Photon_states.append(states['a1'])
            else:
                Photon_states.append(states['a2'])
        else:
            DS.append(random.choice([a3, a4]))
            BS.append('X')
            if DS[i] == a3:
                Photon_states.append(states['a3'])
            else:
                Photon_states.append(states['a4'])
    return DS, Photon_states, BS

def insert_decoys(ds, s2):
    positions = sorted(random.sample(range(len(s2) + 1), len(ds)))
    new_sequence = s2.copy()
    for index, position in enumerate(positions):
        new_sequence.insert(position, ds[index])
    return new_sequence, positions

def remove_decoys(S2_Alice_Bob, insert_positions):
    S2_Alice = S2_Alice_Bob[:]
    for position in reversed(insert_positions):
        S2_Alice.pop(position)
    return S2_Alice

def measure_single_photon(DS, BS):
    photon_state = [] # store the measurement result
    for i in range(0, len(DS)):
        if BS[i] == 'Z':
            m, prob = ns.qubits.measure(DS[i], observable=ns.Z)
            if m == 0:
                photon_state.append(states['a1'])
            else:
                photon_state.append(states['a2'])
        else:
            m, prob = ns.qubits.measure(DS[i], observable=ns.X)
            if m == 0:
                photon_state.append(states['a3'])
            else:
                photon_state.append(states['a4'])
    return photon_state

def compare_single_photon(ps_a, ps_b):
    error = 0
    for i in range(0, len(ps_a)):
        if ps_a[i] == ps_b[i]:
            pass
        else:
            error = error + 1
        return error

def bell_measurement(S1, S2):
    measurement_results = []
    for j in range(0, N):
        meas, prob = ns.qubits.gmeasure((S1[j], S2[j]), meas_operators=bell_operators)
        labels_bell = ("00", "01", "10", "11")
        measurement_results.append(labels_bell[meas])
    return measurement_results

def xor_values(value1, value2):
    return bin(int(value1, 2) ^ int(value2, 2))[2:].zfill(2)

def get_msb(msa, xor_result):
    a, b = msa
    if xor_result == '00':
        msb = a + b
        k = msa + msb
    elif xor_result == '01':
        msb = a + ('1' if b == '0' else '0')
        k = msa + msb
    elif xor_result == '10':
        msb = ('1' if a == '0' else '0') + b
        k = msa + msb
    elif xor_result == '11':
        msb = ('1' if a == '0' else '0') + ('1' if b == '0' else '0')
        k = msa + msb
    return msb, k

def derive_MSB(qab, msa):
    n = len(msa)
    msb_list = []
    k_list = []
    for j in range(n):
        qab_j = qab[2 * j:2 * j + 2]
        qab_nj = qab[2 * (n + j):2 * (n + j) + 2]
        xor_result = xor_values(qab_j, qab_nj)
        msb, k = get_msb(msa[j], xor_result)
        msb_list.append(msb)
        k_list.append(k)
    return msb_list, k_list

def get_msa(msb, xor_result):
    a, b = msb
    if xor_result == '00':
        msa = a + b
        k = msa + msb
    elif xor_result == '01':
        msa = a + ('1' if b == '0' else '0')
        k = msa + msb
    elif xor_result == '10':
        msa = ('1' if a == '0' else '0') + b
        k = msa + msb
    elif xor_result == '11':
        msa = ('1' if a == '0' else '0') + ('1' if b == '0' else '0')
        k = msa + msb
    return msa, k

def derive_MSA(qab, msb):
    n = len(msb)
    msa_list = []
    k_list = []
    for j in range(n):
        qab_j = qab[2 * j:2 * j + 2]
        qab_nj = qab[2 * (n + j):2 * (n + j) + 2]
        xor_result = xor_values(qab_j, qab_nj)
        msa, k = get_msa(msb[j], xor_result)
        msa_list.append(msa)
        k_list.append(k)
    return msa_list, k_list

# Alice and Bob generate ID and BIO
N = int(input("Please enter N: N = "))
ID_Alice = urandom(N)
ID_Bob = urandom(N)
BIO_Alice = urandom(N)
BIO_Bob = urandom(N)
print('BIO_Alice :', BIO_Alice)
print('BIO_Bob :', BIO_Bob)
BIO_binary_a = ''.join(format(byte, '08b') for byte in BIO_Alice)
BIO_binary_b = ''.join(format(byte, '08b') for byte in BIO_Bob)
ID_binary_A = truncate_binary_id(''.join(format(byte, '08b') for byte in ID_Alice), factor=2)
ID_binary_B = truncate_binary_id(''.join(format(byte, '08b') for byte in ID_Bob), factor=2)
print('ID_binary_A :', ID_binary_A)
print('ID_binary_B :', ID_binary_B, '\n')
BIO_binary_A = truncate_binary_id(BIO_binary_a, factor=2)
BIO_binary_B = truncate_binary_id(BIO_binary_b, factor=2)

# Registration authority generates reconstruction parameters PA and PB, private parameters RA and RB
extractor = FuzzyExtractor(N, N/2)
RA, PA = extractor.generate(BIO_Alice)
RB, PB = extractor.generate(BIO_Bob)

# Registration authority generates binary numbers RN_binary_A and RN_binary_B
RN_binary_A = ''.join(random.choice('01') for _ in range(4 * N))
RN_binary_B = ''.join(random.choice('01') for _ in range(4 * N))
a = hash_fuction(bitwise_XOR(ID_binary_A, RN_binary_A), N)
b = hash_fuction(bitwise_XOR(ID_binary_B, RN_binary_B), N)

# Alice (Bob) computes HIDA and q1 (HIDB and q2)
HIDA = bitwise_XOR(ID_binary_A, hash_fuction(bitwise_XOR(ID_binary_A, a), N))
HIDB = bitwise_XOR(ID_binary_B, hash_fuction(bitwise_XOR(ID_binary_B, b), N))
q1 = ''.join(random.choice('01') for _ in range(4 * N))
q2 = ''.join(random.choice('01') for _ in range(4 * N))
print('q1 :', q1)
print('q2 :', q2)
A1 = bitwise_XOR(q1, hash_fuction((HIDA+a), N))
B1 = bitwise_XOR(q2, hash_fuction((HIDB+b), N))
print('A1 :', A1)
print('B1 :', B1, '\n')

# Simulate the generation of new biometrics BIO_Alice_new and BIO_Bob_new
BIO_Alice_new = alter_biometric_data(BIO_Alice, 1)
BIO_Bob_new = alter_biometric_data(BIO_Bob, 1)

data_a = (HIDA, BIO_Alice_new, A1)
data_b = (HIDB, BIO_Bob_new, B1)
timestamp_a = create_message(data_a)[1]
timestamp_b = create_message(data_b)[1]
time.sleep(2)
received_timestamp_a = timestamp_a
received_timestamp_b = timestamp_b
# Registration authority computes private biometric key_RA
if validate_timestamp(received_timestamp_a):
    print("Alice's timestamp is valid")
    # Suppose IDA exists in the database
    IDA_new = bitwise_XOR(HIDA, hash_fuction(bitwise_XOR(ID_binary_A, a), N))
    if IDA_new == ID_binary_A:
        key_RA = extractor.reproduce(BIO_Alice_new, PA)
        if key_RA == RA:
            print("Alice's identity is real!")
            RA_qa = bitwise_XOR(A1, hash_fuction((HIDA + a), N))
            RQA = bitwise_XOR(q2, hash_fuction((a + ID_binary_A + RA_qa), N))
            print('RQA :', RQA, '\n')
    else:
        print('Alice is not registered, please register first!')
else:
    print("Alice's timestamp is invalid")

# Registration authority computes private biometric key_RB
if validate_timestamp(received_timestamp_b):
    print("Bob's timestamp is valid")
    # Suppose IDA exists in the database
    IDB_new = bitwise_XOR(HIDB, hash_fuction(bitwise_XOR(ID_binary_B, b), N))
    if IDB_new == ID_binary_B:
        key_RB = extractor.reproduce(BIO_Bob_new, PB)
        if key_RB == RB:
            print("Bob's identity is real!")
            RA_qb = bitwise_XOR(B1, hash_fuction((HIDB + b), N))
            RQB = bitwise_XOR(q1, hash_fuction((b + ID_binary_B + RA_qb), N))
            print('RQB :', RQB, '\n')
    else:
        print('Alice is not registered, please register first!')
else:
    print("Bob's timestamp is invalid")

# Alice (Bob) computes QAB
Alice_q2 = bitwise_XOR(RQA, hash_fuction((a + ID_binary_A + q1), N))
Alice_QAB = hash_fuction((q1 + Alice_q2), N)
print('Alice_QAB :', Alice_QAB)
Bob_q1 = bitwise_XOR(RQB, hash_fuction((b + ID_binary_B + q2), N))
Bob_QAB = hash_fuction((Bob_q1 + q2), N)
print('Bob_QAB :', Bob_QAB)
print('Alice_QAB should equal to Bob_QAB:', Alice_QAB == Bob_QAB, '\n')

# Alice executes tensor product
State, S1_Alice, S2_Alice = generate_bell_states(Alice_QAB)
print('S1_Alice :', S1_Alice, '\n')
print('S2_Alice :', S2_Alice, '\n')
State_a = State[:N]
State_b = State[N:]
tensor_products = []
for a, b in zip(State_a, State_b):
    tensor_product = np.kron(a, b)
    tensor_products.append(tensor_product)

# Alice inserts decoy photons into S2_Alice to get S2_Alice_Bob
Decoy_binary =''.join([str(random.randint(0, 1)) for _ in range(N)])
DS_Alice, PS_State_Alice, BS_Alice = generate_single_photon(Decoy_binary)
S2_Alice_Bob, insert_positions = insert_decoys(DS_Alice, S2_Alice)
print('Decoy_binary :', Decoy_binary)
print(f"Alice prepares single photon states: {' '.join(PS_State_Alice)}")
print('DS_Alice :', DS_Alice, '\n')
print('S2_Alice_Bob :', S2_Alice_Bob, '\n')
print('insert_positions :', insert_positions, '\n')

# Bob measures the decoy photons and Alice computes the error rate
S2_Bob = remove_decoys(S2_Alice_Bob, insert_positions)
PS_State_Bob = measure_single_photon(DS_Alice, BS_Alice)
print(f"Bob measures single photon states: {' '.join(PS_State_Bob)}\n")
print('S2_Bob :', S2_Bob, '\n')

Error = compare_single_photon(PS_State_Alice, PS_State_Bob)
print(f"The error rate is: {Error/N}")

if Error/N <= 0.05:
    print('S2_Alice_Bob is complete!\n')
    # Alice performs bell measurement on AS1 and AS2
    AS1 = S1_Alice[:N]
    AS2 = S1_Alice[N:]
    BS1 = S2_Alice[:N]
    BS2 = S2_Alice[N:]
    MSA = bell_measurement(AS1, AS2)
    MSB = bell_measurement(BS1, BS2)
    print('MSA :', MSA)
    print('MSB :', MSB, '\n')

    # Alice derives MSB_Alice
    MSB_Alice, KA = derive_MSB(Alice_QAB, MSA)
    print("MSB_Alice :", MSB_Alice)
    print("KA :", KA, '\n')

    # Bob derives MSA_Bob
    MSA_Bob, KB = derive_MSA(Bob_QAB, MSB_Alice)
    print("MSA_Bob :", MSA_Bob)
    print("KB :", KB, '\n')
    print('KA should equal to KB :', KA == KB)
else:
    print('S2_Alice_Bob is not complete, please send S2_Alice_Bob again!')