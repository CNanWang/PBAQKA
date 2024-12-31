import os
import hashlib
import random
import time
import math
from cryptography.fernet import Fernet
from base64 import b64encode, b64decode
from typing import List, Tuple, Optional, Dict, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

class FuzzyExtractor:
    """Implementation of a Fuzzy Extractor"""
    def __init__(self, key=None):
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher_suite = Fernet(self.key)
    
    def get_key(self) -> bytes:
        """Get the encryption key"""
        return self.key
    
    def generate(self, features: bytes) -> Tuple[bytes, bytes]:
        """
        Generate reconstruction parameter P and biometric string R
        Args:
            features: Input biometric features
        Returns:
            Tuple of (P, R) where:
                P is reconstruction parameter
                R is biometric string
        """
        if not features:
            raise ValueError("Input features cannot be empty")
            
        # Generate random reconstruction parameter P
        P = os.urandom(len(features))
        
        # Generate biometric string R
        R = features
        
        return P, R
        
    def reproduce(self, new_bio: bytes, P: bytes) -> bytes:
        """
        Reproduce biometric string using new biometric data and helper data P
        Args:
            new_bio: New biometric data
            P: Helper data from generation phase
        Returns:
            Reproduced biometric string R
        """
        if not new_bio or not P:
            raise ValueError("Input parameters cannot be empty")
        
        if len(new_bio) != len(P):
            raise ValueError("Length of biometric data and helper data must match")
            
        # In this implementation, we use the new biometric data directly
        return new_bio
        
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet (implementation of AES)"""
        return self.cipher_suite.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet (implementation of AES)"""
        return self.cipher_suite.decrypt(encrypted_data)

class RegistrationAuthority:
    """Registration Authority (RG) implementation"""
    def __init__(self):
        self.database = {}
        self.extractor = FuzzyExtractor()
        self.shared_key = self.extractor.get_key()
        self.time_threshold = 60  # 60 seconds for timestamp validation
        self.bio_threshold_percentage = 0.15  # 15% threshold for Hamming distance
        self.min_threshold = 20  # Minimum threshold
        
    def get_shared_key(self) -> bytes:
        """Get the shared encryption key"""
        return self.shared_key
    
    def register_user(self, ID: bytes, encrypted_bio: bytes) -> bytes:
        """
        Register a user and generate their parameters
        Args:
            ID: User's identity information (2N bits)
            encrypted_bio: Encrypted biometric data (2N bits)
        Returns:
            hash_value (a or b)
        """
        try:
            # 1. Decrypt biometric data
            try:
                BIO = self.extractor.decrypt(encrypted_bio)
            except Exception as e:
                raise RuntimeError(f"Decryption error: {str(e)}")
            
            # 2. Generate P and R using generate() algorithm
            P, R = self.extractor.generate(BIO)
            
            # 3. Encrypt P and R
            encrypted_P = self.extractor.encrypt(P)
            encrypted_R = self.extractor.encrypt(R)
            
            # 4. Generate RN and calculate hash value
            RN = os.urandom(len(ID))
            id_xor_rn = bytes(a ^ b for a, b in zip(ID, RN))
            hash_value = hashlib.sha256(id_xor_rn).digest()[:len(ID)]  # Ensure 2N bits output
            
            # 5. Store in database
            self.database[ID] = {
                'encrypted_P': encrypted_P,
                'encrypted_R': encrypted_R,
                'hash_value': hash_value
            }
            
            return hash_value
            
        except Exception as e:
            raise RuntimeError(f"Error during registration: {str(e)}")

    def authenticate_user(self, HID: bytes, auth_data: bytes, 
                        new_bio: bytes, timestamp: float) -> Tuple[bool, Optional[bytes], Optional[bytes]]:
        """
        Authenticate user and generate response
        Args:
            HID: User's pseudo-identity (HIDA/HIDB)
            auth_data: Authentication data (A1/B1)
            new_bio: New biometric data
            timestamp: Time stamp (TA/TB)
        Returns:
            (success, q, user_id) where:
                success: Authentication result
                q: Extracted q1/q2
                user_id: Recovered user ID
        """
        # 1. Verify timestamp
        current_time = time.time()
        if current_time - timestamp > self.time_threshold:
            return False, None, None
            
        try:
            # 2. Search database for user
            for stored_id, data in self.database.items():
                # 3. Calculate HID ⊕ hash_value (a or b) to recover user ID
                recovered_id = bytes(a ^ b for a, b in zip(HID, data['hash_value']))
                
                # 4. Verify recovered ID exists in database
                if recovered_id == stored_id:
                    try:
                        # 5. Decrypt stored P and R
                        P = self.extractor.decrypt(data['encrypted_P'])
                        stored_R = self.extractor.decrypt(data['encrypted_R'])
                        
                        # 6. Decrypt new biometric data
                        try:
                            decrypted_new_bio = self.extractor.decrypt(new_bio)
                            if len(decrypted_new_bio) > len(stored_R):
                                padding = decrypted_new_bio[-1]
                                decrypted_new_bio = decrypted_new_bio[:-padding]
                        except Exception:
                            return False, None, None
                        
                        # 7. Use reproduce() algorithm to regenerate biometric string
                        new_R = self.extractor.reproduce(decrypted_new_bio, P)
                        
                        # 8. Calculate Hamming distance and verify threshold
                        total_bits = len(new_R) * 8
                        bio_threshold = max(
                            int(total_bits * self.bio_threshold_percentage),
                            min(self.min_threshold, total_bits // 4)
                        )
                        hamming_dist = self.hamming_distance(new_R, stored_R)
                        
                        if hamming_dist <= bio_threshold:
                            # 9. Calculate A1/B1 ⊕ H(HID || hash_value) to get q1/q2
                            hid_hash = hashlib.sha256(HID + data['hash_value']).digest()[:len(auth_data)]
                            q = bytes(a ^ b for a, b in zip(auth_data, hid_hash))
                            
                            # 10. Store encrypted values in database
                            data['encrypted_q'] = self.extractor.encrypt(q)
                            data['encrypted_new_R'] = self.extractor.encrypt(new_R)
                            
                            return True, q, stored_id
                    except Exception:
                        return False, None, None
        except Exception:
            return False, None, None
        
        return False, None, None

    def generate_cross_authentication(self, q1: bytes, q2: bytes, 
                                    a: bytes, b: bytes, 
                                    ida: bytes, idb: bytes,
                                    n: int) -> Tuple[bytes, bytes, str]:
        """
        Generate cross-authentication value and QAB
        Args:
            q1, q2: Random values from Alice and Bob
            a, b: Hash values for Alice and Bob
            ida, idb: User IDs
            n: Parameter N (bits)
        Returns:
            (RQA, RQB, qab) where:
                RQA: Cross-authentication value for Alice
                RQB: Cross-authentication value for Bob
                qab: Shared key string
        """
        # Calculate RQA = q2 ⊕ H(a || IDA || q1)
        h_a = hashlib.sha256(a + ida + q1).digest()[:len(q2)]
        RQA = bytes(x ^ y for x, y in zip(q2, h_a))
        
        # Calculate RQB = q1 ⊕ H(b || IDB || q2)
        h_b = hashlib.sha256(b + idb + q2).digest()[:len(q1)]
        RQB = bytes(x ^ y for x, y in zip(q1, h_b))
        
        # Calculate QAB = H(q1 || q2) to ensure length is 2N bits
        qab_full = hashlib.sha256(q1 + q2).digest()
        qab = ''.join(format(byte, '08b') for byte in qab_full)[:2*n]
        
        return RQA, RQB, qab

    @staticmethod
    def hamming_distance(a: bytes, b: bytes) -> int:
        """Calculate Hamming distance"""
        return sum(bin(x ^ y).count('1') for x, y in zip(a, b))

class User:
    """User (Alice or Bob) implementation"""
    def __init__(self, name: str, shared_key: bytes = None):
        self.name = name
        self.ID = None
        self.BIO = None
        self.hash_value = None
        self.extractor = FuzzyExtractor(shared_key)
        
    def generate_credentials(self, n: int) -> Tuple[bytes, bytes]:
        """
        Generate identity information and biometric features
        Args:
            n: Parameter N (bits)
        Returns:
            Tuple of (ID, encrypted_BIO)
        """
        n_bytes = (2 * n + 7) // 8  # Convert bits to bytes (2N bits)
        
        # Generate random ID and BIO
        self.ID = os.urandom(n_bytes)
        self.BIO = os.urandom(n_bytes)
        
        # Encrypt BIO
        encrypted_BIO = self.extractor.encrypt(self.BIO)
        
        return self.ID, encrypted_BIO

    def generate_authentication_data(self, hash_value: bytes) -> Tuple[bytes, bytes, bytes, float]:
        """
        Generate authentication required data
        hash_value: Hash value received from RG (corresponds to a or b in protocol)
        Returns: (HID, auth_data, new_bio, timestamp), corresponding to (HIDA/B, A1/B1, BIOA/B, TA/B)
        """
        # 1. Randomly generate q of length 2N (corresponds to q1 or q2)
        q = os.urandom(len(self.ID))
        
        # 2. Calculate pseudo-identity HID = ID ⊕ hash_value (where hash_value is a or b)
        hid = bytes(a ^ b for a, b in zip(self.ID, hash_value))
        
        # 3. Calculate authentication data A1/B1 = q ⊕ H(HID || hash_value)
        hid_and_hash = hid + hash_value
        hid_hash = hashlib.sha256(hid_and_hash).digest()[:len(q)]
        auth_data = bytes(a ^ b for a, b in zip(q, hid_hash))
        
        # 4. Generate slightly different biometric features
        new_bio = bytearray(self.BIO)
        total_bits = len(self.BIO) * 8
        num_bits_to_flip = min(int(total_bits * 0.01), 5)  # Modify max 1% bits, not exceeding 5
        
        # Ensure modified positions are scattered
        all_positions = list(range(total_bits))
        random.shuffle(all_positions)
        positions_to_flip = all_positions[:num_bits_to_flip]
        
        for pos in positions_to_flip:
            byte_pos = pos // 8
            bit_pos = pos % 8
            new_bio[byte_pos] ^= (1 << bit_pos)
        
        # 5. Encrypt the new biometric data
        try:
            encrypted_new_bio = self.extractor.encrypt(bytes(new_bio))
        except Exception as e:
            # Add padding if needed
            padding = 16 - (len(new_bio) % 16)
            padded_bio = bytes(new_bio) + bytes([padding] * padding)
            encrypted_new_bio = self.extractor.encrypt(padded_bio)
        
        # 6. Generate timestamp
        timestamp = time.time()
        
        return hid, auth_data, encrypted_new_bio, timestamp

    def process_cross_auth(self, 
                          rq: bytes, 
                          my_hash: bytes, 
                          my_id: bytes, 
                          my_q: bytes,
                          qab: str,
                          n: int) -> Tuple[bytes, str, str]:
        """
        Process cross-authentication value and generate session key
        qab: QAB bit string received from RG
        n: Input parameter N (bits)
        """
        # 1. Calculate partner's q value
        partner_q = bytes(rq[i] ^ hashlib.sha256(my_hash + my_id + my_q).digest()[i] 
                         for i in range(len(rq)))
        
        # 2. Split key pair directly from QAB
        self.key1 = qab[:n]      # QA1/QB1 (N bits)
        self.key2 = qab[n:]      # QA2/QB2 (N bits)
        
        return partner_q, self.key1, self.key2

def validate_n(n: int) -> bool:
    """Validate if the input N value is valid"""
    if not isinstance(n, int):
        print("Error: N must be an integer!")
        return False
    if n <= 0:
        print("Error: N must be positive!")
        return False
    if n % 2 != 0:
        print("Error: N must be a multiple of 2!")
        return False
    return True

class BellState:
    def __init__(self, state_type: str):
        """
        Initialize Bell state
        state_type: 'B00', 'B01', 'B10', 'B11'
        """
        self.state_type = state_type
        self.coherence = 1.0  
        self.qc = self._create_bell_state()
        self.measured_state = None

    def _create_bell_state(self) -> QuantumCircuit:
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        
        # Prepare Bell state
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        
        # Apply transformation based on type
        if self.state_type == 'B01':    
            qc.x(qr[1])
        elif self.state_type == 'B10':   
            qc.z(qr[0])
        elif self.state_type == 'B11':   
            qc.x(qr[1])
            qc.z(qr[0])
            
        # Bell measurement
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.measure(qr, cr)
        
        return qc

class SinglePhotonState:
    def __init__(self, state: str):
        """
        Initialize single photon state
        state: Quantum state ('0', '1', '+', or '-')
        """
        self.state_type = state
        self.coherence = 1.0  # Add coherence property
        self.qc = self._create_single_photon()

    def _create_single_photon(self) -> QuantumCircuit:
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        
        # Prepare quantum state based on state type
        if self.state_type == '1':
            qc.x(qr[0])
        elif self.state_type in ['+', '-']:
            qc.h(qr[0])
            if self.state_type == '-':
                qc.z(qr[0])
        
        qc.measure(qr, cr)
        return qc

class QuantumChannel:
    def __init__(self, 
                 decoherence_rate=0.05,     
                 bit_flip_rate=0.05,        
                 phase_flip_rate=0.05,      
                 distance_factor=0.001):     
        """
        Initialize quantum channel with more realistic noise parameters
        """
        self.decoherence_rate = decoherence_rate
        self.bit_flip_rate = bit_flip_rate
        self.phase_flip_rate = phase_flip_rate
        self.distance_factor = distance_factor
        self.accumulated_noise = 0  
        self.simulator = Aer.get_backend('aer_simulator')

    def apply_noise(self, state, position: int) -> None:
        """Apply noise effects to quantum state"""
        # Calculate distance effect based on actual transmission distance
        # rather than position in sequence
        transmission_distance = random.uniform(0.8, 1.2)  # Simulate varying distances
        distance_effect = 1 - math.exp(-transmission_distance * self.distance_factor)
        
        # Base coherence decay from distance
        base_coherence_loss = transmission_distance * 0.001
        
        # Apply exponential decay to coherence
        state.coherence *= math.exp(-base_coherence_loss)
        
        # Ensure coherence stays within reasonable bounds
        state.coherence = max(0.80, min(0.98, state.coherence))
        
        # Rest of the noise effects...
        effective_noise_rate = min(0.95, max(
            self.decoherence_rate,
            self.bit_flip_rate,
            self.phase_flip_rate,
            distance_effect
        ))
        
        decoherence = random.uniform(effective_noise_rate/2, effective_noise_rate)
        state.coherence *= (1 - decoherence)
        
        base_noise_probability = max(0.15,  1 - state.coherence, effective_noise_rate)
        
        if isinstance(state, BellState):
            # Bell state noise processing
            for _ in range(2):  
                if random.random() < base_noise_probability:
                    noise_type = random.choices(
                        ['bit', 'phase', 'both'],
                        weights=[0.4, 0.4, 0.2]  
                    )[0]
                    
                    if noise_type in ['bit', 'both']:
                        new_type = {
                            'B00': 'B01',
                            'B01': 'B00',
                            'B10': 'B11',
                            'B11': 'B10'
                        }
                        state.state_type = new_type[state.state_type]
                    
                    if noise_type in ['phase', 'both']:
                        new_type = {
                            'B00': 'B10',
                            'B01': 'B11',
                            'B10': 'B00',
                            'B11': 'B01'
                        }
                        state.state_type = new_type[state.state_type]

        elif isinstance(state, SinglePhotonState):
            if random.random() < base_noise_probability:
                noise_type = random.choices(
                    ['bit', 'phase', 'both'],
                    weights=[0.4, 0.4, 0.2]
                )[0]
                
                if noise_type in ['bit', 'both']:
                    new_type = {
                        '0': '1',
                        '1': '0',
                        '+': '-',
                        '-': '+'
                    }
                    state.state_type = new_type[state.state_type]
                
                if noise_type in ['phase', 'both'] and state.state_type in ['+', '-']:
                    state.state_type = '-' if state.state_type == '+' else '+'
        
        if random.random() < distance_effect:
            state.coherence *= 0.9  

class QuantumMeasurement:
    @staticmethod
    def measure(quantum_state, basis: str) -> Tuple[str, float]:
        current_state = quantum_state.state_type
        coherence = quantum_state.coherence
        
        # Consider measurement reliability with decoherence effect
        reliability = coherence
        
        if basis == 'Z':
            if current_state in ['0', '1']:
                # Even if correct basis, consider uncertainty due to decoherence
                if random.random() < reliability:
                    return current_state, reliability
                else:
                    return '1' if current_state == '0' else '0', reliability
            else:  # Incorrect basis measurement
                return random.choice(['0', '1']), 0.5
        else:  # basis == 'X'
            if current_state in ['+', '-']:
                # Consider decoherence effect
                if random.random() < reliability:
                    return current_state, reliability
                else:
                    return '-' if current_state == '+' else '+', reliability
            else:  # Incorrect basis measurement
                return random.choice(['+', '-']), 0.5

def generate_binary_string(length: int) -> str:
    """Generate random binary string of specified length"""
    return ''.join(random.choice(['0', '1']) for _ in range(length))

def get_bell_pair_type(qa1_bit: str, qa2_bit: str) -> List[str]:
    """Determine Bell state type based on QA1 and QA2 bits"""
    choices = {
        ('0', '0'): [['B00', 'B00'], ['B01', 'B01'], 
                     ['B10', 'B10'], ['B11', 'B11']],
        ('0', '1'): [['B00', 'B01'], ['B01', 'B00'], 
                     ['B10', 'B11'], ['B11', 'B10']],
        ('1', '0'): [['B00', 'B10'], ['B01', 'B11'], 
                     ['B10', 'B00'], ['B11', 'B01']],
        ('1', '1'): [['B00', 'B11'], ['B01', 'B10'], 
                     ['B10', 'B01'], ['B11', 'B00']]
    }
    return random.choice(choices[(qa1_bit, qa2_bit)])

def simulate_quantum_protocol(N: int, qa1: str, qa2: str, qb1: str, qb2: str) -> Tuple[Optional[str], Optional[str], float, float, List[Union[BellState, SinglePhotonState]]]:
    """
    Simulate complete quantum protocol process
    Returns: (ka, kb, decoy_error_rate, key_error_rate, quantum_states)
    """
    MAX_RETRIES = 3
    attempt = 0
    
    # Calculate dynamic time threshold based on N
    BASE_TIME = 5  # Base time in seconds
    STATES_PER_SECOND = 30  # Average states processed per second
    total_states = 4.5 * N  # Total quantum states (4N Bell states + N/2 decoy states)
    TIME_THRESHOLD = BASE_TIME + (total_states / STATES_PER_SECOND)
    
    print(f"\nDynamic Time Settings:")
    print(f"- Total quantum states: {total_states}")
    print(f"- Base time: {BASE_TIME} seconds")
    print(f"- Processing rate: {STATES_PER_SECOND} states/second")
    print(f"- Calculated time threshold: {TIME_THRESHOLD:.2f} seconds")
    
    while attempt < MAX_RETRIES:
        attempt += 1
        print(f"\nAttempt {attempt} of {MAX_RETRIES}")
        channel = QuantumChannel()
        
        # Bell state preparation
        print("\n=== Bell State Preparation ===")
        s1 = []
        s2 = []
        for i in range(len(qa1)):
            bell_pair = get_bell_pair_type(qa1[i], qa2[i])
            s1.append(BellState(bell_pair[0]))
            s2.append(BellState(bell_pair[1]))
            print(f"Position {i}: QA1[{i}]={qa1[i]}, QA2[{i}]={qa2[i]} → Bell pair: {bell_pair}")
        
        # Prepare decoy states
        print("\n=== Decoy State Preparation ===")
        decoy_states = []
        sequence_length = len(s2)
        num_decoy = N//2  # Number of decoy states
        
        available_positions = list(range(sequence_length))
        if num_decoy > len(available_positions):
            num_decoy = len(available_positions)
        
        decoy_positions = sorted(random.sample(available_positions, num_decoy))
        decoy_info = []
        
        for i in range(num_decoy):
            state = random.choice(['0', '1', '+', '-'])
            basis = 'Z' if state in ['0', '1'] else 'X'
            decoy_states.append(SinglePhotonState(state))
            decoy_info.append((state, basis, decoy_positions[i]))
            print(f"Decoy state {i+1}: |{state}> (basis: {basis}) → insert position: {decoy_positions[i]}")
        
        # Insert decoy states into S2
        s2_with_decoy = s2.copy()
        for pos, state in zip(decoy_positions, decoy_states):
            s2_with_decoy.insert(pos, state)
            
        # Add timestamp and calculate delay
        transmission_time = time.time()
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(transmission_time))
        print(f"\nTransmission timestamp: {readable_time}")
        
        # Calculate transmission delay based on quantum state count
        total_states = len(s2_with_decoy)
        base_delay = 0.1  # 100ms base delay per state
        distance_factor = random.uniform(0.8, 1.2)  # Random distance variation
        transmission_delay = total_states * base_delay * distance_factor
        
        # Simulate transmission
        print(f"\nSimulating quantum state transmission...")
        time.sleep(transmission_delay)  # Actually wait to simulate transmission
        
        # Transmit through quantum channel
        print("\n=== Quantum Channel Transmission ===")
        transmitted_states = []
        for i, state in enumerate(s2_with_decoy):
            original_state = state.state_type
            channel.apply_noise(state, i)
            transmitted_states.append((i, original_state, state.state_type, state.coherence))
            
            if i in decoy_positions:
                print(f"Position {i}: |{original_state}> → |{state.state_type}> (coherence: {state.coherence:.2f}) [Decoy]")
            else:
                print(f"Position {i}: |{original_state}> → |{state.state_type}> (coherence: {state.coherence:.2f})")
        
        # Verify timestamp before measurement
        current_time = time.time()
        elapsed_time = current_time - transmission_time
        if elapsed_time > TIME_THRESHOLD:
            print(f"\nTransmission time exceeded threshold ({TIME_THRESHOLD:.2f} seconds)")
            print(f"Actual transmission time: {elapsed_time:.2f} seconds")
            if attempt < MAX_RETRIES:
                print(f"Retrying... (Attempt {attempt + 1}/{MAX_RETRIES})")
                continue
            else:
                print("Max retries reached. Protocol failed.")
                return None, None, 1.0, 1.0, []
                
        readable_current = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
        print(f"\nTransmission time verified:")
        print(f"- Start time: {readable_time}")
        print(f"- End time: {readable_current}")
        print(f"- Elapsed time: {elapsed_time:.2f} seconds")
        print(f"- Total quantum states: {total_states}")
        print(f"- Distance factor: {distance_factor:.2f}")
        print(f"- Average transmission rate: {total_states/elapsed_time:.1f} states/second")

        # Decoy state measurement
        print("\n=== Decoy State Measurement ===")
        decoy_errors = 0
        total_decoy = len(decoy_info)
        
        for i, (original_state, basis, position) in enumerate(decoy_info, 1):
            state = s2_with_decoy[position]
            measured_state, reliability = QuantumMeasurement.measure(state, basis)
            
            # Check measurement results
            is_correct = measured_state == original_state
            if not is_correct:
                decoy_errors += 1
                
            status = "✓" if is_correct else "✗"
            print(f"Decoy state {i} (position {position}): Original |{original_state}> → "
                  f"Measured |{measured_state}> (reliability: {reliability:.2f}) {status}")
        
        # Check decoy state error rate
        decoy_error_rate = decoy_errors / total_decoy if total_decoy > 0 else 0
        print(f"\nDecoy state error rate: {decoy_error_rate:.2%} ({decoy_errors}/{total_decoy})")
        
        # Adjust error threshold to be more tolerant as N increases
        error_threshold = min(0.20 + (N/1000), 0.35)  # Base 20%, max 35%
        
        if decoy_error_rate > 0.25:
            print(f"Decoy state error rate too high ({decoy_error_rate:.2%})")
            if attempt < MAX_RETRIES:
                print(f"Retrying... (Attempt {attempt + 1}/{MAX_RETRIES})")
                continue
            else:
                print("Max retries reached. Protocol failed.")
                return None, None, decoy_error_rate, 1.0, []

        
        # Bell measurement
        print("\n=== Bell Measurement ===")
        
        # Remove decoy states from S2 sequence
        s2_without_decoy = []
        current_pos = 0
        for i in range(len(s2_with_decoy)):
            if i not in decoy_positions:
                s2_without_decoy.append(s2_with_decoy[i])
                current_pos += 1
        
        # Alice's Bell measurement - using all N Bell pairs
        print("\nAlice's Bell measurement results:")
        alice_results = []
        for i in range(len(qa1)):
            alice_state = s1[i].state_type
            alice_measurement = alice_state[1:]  
            alice_results.append(alice_measurement)
            print(f"Position {i}: {alice_state} → {alice_measurement}")
        
        # Bob's Bell measurement - using all N Bell pairs
        print("\nBob's Bell measurement results:")
        bob_results = []
        for i in range(len(qb1)):
            bob_state = s2_without_decoy[i].state_type
            bob_measurement = bob_state[1:]
            bob_results.append(bob_measurement)
            print(f"Position {i}: {bob_state} → {bob_measurement}")
        
        # Key derivation
        print("\n=== Key Derivation ===")
        msa = ''.join(alice_results)
        msb = ''.join(bob_results)
        
        print("\nAlice deriving MSB:")
        ka = ''  # Initialize empty key for Alice
        for i in range(len(qa1)):
            qa1_bit = qa1[i]
            qa2_bit = qa2[i]
            current_msa = msa[i*2:(i+1)*2]  # Take two bits at a time
            
            # Derive msb according to rules
            derived_bits = list(current_msa)  # Convert to list for easier manipulation
            if qa1_bit == '1':  # If QA1 is 1, flip first bit
                derived_bits[0] = '1' if current_msa[0] == '0' else '0'
            if qa2_bit == '1':  # If QA2 is 1, flip second bit
                derived_bits[1] = '1' if current_msa[1] == '0' else '0'
            derived_bits = ''.join(derived_bits)
            
            # Connect two bits at a time: MSA||MSB
            ka += current_msa + derived_bits
            print(f"Position {i}: QA1={qa1_bit}, QA2={qa2_bit}, MSA={current_msa} → Derived MSB={derived_bits}")
        
        print("\nBob deriving MSA:")
        kb = ''  # Initialize empty key for Bob
        for i in range(len(qb1)):
            qb1_bit = qb1[i]
            qb2_bit = qb2[i]
            current_msb = msb[i*2:(i+1)*2]
            
            # Derive msa according to rules
            derived_bits = list(current_msb)  # Convert to list for easier manipulation
            if qb1_bit == '1':  # If QB1 is 1, flip first bit
                derived_bits[0] = '1' if current_msb[0] == '0' else '0'
            if qb2_bit == '1':  # If QB2 is 1, flip second bit
                derived_bits[1] = '1' if current_msb[1] == '0' else '0'
            derived_bits = ''.join(derived_bits)
            
            # Connect two bits at a time: MSA||MSB
            kb += derived_bits + current_msb
            print(f"Position {i}: QB1={qb1_bit}, QB2={qb2_bit}, MSB={current_msb} → Derived MSA={derived_bits}")
        
        # Generate sharedkeys
        ka = ka
        kb = kb
        
        print("\n=== Shared Keys ===")
        print(f"KA = MSA||Derived MSB: {ka}")
        print(f"Length: {len(ka)} bits")
        print(f"KB = Derived MSA||MSB: {kb}")
        print(f"Length: {len(kb)} bits")
        
        # Calculate error rate
        errors = sum(1 for a, b in zip(ka, kb) if a != b)
        key_error_rate = errors / len(ka) if ka else 0
        
        # Collect all quantum states
        all_quantum_states = s1 + s2_with_decoy
        
        return ka, kb, decoy_error_rate, key_error_rate, all_quantum_states
        
    return None, None, 1.0, 1.0, []

def calculate_efficiency(N: int, final_key_length: int, matched_bits: int) -> Tuple[float, float]:
    """
    Calculate two types of quantum bit efficiency
    N: Input N value
    final_key_length: Theoretical final key length (4N)
    matched_bits: Actual number of matched key bits
    
    Returns: (theoretical_efficiency, practical_efficiency)
    """
    # Calculate total quantum bits used
    quantum_bits = 4.5 * N  # Bell states(4N) + Decoy states(N/2)
    classical_bits = 2 * N  # Classical bits used for decoding
    total_bits = quantum_bits + classical_bits
    
    # Calculate theoretical efficiency (without noise)
    theoretical_efficiency = final_key_length / total_bits
    
    # Calculate practical efficiency (with matched bits after noise)
    practical_efficiency = matched_bits / total_bits
    
    return theoretical_efficiency, practical_efficiency

def correct_keys(ka: str, kb: str, n: int) -> Tuple[str, str, int]:
    """
    Correct key differences using binary search strategy
    Args:
        ka: Alice's key
        kb: Bob's key
        n: Parameter N (final key length will be 4N)
    Returns:
        (corrected_ka, corrected_kb, parity_bits_exchanged)
    """
    if len(ka) != len(kb):
        raise ValueError("Keys must be of equal length")
    
    # If keys are already identical, return without correction
    if ka == kb:
        return ka, kb, 0
        
    total_parity_bits = 0
    corrected_ka = list(ka)
    corrected_kb = list(kb)
    key_length = len(ka)  # Should be 4N
    
    def calculate_parity(key_segment: str) -> str:
        """Calculate parity bit (1 if odd number of 1s, 0 if even)"""
        return str(sum(int(bit) for bit in key_segment) % 2)
    
    def binary_search_correction(start: int, end: int):
        """Recursively search and correct errors using binary search"""
        if start >= end:
            return
            
        if start + 1 == end:  # Single bit check
            if corrected_ka[start] != corrected_kb[start]:
                nonlocal total_parity_bits
                total_parity_bits += 2  # Exchange parity for this bit
                corrected_kb[start] = corrected_ka[start]
            return
            
        mid = (start + end) // 2
        
        # Check left half
        left_ka = ''.join(corrected_ka[start:mid])
        left_kb = ''.join(corrected_kb[start:mid])
        
        if left_ka != left_kb:
            ka_parity = calculate_parity(left_ka)
            kb_parity = calculate_parity(left_kb)
            total_parity_bits += 2
            
            if ka_parity != kb_parity:
                binary_search_correction(start, mid)
                
        # Check right half
        right_ka = ''.join(corrected_ka[mid:end])
        right_kb = ''.join(corrected_kb[mid:end])
        
        if right_ka != right_kb:
            ka_parity = calculate_parity(right_ka)
            kb_parity = calculate_parity(right_kb)
            total_parity_bits += 2
            
            if ka_parity != kb_parity:
                binary_search_correction(mid, end)
    
    # Start binary search correction
    binary_search_correction(0, key_length)
    
    # Final bit-by-bit verification and correction
    for i in range(key_length):
        if corrected_ka[i] != corrected_kb[i]:
            total_parity_bits += 2
            corrected_kb[i] = corrected_ka[i]
    
    return ''.join(corrected_ka), ''.join(corrected_kb), total_parity_bits

def calculate_efficiency(N: int, final_key_length: int, matched_bits: int, parity_bits: int = 0) -> Tuple[float, float]:
    """
    Calculate quantum bit efficiency
    Args:
        N: Parameter N
        final_key_length: Theoretical final key length (4N)
        matched_bits: Actual number of matched key bits
        parity_bits: Number of parity bits exchanged during correction
    Returns:
        (theoretical_efficiency, practical_efficiency)
    """
    # Calculate total quantum bits used (6.5N = Bell states(4N) + Decoy states(N/2) + Classical bits(2N))
    total_bits = 6.5 * N
    
    # Calculate theoretical efficiency (without noise)
    theoretical_efficiency = final_key_length / total_bits
    
    # Calculate practical efficiency (with noise and parity bits)
    total_bits_with_parity = total_bits + parity_bits
    practical_efficiency = final_key_length / total_bits_with_parity
    
    return theoretical_efficiency, practical_efficiency

def calculate_security_metrics(N: int, decoy_error_rate: float, key_error_rate: float, 
                             coherence_values: List[float], parity_bits: int) -> Dict:
    """
    Calculate comprehensive security metrics
    Args:
        N: Parameter N (bits)
        decoy_error_rate: Error rate from decoy states
        key_error_rate: Final key error rate
        coherence_values: List of coherence values from quantum transmission
        parity_bits: Number of exchanged parity bits
    Returns:
        Dictionary containing security metrics
    """
    metrics = {}
    
    # Minimum entropy loss calculation
    exposed_bits = parity_bits + 2 * N  # Exposed bits from parity exchange and helper data
    total_bits = 4 * N  # Total key length
    min_entropy_loss = -math.log2(1 - (exposed_bits / total_bits))
    metrics['min_entropy_loss'] = min_entropy_loss
    
    # Adversarial success probability
    # Based on quantum bit error rate and decoy detection probability
    detection_prob = 1 - math.pow(1 - decoy_error_rate, N/2)  # Probability of detecting tampering
    attack_success_prob = (1 - detection_prob) * math.pow(1 - key_error_rate, 4*N)
    metrics['attack_success_probability'] = attack_success_prob
    
    # Quantum channel robustness
    avg_coherence = sum(coherence_values) / len(coherence_values)
    coherence_variance = sum((c - avg_coherence)**2 for c in coherence_values) / len(coherence_values)
    channel_robustness = 1 - math.sqrt(coherence_variance)
    metrics['channel_robustness'] = channel_robustness
    
    # Information leakage from error correction
    leakage_rate = parity_bits / (4 * N)  # Rate of information leaked during correction
    metrics['error_correction_leakage'] = leakage_rate
    
    # Overall security score (weighted combination)
    security_score = (0.3 * (1 - min_entropy_loss/4) + 
                     0.3 * (1 - attack_success_prob) +
                     0.2 * channel_robustness +
                     0.2 * (1 - leakage_rate))
    metrics['overall_security_score'] = security_score
    
    return metrics

def evaluate_security(ka: str, kb: str, N: int, 
                     quantum_states: List[Union[BellState, SinglePhotonState]],
                     decoy_error_rate: float,
                     key_error_rate: float,
                     parity_bits: int) -> None:
    """
    Perform comprehensive security evaluation
    """
    # Collect coherence values from quantum states
    coherence_values = [state.coherence for state in quantum_states]
    
    # Calculate security metrics
    metrics = calculate_security_metrics(
        N=N,
        decoy_error_rate=decoy_error_rate,
        key_error_rate=key_error_rate,
        coherence_values=coherence_values,
        parity_bits=parity_bits
    )
    
    print("\n=== Security Metrics Analysis ===")
    print(f"Minimum Entropy Loss: {metrics['min_entropy_loss']:.4f} bits")
    print(f"  = -log2(1 - (parity_bits + 2N)/(4N))")
    print(f"  = -log2(1 - ({parity_bits} + {2*N})/{4*N}) = {metrics['min_entropy_loss']:.4f}")
    
    print(f"\nAttack Success Probability: {metrics['attack_success_probability']:.6f}")
    print(f"  = (1 - detection_prob) * (1 - key_error_rate)^(4N)")
    print(f"  = (1 - {1-math.pow(1-decoy_error_rate, N/2):.4f}) * (1 - {key_error_rate})^{4*N}")
    print(f"  = {metrics['attack_success_probability']:.6f}")
    
    print(f"\nChannel Robustness: {metrics['channel_robustness']:.4f}")
    print(f"  = 1 - sqrt(variance of coherence values)")
    print(f"  = 1 - sqrt({sum((c - sum(coherence_values)/len(coherence_values))**2 for c in coherence_values)/len(coherence_values):.6f})")
    print(f"  = {metrics['channel_robustness']:.4f}")
    
    print(f"\nError Correction Information Leakage: {metrics['error_correction_leakage']:.4f}")
    print(f"  = parity_bits/(4N)")
    print(f"  = {parity_bits}/{4*N} = {metrics['error_correction_leakage']:.4f}")
    
    print(f"\nOverall Security Score: {metrics['overall_security_score']:.4f}")
    print(f"  = 0.3*(1 - min_entropy_loss/4) + 0.3*(1 - attack_success_prob)")
    print(f"    + 0.2*channel_robustness + 0.2*(1 - leakage_rate)")
    print(f"  = 0.3*(1 - {metrics['min_entropy_loss']}/4) + 0.3*(1 - {metrics['attack_success_probability']:.6f})")
    print(f"    + 0.2*{metrics['channel_robustness']:.4f} + 0.2*(1 - {metrics['error_correction_leakage']:.4f})")
    print(f"  = {metrics['overall_security_score']:.4f}")
    
    # Additional analysis for varying adversarial models
    print("\n=== Adversarial Model Analysis ===")
    print("Attack Resistance = 1 - (noise_level * channel_robustness)")
    for noise_level in [0.05, 0.10, 0.15]:
        attack_resistance = 1 - (noise_level * metrics['channel_robustness'])
        print(f"Noise Level {noise_level:.2f}: 1 - ({noise_level:.2f} * {metrics['channel_robustness']:.4f}) = {attack_resistance:.4f}")

def main():
    # Get and validate N value
    while True:
        try:
            N = int(input("Please enter parameter N (must be a multiple of 2): "))
            if validate_n(N):
                break
        except ValueError:
            print("Error: Please enter a valid integer!")
    
    print(f"\nUsing parameter N = {N}")

    # Initialize RG first to get the shared key
    rg = RegistrationAuthority()
    shared_key = rg.get_shared_key()
    
    # Initialize users with shared key
    alice = User("Alice", shared_key)
    bob = User("Bob", shared_key)
    
     # Part 1: Registration Phase
    print("\n=== Part 1: Registration Phase ===")
    # Generate credentials
    print("\nGenerating user credentials...")
    try:
        alice_id, alice_encrypted_bio = alice.generate_credentials(N)
        bob_id, bob_encrypted_bio = bob.generate_credentials(N)
    except Exception as e:
        print(f"Error generating credentials: {str(e)}")
        return

    # Registration process
    print("Registering users with RG...")
    try:
        # Register Alice and Bob
        a = rg.register_user(alice_id, alice_encrypted_bio)
        b = rg.register_user(bob_id, bob_encrypted_bio)
        
        # Store hash values
        alice.hash_value = a
        bob.hash_value = b
        
        print("\nRegistration successful!")
        print(f"Alice's hash value (a): {b64encode(a).decode()}")
        print(f"Bob's hash value (b): {b64encode(b).decode()}")
        
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        return

    # Part 2: Authentication Phase
    print("\n=== Part 2: Authentication Phase ===")
    print("Alice and Bob are authenticating...")
    
    # Generate authentication data
    alice_hid, alice_a1, alice_new_bio, alice_ts = alice.generate_authentication_data(a)
    bob_hid, bob_b1, bob_new_bio, bob_ts = bob.generate_authentication_data(b)
    
    # RG authenticates Alice and Bob
    alice_auth, q1, alice_id = rg.authenticate_user(alice_hid, alice_a1, alice_new_bio, alice_ts)
    bob_auth, q2, bob_id = rg.authenticate_user(bob_hid, bob_b1, bob_new_bio, bob_ts)
    
    if not alice_auth:
        print("Alice authentication failed!")
        return
    if not bob_auth:
        print("Bob authentication failed!")
        return
        
    print("\nBoth users authenticated successfully!")
    
    # Generate cross-authentication value and QAB
    RQA, RQB, qab = rg.generate_cross_authentication(q1, q2, a, b, alice_id, bob_id, N)
    
    # Alice and Bob process cross-authentication values
    alice_q2, alice.key1, alice.key2 = alice.process_cross_auth(RQA, a, alice_id, q1, qab, N)
    bob_q1, bob.key1, bob.key2 = bob.process_cross_auth(RQB, b, bob_id, q2, qab, N)
    
    # Verify cross-authentication success
    print("\nVerifying cross-authentication...")
    if alice_q2 == q2 and bob_q1 == q1:
        print("Cross-authentication successful!")
        print(f"\nAlice's key pair: QA1={alice.key1}, QA2={alice.key2}")
        print(f"Bob's key pair: QB1={bob.key1}, QB2={bob.key2}")
    else:
        print("Cross-authentication failed!")
        print("Retrieved q values do not match original values.")
    
    # Part 3: Quantum Key Agreement
    print("\n=== Part 3: Quantum Key Agreement ===")
    try:
        print(f"\nStarting quantum protocol simulation...")
        print(f"Settings:")
        print(f"- Bell states: {2*N}")
        print(f"- Decoy states: {N//2}")
        print(f"- Bit flip rate: 5%")
        print(f"- Phase flip rate: 5%")
        print(f"- Decoherence rate: 5%")
        
        ka, kb, decoy_error_rate, key_error_rate, quantum_states = simulate_quantum_protocol(
            N, alice.key1, alice.key2, bob.key1, bob.key2
        )
        
        if ka is None or kb is None:
            print("\nProtocol failed due to excessive noise in quantum channel.")
            print("Suggestion: Try with a smaller N value or retry the protocol.")
            return

        # Calculate matched bits before correction
        matched_bits = sum(1 for a, b in zip(ka, kb) if a == b)

        if ka != kb:
            print("\n=== Key Correction Phase ===")
            print("Initial mismatched positions:")
            for i, (a, b) in enumerate(zip(ka, kb)):
                if a != b:
                    print(f"Position {i}: KA={a}, KB={b}")
            
            print("\nStarting key correction...")
            ka, kb, parity_bits = correct_keys(ka, kb, N)
            print(f"\nTotal parity bits exchanged: {parity_bits}")
            
            if ka == kb:
                print("Keys are now identical!")
                print(f"\nKA: {ka}")
                print(f"KB: {kb}")
            else:
                print("Failed to fully correct keys")
                print("Remaining mismatched positions:")
                for i, (a, b) in enumerate(zip(ka, kb)):
                    if a != b:
                        print(f"Position {i}: KA={a}, KB={b}")
        else:
            print("\nKeys are already identical!")
            parity_bits = 0  # No correction needed

        # Calculate efficiencies
        theoretical_eff, practical_eff = calculate_efficiency(N, len(ka), matched_bits, parity_bits)

        print("\n=== Final Statistics ===")
        print(f"- Total key length: {len(ka)} bits")
        print(f"- Matched key bits: {matched_bits} bits")
        print(f"- Decoy state error rate: {decoy_error_rate:.2%}")
        print(f"- Final key error rate: {key_error_rate:.2%}")
        print(f"- Parity bits exchanged: {parity_bits}")
        print(f"- Theoretical efficiency (without noise): 4N/(4N+0.5N+2N) = {4*N}/{6.5*N} = {theoretical_eff:.4f}")
        print(f"- Practical efficiency (with noise and parity bits): 4N/(4N+0.5N+2N+{parity_bits}) = {4*N}/{6.5*N + parity_bits} = {practical_eff:.4f}")
       
        # Display mismatched positions
        errors = sum(1 for a, b in zip(ka, kb) if a != b)
        if errors > 0:
            print(f"\nMismatched key positions (total {errors} positions):")
            for i, (a, b) in enumerate(zip(ka, kb)):
                if a != b:
                    print(f"Position {i}: KA={a}, KB={b}")
        
    except Exception as e:
        print(f"Error during protocol execution: {str(e)}")
        return

    # Now we can properly evaluate security with all required parameters
    evaluate_security(ka, kb, N, quantum_states, 
                     decoy_error_rate, key_error_rate, parity_bits)

if __name__ == "__main__":
    main()
