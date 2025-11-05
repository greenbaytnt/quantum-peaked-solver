import gc
import logging

import re
import time
import os
import torch
from typing import Any, Dict, Optional

import numpy as np
from pytket.qasm import circuit_from_qasm_str
from pytket.extensions.cutensornet.general_state import GeneralState
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, U3Gate
from qiskit.qasm2 import CustomInstruction, loads
from qiskit_aer import AerSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



class DefaultPeakedSolver:
    def _wait_for_gpu_memory(self, required_memory_gb: float = 2.0, target_gpu: int = 3) -> bool:
        """
        Wait until GPU memory is available on the target GPU.
        
        Args:
            required_memory_gb: Required GPU memory in GB
            target_gpu: Target GPU index (default: 3 for 4th GPU)
            
        Returns:
            True if GPU memory is available
        """
        try:
                
            while True:
                # Check memory for the target GPU only
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                # Get actual memory usage using nvidia-ml-py or direct CUDA calls
                try:
                    # Use nvidia-ml-py for accurate memory tracking
                    import pynvml
                    pynvml.nvmlInit()
                    cuda_device = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
                    handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_device))
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_memory_gb = mem_info.used / (1024**3)
                    free_memory_gb = mem_info.free / (1024**3)
                except ImportError:
                    # Fallback to PyTorch method (less accurate)
                    allocated = torch.cuda.memory_allocated(0)
                    cached = torch.cuda.memory_reserved(0)
                    used_memory_gb = (allocated + cached) / (1024**3)
                    free_memory_gb = gpu_memory_gb - used_memory_gb
                except Exception:
                    # If nvidia-ml-py fails, use PyTorch
                    allocated = torch.cuda.memory_allocated(0)
                    cached = torch.cuda.memory_reserved(0)
                    used_memory_gb = (allocated + cached) / (1024**3)
                    free_memory_gb = gpu_memory_gb - used_memory_gb
                
                logging.info(f"GPU {0} memory: {free_memory_gb:.2f}GB free / {gpu_memory_gb:.2f}GB total (used: {used_memory_gb:.2f}GB)")
                
                if free_memory_gb >= required_memory_gb:
                    logging.info(f"GPU {0} has enough memory: {free_memory_gb:.2f}GB >= {required_memory_gb}GB")
                    return True
                
                logging.info(f"Waiting for GPU {0} memory... (need {required_memory_gb}GB)")
                time.sleep(5)
            
        except Exception as e:
            logging.error(f"Error checking GPU memory: {e}")
            return True  # Continue with CPU if GPU check fails
    
    def _clear_gpu_memory(self):
        """Clear GPU memory and cache."""
        try:
            # Force garbage collection multiple times
            for _ in range(8):
                gc.collect()
            
            try:
                if torch.cuda.is_available():
                    # Clear cache for ALL available GPUs
                    device_count = torch.cuda.device_count()
                    for device_id in range(device_count):
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats(device_id)
                        torch.cuda.reset_accumulated_memory_stats(device_id)
                    
                    # Force garbage collection again after CUDA operations
                    for _ in range(8):
                        gc.collect()
                    
                    # Final cache clear for all devices
                    for device_id in range(device_count):
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Reset to device 0
                    torch.cuda.set_device(0)
                    
                    logging.info(f"GPU cache cleared successfully for {device_count} device(s)")
                    
            except ImportError:
                pass
            except Exception as e:
                logging.error(f"Error clearing GPU memory: {e}")
                
        except Exception as e:  # nosec
            logging.error(f"Error in _clear_gpu_memory: {e}")
            pass
            
    
    def _estimate_memory_requirement(self, num_qubits: int) -> float:
        """
        Estimate memory requirement for statevector simulation.
        
        Args:
            num_qubits: Number of qubits in the circuit
            
        Returns:
            Estimated memory requirement in GB
        """
        # Statevector size: 2^n_qubits complex numbers
        # Each complex number: 16 bytes (8 bytes real + 8 bytes imaginary)
        statevector_size = 2 ** num_qubits
        memory_bytes = statevector_size * 16  # 16 bytes per complex number
        memory_gb = memory_bytes / (1024**3)
        
        # Add overhead for processing (2x the statevector size)
        total_memory_gb = memory_gb
        
        return total_memory_gb

    def solve(self, qasm: str, cid: str, validator_hotkey: str, difficulty_level: float = 0.0) -> str:
        """
        Solve a quantum circuit to find the peaked bitstring.

        Strategy:
        - ≤32 qubits: GPU-accelerated statevector (if available)
        - >32 qubits: Try MPS first, fallback to CPU statevector

        Args:
            qasm: QASM string of the circuit

        Returns:
            Most probable bitstring, or empty string if failed
        """
        try:
            start_time = time.time()
            num_qubits = self._count_qubits(qasm)

            # logging.info(f"Solving circuit with {num_qubits} qubits")
            # logging.info(f"==========Qasm: {len(qasm)}")
            # logging.info(f"==========cid: {cid}")
            # logging.info(f"==========validator_hotkey: {validator_hotkey}")
            # logging.info(f"==========difficulty_level: {difficulty_level}")

            if num_qubits > 32:
                devide_num = 31
                split_num = 8
                parsed_qasm = self.parse_qasm(qasm, devide_num, under=True)
                result_under, bitstrings_under = self._run(parsed_qasm, devide_num)
                logging.info(f"bitstrings_under:\n{bitstrings_under}")
                logging.info(f"result under:{result_under}")
                
                
                parsed_qasm = self.parse_qasm(qasm, devide_num, under=False)
                result_over, bitstrings_over = self._run(parsed_qasm, devide_num)
                logging.info(f"bitstrings_over:\n{bitstrings_over}")
                logging.info(f"result over: {result_over}")
                bitstrings = {}
                for i, bitstring_under in enumerate(bitstrings_under):
                    for j,bitstring_over in enumerate(bitstrings_over):
                        for k in range(num_qubits - split_num * 2 + 1):
                            added_bitstring = bitstring_under['bitstring'][:split_num+k] + bitstring_over['bitstring'][split_num+k-num_qubits+devide_num:]
                            bitstrings[added_bitstring] = bitstrings.get(added_bitstring, 0) + 1
                bitstrings = sorted(bitstrings.items(), key=lambda x: x[1], reverse=True)
                
                self._wait_for_gpu_memory(18)
                amplitudes = self.calculate_amplitude_from_qasm(qasm, bitstrings)
                if not amplitudes:
                    return ""
                target_state = amplitudes[0][0]
                logging.info(f"amplitudes:\n{amplitudes}")
                logging.info(f"target state: {target_state}")
                # logging.info(f"bitstrings:\n{bitstrings}")
                logging.info(f"Time taken: {time.time() - start_time}")
                logging.info("=" * 50 + "\n")
                del result_under
                del result_over
                del bitstrings_under
                del bitstrings_over
                del parsed_qasm
                del amplitudes
                del bitstrings
                self._clear_gpu_memory()
                if target_state:
                    return target_state

                logging.info("MPS failed, falling back to standard simulation")

            result, bitstrings = self._run(qasm)
            logging.info(f"result: {result}")
            logging.info(f"bitstrings:\n{bitstrings}")
            logging.info(f"Time taken: {time.time() - start_time}")
            del bitstrings
            self._clear_gpu_memory()
            return result

        except Exception as e:
            logging.error(f"Circuit solving failed: {e}")
            self._clear_gpu_memory()
            return ""
        finally:
            # Always clear memory after processing, regardless of success or failure
            self._clear_gpu_memory()

    def _run(self, qasm: str, devide_num: int = 30) -> tuple:
        for device in ["GPU", "CPU"]:
            sim = None
            statevector = None
            processor = None
            result = None
            
            try:
                # Clear GPU memory before each attempt
                self._clear_gpu_memory()
                
                start_time = time.time()
                # sim = create_simulator("qiskit", method="statevector", device=device)
                # logging.info(f"Attempting statevector simulation on device: {getattr(sim, 'device', 'unknown')}")
                # logging.info(f"Create simulator Time taken: {time.time() - start_time}")
                
                # Estimate memory requirement and wait for GPU memory
                required_memory = self._estimate_memory_requirement(devide_num)
                logging.info(f"required_memory: {required_memory:.2f}GB")
                logging.info(f"Estimated memory requirement: {required_memory:.2f}GB")
                
                if not self._wait_for_gpu_memory(required_memory):
                    logging.warning("GPU memory not available, proceeding with CPU")
                # start_time = time.time()
                statevector = self.get_statevector(qasm)
                logging.info(f"Get statevector Time taken: {time.time() - start_time}")
                
                start_time = time.time()
                if statevector is not None:
                    # processor = PeakedCircuitProcessor(use_exact=True)
                    # logging.info(f"Create processor Time taken: {time.time() - start_time}")
                    
                    start_time = time.time()
                    result = self.process(statevector)
                    peak_bitstring = result.get("peak_bitstring")
                    bitstrings = result.get("bitstrings")
                    logging.info(f"Process statevector Time taken: {time.time() - start_time}")
                    
                    # Store results before cleanup
                    return_peak = peak_bitstring
                    return_bitstrings = bitstrings
                    
                    # Explicit cleanup in reverse order of creation
                    del result
                    # del processor
                    del statevector
                    # del sim
                    
                    # Clear GPU memory after processing
                    self._clear_gpu_memory()
                    
                    if return_peak:
                        logging.info(f"Statevector simulation successful on {device}")
                        return return_peak, return_bitstrings

            except Exception as e:
                logging.error(f"Simulation failed on {device}: {e}")
                
                # Clear GPU memory on failure
                self._clear_gpu_memory()
                
                if device == "CPU":
                    logging.error(f"Standard simulation failed on both GPU and CPU: {e}")
                    self._force_gpu_cache_clear()
                continue

        return "", []

    def _count_qubits(self, qasm: str) -> int:
        import re

        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        raise ValueError("Could not determine number of qubits from QASM")
    
    def _log_gpu_memory_usage(self, context: str = ""):
        """Log current GPU memory usage for debugging."""
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            try:
                # Use nvidia-ml-py for accurate memory tracking
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_memory_gb = mem_info.used / (1024**3)
                free_memory_gb = mem_info.free / (1024**3)
            except ImportError:
                # Fallback to PyTorch method (less accurate)
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                used_memory_gb = (allocated + cached) / (1024**3)
                free_memory_gb = gpu_memory_gb - used_memory_gb
            except Exception:
                # If nvidia-ml-py fails, use PyTorch
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                used_memory_gb = (allocated + cached) / (1024**3)
                free_memory_gb = gpu_memory_gb - used_memory_gb
                
                
                
                logging.info(f"Free memory: {free_memory_gb:.2f}GB")
                logging.info(f"Used memory: {used_memory_gb:.2f}GB")
                logging.info(f"Total memory: {gpu_memory_gb:.2f}GB")
        except Exception as e:
            logging.error(f"Error logging GPU memory: {e}")

    def parse_qasm(self, qasm:str = '', devide_num:int = 30, under:bool = True):
        qasm_arr = qasm.split('\n')[7:-2]
        IS_CNOTS = False
        IS_ISING = False
        return_qasm = ''
        if qasm_arr[0].startswith("rz"):
            IS_CNOTS = True
        elif qasm_arr[0].startswith("u3"):
            IS_ISING = True
        
        nqubits = self._count_qubits(qasm)
        if not under:
            devide_num = nqubits - devide_num
        
        def subtract_10_in_brackets(match):
            num = int(match.group(1))
            return f"q[{num - devide_num}]"
        
        if under:
            if IS_CNOTS:
                chunk_size = 10
                new_arr = [qasm_arr[i:i + chunk_size] for i in range(0, len(qasm_arr), chunk_size)]

                for arr_ele in new_arr:
                    first_num = int(arr_ele[2].split("[")[1].split("]")[0])
                    second_num = int(arr_ele[3].split("[")[1].split("]")[0])
                    if first_num >= devide_num or second_num >= devide_num:
                        continue
                    return_qasm += "\n".join(arr_ele) + "\n"
            elif IS_ISING:
                chunk_size = 29
                new_arr = [qasm_arr[i:i + chunk_size] for i in range(0, len(qasm_arr), chunk_size)]

                for arr_ele in new_arr:
                    first_num = int(arr_ele[0].split("[")[1].split("]")[0])
                    second_num = int(arr_ele[1].split("[")[1].split("]")[0])
                    if first_num >= devide_num or second_num >= devide_num:
                        continue
                    return_qasm += "\n".join(arr_ele) + "\n"
            else:
                return ''
            return_qasm = f'''
OPENQASM 2.0;
include "qelib1.inc";

qreg q[{devide_num}];
creg c[{devide_num}];

{return_qasm}measure q -> c;
'''
        else:
            if IS_CNOTS:
                chunk_size = 10
                new_arr = [qasm_arr[i:i + chunk_size] for i in range(0, len(qasm_arr), chunk_size)]

                for arr_ele in new_arr:
                    first_num = int(arr_ele[2].split("[")[1].split("]")[0])
                    second_num = int(arr_ele[3].split("[")[1].split("]")[0])
                    if first_num < devide_num or second_num < devide_num:
                        continue
                    return_qasm += "\n".join(arr_ele) + "\n"
            elif IS_ISING:
                chunk_size = 29
                new_arr = [qasm_arr[i:i + chunk_size] for i in range(0, len(qasm_arr), chunk_size)]

                for arr_ele in new_arr:
                    first_num = int(arr_ele[0].split("[")[1].split("]")[0])
                    second_num = int(arr_ele[1].split("[")[1].split("]")[0])
                    if first_num < devide_num or second_num < devide_num:
                        continue
                    return_qasm += "\n".join(arr_ele) + "\n"
            else:
                return ''
            # Replace only numbers inside q[...]
            return_qasm = re.sub(r"q\[(\d+)\]", subtract_10_in_brackets, return_qasm)
        

            return_qasm = f'''
OPENQASM 2.0;
include "qelib1.inc";

qreg q[{nqubits - devide_num}];
creg c[{nqubits - devide_num}];

{return_qasm}measure q -> c;
'''
        return return_qasm

    def calculate_amplitude_from_qasm(self, qasm_string: str, target_bitstrings: str):
        """
        Calculates the amplitude of a specific bitstring from a large QASM circuit
        using a tensor network simulation with cuTensorNet.

        This function requires an NVIDIA GPU with a Compute Capability of 7.0+
        and the cuquantum-python library to be installed.

        Args:
            qasm_string: The OPENQASM 2.0 string representing the circuit.
            target_bitstring: The bitstring for which to find the amplitude (e.g., '0' * 39).

        Returns:
            The complex amplitude for the target bitstring.
        """
        # Clear GPU memory before starting
        self._clear_gpu_memory()
        
        # Parse the QASM string into a pytket circuit,
        # specifying a `maxwidth` larger than the 39-qubit classical register.
        pytket_circuit = circuit_from_qasm_str(qasm_string, maxwidth=40)
        n_qubits = pytket_circuit.n_qubits

        
        # Use the GeneralState class for exact simulation
        try:
            # Synchronize GPU before creating state
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Create a GeneralState object from the pytket circuit,
            # without passing the libhandle directly.
            state = GeneralState(pytket_circuit)
            
            amplitudes = []
            for target_bitstring in target_bitstrings:
            
                # Compute the amplitude of the specified bitstring
                target_int = int(target_bitstring[0], 2)
                amplitude = state.get_amplitude(target_int)
                amplitudes.append((target_bitstring[0], np.abs(amplitude)**2))
                if np.abs(amplitude)**2 > 1e-9 and n_qubits == 39:
                    break
                del target_int
                del amplitude
            
            sorted_amplitudes = sorted(amplitudes, key=lambda x: x[1], reverse=True)
            
            # Clean up cuTensorNet state first
            del state
            
            # Clean up remaining variables
            del pytket_circuit
            del amplitudes
            
            # Aggressive GPU memory cleanup
            self._clear_gpu_memory()
                
            return sorted_amplitudes
        except Exception as e:
            logging.error(f"An error occurred during cuTensorNet simulation: {e}")
            # Clean up on error
            self._clear_gpu_memory()
            return []
        
    def _parse_qasm(self, qasm: str) -> QuantumCircuit:
        """Parse QASM string to QuantumCircuit with fallback for custom instructions."""
        try:
            return QuantumCircuit.from_qasm_str(qasm)
        except Exception:
            custom_instructions = [
                CustomInstruction("rxx", 1, 2, RXXGate, builtin=True),
                CustomInstruction("ryy", 1, 2, RYYGate, builtin=True),
                CustomInstruction("rzz", 1, 2, RZZGate, builtin=True),
                CustomInstruction("u3", 3, 1, U3Gate, builtin=True),
            ]
            return loads(qasm, custom_instructions=custom_instructions)
        
    def get_statevector(self, qasm: str) -> Optional[np.ndarray]:
        """Get the statevector of the circuit."""
        try:
            circuit = self._parse_qasm(qasm)
            circuit_no_meas = circuit.remove_final_measurements(inplace=False)

            backend_sv = AerSimulator(method="statevector", device="GPU")
            circuit_no_meas.save_statevector()  # type: ignore
            job = backend_sv.run(circuit_no_meas, shots=1)
            result = job.result()
            statevector = result.data(0)["statevector"]
            np_statevector = np.array(statevector)
            
            return np_statevector

        except Exception as e:
            # Clean up on error
            self._clear_gpu_memory()
            raise RuntimeError(f"Failed to get statevector: {str(e)}")
    
    def process(self, counts_or_statevector: Any, **kwargs) -> Dict[str, Any]:
        """
        Extracts the most probable bitstring

        Args:
            counts_or_statevector: Either:
                - Dict[str, int]: measurement counts (if use_exact=False)
                - np.ndarray: statevector (if use_exact=True)

        Returns:
            Dictionary containing:
                - peak_bitstring: The most probable bitstring
                - peak_probability: Its probability
        """
        
        if isinstance(counts_or_statevector, dict):
            raise ValueError("Expected statevector array but got counts dict. Set use_exact=False for sampling.")

        statevector = counts_or_statevector
        n_qubits = int(np.log2(len(statevector)))

        # Use optimized processing method based on statevector size
        return self._process_large_statevector(statevector, n_qubits)
        
    
    def _process_large_statevector(self, statevector: np.ndarray, n_qubits: int) -> Dict[str, Any]:
        """
        Optimized processing for very large statevectors (>2^20 elements).
        Uses chunked processing and memory-efficient operations.
        """
        # For very large statevectors, use chunked processing
        if len(statevector) > 2**20:  # 1M elements
            return self._process_chunked_statevector(statevector, n_qubits)
        
        # Standard optimized processing for smaller statevectors
        return self._process_standard_statevector(statevector, n_qubits)
    
    def _process_chunked_statevector(self, statevector: np.ndarray, n_qubits: int) -> Dict[str, Any]:
        """
        Process very large statevectors in chunks to avoid memory issues.
        Uses adaptive chunking based on available memory.
        """
        # Adaptive chunk size based on statevector size
        if len(statevector) > 2**25:  # 32M elements
            chunk_size = 2**16  # 64K elements per chunk
        else:
            chunk_size = 2**18  # 256K elements per chunk
            
        n_chunks = (len(statevector) + chunk_size - 1) // chunk_size
        
        max_prob = 0.0
        max_idx = 0
        top_candidates = []
        
        # Process chunks with memory-efficient operations
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(statevector))
            chunk = statevector[start:end]
            
            # Compute probabilities without modifying the original chunk
            # Use np.abs to get magnitudes, then square for probabilities
            chunk_probs = np.abs(chunk) ** 2
            
            # Find maximum in this chunk
            chunk_max_idx = np.argmax(chunk_probs)
            chunk_max_prob = chunk_probs[chunk_max_idx]
            
            # Update global maximum
            if chunk_max_prob > max_prob:
                max_prob = chunk_max_prob
                max_idx = start + chunk_max_idx
            
            # Collect top candidates from this chunk using argpartition
            n_candidates = min(5, len(chunk_probs))
            chunk_top_indices = np.argpartition(chunk_probs, -n_candidates)[-n_candidates:]
            
            # Only add candidates with significant probability
            for idx in chunk_top_indices:
                prob = chunk_probs[idx]
                if prob > 1e-10:  # Only add non-negligible probabilities
                    top_candidates.append((start + idx, prob))
        
        # Sort all candidates and take top 5
        top_candidates.sort(key=lambda x: x[1], reverse=True)
        top_5_candidates = top_candidates[:10]
        
        # Generate results with pre-computed format string
        bitstring_format = f"0{n_qubits}b"
        peak_bitstring = format(max_idx, bitstring_format)[::-1]
        
        bitstrings = []
        for idx, prob in top_5_candidates:
            raw_bitstr = format(idx, bitstring_format)
            bitstr = raw_bitstr[::-1]
            bitstrings.append({"bitstring": bitstr, "probability": prob})
        
        peaking = top_5_candidates[0][1] / top_5_candidates[1][1] if len(top_5_candidates) > 1 else 1.0
        
        del statevector
        
        return {
            "peak_bitstring": peak_bitstring,
            "peak_probability": float(max_prob),
            "peaking_ratio": peaking,
            "bitstrings": bitstrings,
        }
    
    def _process_standard_statevector(self, statevector: np.ndarray, n_qubits: int) -> Dict[str, Any]:
        """
        Standard optimized processing for smaller statevectors.
        """
        # Compute probabilities without modifying the original statevector
        # Use np.abs to get magnitudes, then square for probabilities
        probabilities = np.abs(statevector) ** 2

        # Use argpartition for O(n) top-k selection instead of O(n log n) full sort
        # This is much faster than sorting the entire array
        top_5_indices = np.argpartition(probabilities, -5)[-5:]
        
        # Sort only the top 5 indices (much smaller operation)
        top_5_probs = probabilities[top_5_indices]
        sorted_indices = np.argsort(top_5_probs)[::-1]
        top_5_indices = top_5_indices[sorted_indices]
        
        peak_idx = top_5_indices[0]
        peak_probability = probabilities[peak_idx]

        # Pre-compute format string to avoid repeated string formatting
        bitstring_format = f"0{n_qubits}b"
        peak_bitstring = format(peak_idx, bitstring_format)[::-1]

        # Optimized bitstring generation with minimal allocations
        bitstrings = []
        for idx in top_5_indices:
            raw_bitstr = format(idx, bitstring_format)
            bitstr = raw_bitstr[::-1]
            prob = probabilities[idx]
            bitstrings.append({"bitstring": bitstr, "probability": prob})
        
        peaking = probabilities[top_5_indices[0]] / probabilities[top_5_indices[1]]

        return {
            "peak_bitstring": peak_bitstring,
            "peak_probability": float(peak_probability),
            "peaking_ratio": peaking,
            "bitstrings": bitstrings,
        }
