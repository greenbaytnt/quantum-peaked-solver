import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import logging
from datetime import datetime
from solver import DefaultPeakedSolver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_oldest_json_file(base_path: str = "/root/sn63/peaked_circuits") -> Optional[Path]:
    """Get the oldest JSON file by timestamp in filename (excluding queue, processed, failed dirs)"""
    base = Path(base_path)
    if not base.exists():
        return None
    
    exclude_dirs = {'queue', 'processed', 'failed'}
    json_files = []
    
    for json_file in base.glob('*.json'):
        try:
            # Extract timestamp from filename format: {cid}_{timestamp}.json
            filename = json_file.stem
            if '_' in filename:
                timestamp_str = filename.split('_')[-1]
                timestamp = float(timestamp_str)
                json_files.append((timestamp, json_file))
        except (ValueError, IndexError):
            continue
    
    if not json_files:
        return None
    
    # Sort by timestamp and return oldest
    json_files.sort(key=lambda x: x[0])
    return json_files[0][1]


def read_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Read a single JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def process_circuit(data: Dict[str, Any], file_path: Path) -> bool:
    """Process the circuit data - implement your logic here"""
    logging.info(f"Processing: {file_path.name}")
    logging.info(f"  CID: {data.get('cid', 'N/A')}")
    logging.info(f"  Circuit Type: {data.get('circuit_type', 'N/A')}")
    logging.info(f"  Miner: {data.get('miner_name', 'N/A')}")
    logging.info(f"  Validator: {data.get('validator_hotkey', 'N/A')}")
    logging.info(f"  Solved: {data.get('solved', False)}")
    
    cid = data.get('cid', 'N/A')
    validator_hotkey = data.get('validator_hotkey', 'N/A')
    miner_name = data.get('miner_name', 'N/A')
    
    qasm_path = file_path.parent / f"{cid}.qasm"
    with open(qasm_path, 'r', encoding='utf-8') as f:
        qasm_to_process = f.read()
    
    # YOUR PROCESSING LOGIC HERE
    # For example: solve the circuit, validate, etc.
    peaked_solver = DefaultPeakedSolver()
    result = peaked_solver.solve(qasm_to_process, cid, validator_hotkey)
    
    payload = {
        "challenge_id": cid,
        "solution_bitstring": result,
        "timestamp": time.time(),
        "validator_hotkey": validator_hotkey
    }
    
    save_path = Path(f"/root/sn63-quantum/qbittensor/miner/{miner_name}/peaked_circuits/solved_circuits/{cid}.json")
    
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        logging.info(f"  Saved result to: {save_path}")
        return True, save_path
    except Exception as e:
        logging.error(f"  Error saving result: {e}")
        return False, None


def move_to_queue(file_path: Path, base_path: Path) -> Optional[Path]:
    """Move file to queue directory to claim it for processing"""
    try:
        queue_dir = base_path.parent / "queue"
        queue_dir.mkdir(exist_ok=True)
        new_path = queue_dir / file_path.name
        file_path.rename(new_path)
        logging.info(f"  Claimed: {file_path.name}")
        filename = file_path.stem
        if '_' in filename:
            cid = filename.split('_')[0]
            qasm_path = base_path.parent / f"{cid}.qasm"
            qasm_new_path = qasm_path.parent / "queue" / f"{cid}.qasm"
            qasm_path.rename(qasm_new_path)
            logging.info(f"  Moved to: {qasm_new_path.name}")
        return new_path
    except FileNotFoundError:
        # Another process already moved it
        return None
    except Exception as e:
        logging.error(f"  Error claiming file: {e}")
        return None


def move_to_result(file_path: Path, base_path: Path, success: bool, save_path: Optional[Path] = None) -> None:
    """Move file from queue to processed or failed directory"""
    try:
        result_dir = base_path.parent / ("processed" if success else "failed")
        result_dir.mkdir(exist_ok=True)
        new_path = result_dir / file_path.name
        file_path.rename(new_path)
        logging.info(f"  Moved to: {result_dir.name}/{file_path.name}")
        if save_path:
            new_save_path = save_path.parent / ("processed" if success else "failed") / save_path.name
            save_path.rename(new_save_path)
            logging.info(f"  Moved to: {new_save_path.name}")
    except Exception as e:
        logging.error(f"  Error moving to result: {e}")


def process_loop(base_path: str = "/root/sn63/peaked_circuits", 
                 delay: float = 1.0) -> None:
    """Continuously process oldest files using queue system"""
    base = Path(base_path)
    
    # Get oldest file from main directory
    oldest_file = get_oldest_json_file(base_path)
    
    while True:
    
        if not oldest_file:
            logging.info("No more files to process")
            time.sleep(delay)
            oldest_file = get_oldest_json_file(base_path)
        else:
            break
    # Move to queue immediately to claim it (prevents race condition)
    queued_file = move_to_queue(oldest_file, base)
    
    if not queued_file:
        # Another process claimed it, try next file
        logging.warning("File already claimed by another process")
        return
    
    # Read file from queue
    data = read_json_file(queued_file)
    if not data:
        logging.error("Invalid file, moving to failed")
        move_to_result(queued_file, base, False)
        return
    
    # Process the circuit
    try:
        success, save_path = process_circuit(data, queued_file)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        success = False
    
    # Move to result directory
    move_to_result(queued_file, base, success, save_path)
        


if __name__ == "__main__":
    base_path = "/root/sn63/peaked_circuits"
    
    logging.info(f"Starting process with PID: {os.getpid()}")
    logging.info(f"Processing circuits from: {base_path}")
    logging.info("=" * 60)
    
    try:
        process_loop(
            base_path=base_path,
            delay=5,
        )
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        sys.exit(0) # Always exit cleanly after a task

