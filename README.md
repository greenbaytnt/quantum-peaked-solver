# Quantum Peaked Solver

Process JSON files from peaked circuits using a queue-based system to prevent race conditions when running multiple processes.

## Usage

### Single Process
```bash
python process_miner_data.py
```

### Multiple Processes with PM2 (Recommended)
```bash
# Install PM2 if not already installed
npm install -g pm2

# Create logs directory
mkdir -p /root/sn63/logs

# Start 16 processes
pm2 start ecosystem.config.js

# View logs
pm2 logs

# View specific process
pm2 logs quantum-solver-1

# View status
pm2 status

# Stop all processes
pm2 stop all

# Restart all processes
pm2 restart all

# Delete all processes
pm2 delete all
```

## How It Works

1. **Oldest file selection**: Finds oldest JSON file by timestamp in filename
2. **Move to queue**: Immediately moves file to `queue/` directory (claims it)
3. **Process**: Processes the circuit (3-4 minutes)
4. **Move to result**: Moves to `processed/` or `failed/` directory

This queue system prevents multiple processes from working on the same file.

## Directory Structure

```
/root/sn63/
├── peaked_circuits/
│   └── {cid}_{timestamp}.json   # New files waiting to be processed
├── queue/                       # Files currently being processed
│   └── {cid}_{timestamp}.json
├── processed/                   # Successfully processed files
│   └── {cid}_{timestamp}.json
└── failed/                      # Failed files
    └── {cid}_{timestamp}.json
```

## JSON File Format

Files are saved with format: `{cid}_{timestamp}.json`

Example content:
```json
{
  "cid": "QmXxxx...",
  "qasm_to_process": "OPENQASM 2.0;...",
  "circuit_type": "type1",
  "validator_hotkey": "5Fxxx...",
  "solved": false,
  "miner_name": "miner_1"
}
```

## Logging

The script uses Python logging with timestamps and process IDs:
- Format: `YYYY-MM-DD HH:MM:SS - PID:12345 - LEVEL - MESSAGE`
- Logs are automatically managed by PM2 when using the ecosystem config
- Separate log files for each process in `/root/sn63/logs/`
- Each process has both output and error logs

