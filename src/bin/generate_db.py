import os
import math
import struct
from tqdm import tqdm

FILENAME = "ypir_db.bin"
NUM_ROWS = 131_072
ITEM_SIZE_BITS = 184_320
BITS_PER_COEFF = 15
PT_MODULUS = 1 << 15  # 32768

def get_pattern(index, num_coeffs):
    """
    Encode index into first few coefficients (base 32768),
    rest are zeros.
    """
    coeffs = []
    temp = index
    if temp == 0:
        coeffs = [0]
    while temp > 0:
        coeffs.append(temp % PT_MODULUS)
        temp //= PT_MODULUS
    
    # Pad to full row
    while len(coeffs) < num_coeffs:
        coeffs.append(0)
    
    return b"".join(struct.pack('<H', c) for c in coeffs)

def generate_db():
    coeffs_per_row = ITEM_SIZE_BITS // BITS_PER_COEFF  # 12288
    bytes_per_row = coeffs_per_row * 2
    total_bytes = NUM_ROWS * bytes_per_row
    
    print(f"Configuration:")
    print(f"  Rows: {NUM_ROWS}")
    print(f"  Cols: {coeffs_per_row}")
    print(f"  PT Modulus: {PT_MODULUS}")
    print(f"  Bytes/Row: {bytes_per_row}")
    print(f"  Total Size: {total_bytes / (1024**3):.2f} GB")
    
    rows_per_batch = 1000
    
    with open(FILENAME, "wb") as f:
        pbar = tqdm(total=NUM_ROWS, desc="Generating")
        
        for start in range(0, NUM_ROWS, rows_per_batch):
            end = min(start + rows_per_batch, NUM_ROWS)
            batch = bytearray()
            
            for i in range(start, end):
                batch.extend(get_pattern(i, coeffs_per_row))
            
            f.write(batch)
            pbar.update(end - start)
        
        pbar.close()
    
    print(f"\nCreated {FILENAME} ({os.path.getsize(FILENAME):,} bytes)")

if __name__ == "__main__":
    generate_db()