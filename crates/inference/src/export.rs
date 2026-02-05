//! Export model to packed .1bit binary format.
//!
//! # Format (.1bit)
//!
//! ```text
//! [MAGIC: 4 bytes "1BIT"]
//! [VERSION: u32 LE]
//! [NUM_TENSORS: u32 LE]
//! [CONFIG_JSON_LEN: u32 LE]
//! [CONFIG_JSON: utf-8]
//! [TENSOR_HEADER_0: name_len(u32) + name + dtype(u8) + ndims(u32) + dims + data_len(u64)]
//! [TENSOR_DATA_0: packed 2-bit for ternary, 1-bit for binary]
//! ...
//! ```
//!
//! Ternary packing: 4 values per byte (2 bits each: 00=0, 01=+1, 11=-1).
//! Binary packing: 8 values per byte (1 bit each: 0=-1, 1=+1).

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use candle_nn::VarMap;

use ternary_common::OneBitLlmConfig;

const MAGIC: &[u8; 4] = b"1BIT";
const VERSION: u32 = 1;

/// Weight type for packing.
#[derive(Clone, Copy, Debug)]
pub enum PackMode {
    /// 1-bit: {-1, +1}
    Binary,
    /// 2-bit: {-1, 0, +1}
    Ternary,
}

/// Export a VarMap to the `.1bit` format.
pub fn export_1bit(
    varmap: &VarMap,
    config: &OneBitLlmConfig,
    output_path: &Path,
    pack_mode: PackMode,
) -> Result<u64> {
    let mut file = std::fs::File::create(output_path)?;

    // Header
    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;

    let vars = varmap.all_vars();
    let num_tensors = vars.len() as u32;
    file.write_all(&num_tensors.to_le_bytes())?;

    let config_json = serde_json::to_string(config)?;
    let json_bytes = config_json.as_bytes();
    file.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    file.write_all(json_bytes)?;

    let data = varmap.data().lock().unwrap();
    let mut total_bytes = 0u64;

    for (name, var) in data.iter() {
        let tensor = var.as_tensor();
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let dims: Vec<usize> = tensor.dims().to_vec();

        // Name
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;

        // Dtype tag
        let dtype_tag: u8 = match pack_mode {
            PackMode::Binary => 1,
            PackMode::Ternary => 2,
        };
        file.write_all(&[dtype_tag])?;

        // Dims
        file.write_all(&(dims.len() as u32).to_le_bytes())?;
        for &d in &dims {
            file.write_all(&(d as u64).to_le_bytes())?;
        }

        // Pack and write data
        let packed = match pack_mode {
            PackMode::Binary => pack_binary(&flat),
            PackMode::Ternary => pack_ternary(&flat),
        };
        file.write_all(&(packed.len() as u64).to_le_bytes())?;
        file.write_all(&packed)?;
        total_bytes += packed.len() as u64;
    }

    Ok(total_bytes)
}

/// Pack ternary values into 2-bit: 4 values per byte.
/// Encoding: 00 = 0, 01 = +1, 11 = -1.
fn pack_ternary(values: &[f32]) -> Vec<u8> {
    let num_bytes = (values.len() + 3) / 4;
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let code: u8 = if v > 0.5 {
            0b01 // +1
        } else if v < -0.5 {
            0b11 // -1
        } else {
            0b00 // 0
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= code << bit_offset;
    }
    packed
}

/// Pack binary values into 1-bit: 8 values per byte.
/// Encoding: 1 = +1, 0 = -1.
fn pack_binary(values: &[f32]) -> Vec<u8> {
    let num_bytes = (values.len() + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        if v > 0.0 {
            let byte_idx = i / 8;
            let bit_offset = i % 8;
            packed[byte_idx] |= 1 << bit_offset;
        }
    }
    packed
}

/// Generate a C header for loading .1bit files.
pub fn generate_c_header() -> String {
    r#"// onebit_format.h — Auto-generated header for .1bit model format
// Usage: #include "onebit_format.h"
#pragma once

#include <stdint.h>

#define ONEBIT_MAGIC "1BIT"
#define ONEBIT_VERSION 1

// Dtype tags
#define ONEBIT_DTYPE_BINARY  1  // 1-bit: 8 values per byte
#define ONEBIT_DTYPE_TERNARY 2  // 2-bit: 4 values per byte

// Ternary decoding: 2 bits → {-1, 0, +1}
// 00 → 0, 01 → +1, 11 → -1
static inline int8_t onebit_decode_ternary(uint8_t byte, int pos) {
    uint8_t code = (byte >> (pos * 2)) & 0x03;
    if (code == 0x01) return  1;
    if (code == 0x03) return -1;
    return 0;
}

// Binary decoding: 1 bit → {-1, +1}
// 0 → -1, 1 → +1
static inline int8_t onebit_decode_binary(uint8_t byte, int pos) {
    return ((byte >> pos) & 1) ? 1 : -1;
}
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_pack_round_trip() {
        let vals = vec![-1.0f32, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let packed = pack_ternary(&vals);
        // Unpack and verify
        for (i, &original) in vals.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let code = (packed[byte_idx] >> bit_offset) & 0b11;
            let unpacked = match code {
                0b01 => 1.0f32,
                0b11 => -1.0,
                _ => 0.0,
            };
            assert_eq!(original, unpacked, "mismatch at index {i}");
        }
    }

    #[test]
    fn binary_pack_round_trip() {
        let vals = vec![-1.0f32, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0];
        let packed = pack_binary(&vals);
        for (i, &original) in vals.iter().enumerate() {
            let byte_idx = i / 8;
            let bit_offset = i % 8;
            let bit = (packed[byte_idx] >> bit_offset) & 1;
            let unpacked = if bit == 1 { 1.0f32 } else { -1.0 };
            assert_eq!(original, unpacked, "mismatch at index {i}");
        }
    }
}
