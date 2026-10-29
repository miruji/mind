//! Minimal binary helpers for the checkpoint format. The project stays free
//! of external crates, so the few primitives are written by hand.

use std::io::{Read, Result, Write};

pub fn writeU32(out: &mut dyn Write, value: u32) -> Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn writeU64(out: &mut dyn Write, value: u64) -> Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn writeF64(out: &mut dyn Write, value: f64) -> Result<()> {
    out.write_all(&value.to_le_bytes())
}

pub fn writeStr(out: &mut dyn Write, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    writeU32(out, bytes.len() as u32)?;
    out.write_all(bytes)
}

pub fn readU32(input: &mut dyn Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn readU64(input: &mut dyn Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn readF64(input: &mut dyn Read) -> Result<f64> {
    let mut buf = [0u8; 8];
    input.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub fn readStr(input: &mut dyn Read) -> Result<String> {
    let len = readU32(input)? as usize;
    let mut buf = vec![0u8; len];
    input.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
