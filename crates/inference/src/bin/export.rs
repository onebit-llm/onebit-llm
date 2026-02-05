//! Export trained model to .1bit binary format.

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;

use ternary_common::OneBitLlmConfig;
use ternary_core::OneBitLlm;
use ternary_infer::{export_1bit, generate_c_header, PackMode};

#[derive(Parser)]
#[command(name = "onebit-export", about = "Export model to .1bit binary")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "model.1bit")]
    output: PathBuf,
    #[arg(long, default_value = "ternary", value_parser = ["binary", "ternary"])]
    pack_mode: String,
    #[arg(long)]
    emit_c_header: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu; // Export runs on CPU

    let config = OneBitLlmConfig::load(&args.model_dir.join("config.json"))?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = OneBitLlm::new(vb, &config)?;
    varmap.load(&args.model_dir.join("model.safetensors"))?;

    let pack_mode = match args.pack_mode.as_str() {
        "binary" => PackMode::Binary,
        _ => PackMode::Ternary,
    };

    eprintln!("Exporting to {} ...", args.output.display());
    let total = export_1bit(&varmap, &config, &args.output, pack_mode)?;
    eprintln!("Done. Packed data: {} bytes ({:.2} MB)", total, total as f64 / 1024.0 / 1024.0);

    if args.emit_c_header {
        let header = generate_c_header();
        let header_path = args.output.with_extension("h");
        std::fs::write(&header_path, header)?;
        eprintln!("C header: {}", header_path.display());
    }

    Ok(())
}
