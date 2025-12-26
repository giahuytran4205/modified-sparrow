extern crate core;

use clap::Parser as Clap;
use jagua_rs::probs::spp::io::ext_repr::ExtSPInstance;
use log::{info, warn, error, Level};
use rand::SeedableRng;
use sparrow::config::*;
use sparrow::optimizer::optimize;
use sparrow::util::io;
use sparrow::util::io::{MainCli, SPOutput};
use std::fs::{self, File};
use std::path::Path;
use std::io::Write;
use std::time::Duration;
use jagua_rs::io::import::Importer;
use sparrow::EPOCH;

// Import các struct config cần thiết
use sparrow::sample::search::SampleConfig; 
use sparrow::optimizer::separator::SeparatorConfig;

use anyhow::{bail, Result};
use rand_xoshiro::Xoshiro256PlusPlus;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, LOG_LEVEL_FILTER_RELEASE};
use sparrow::util::svg_exporter::SvgExporter;
use sparrow::util::ctrlc_terminator::CtrlCTerminator;
use std::panic;
use rand::Rng;

// Không cần crossbeam nữa
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub const OUTPUT_DIR: &str = "output";

#[derive(Clap)]
#[clap(name = "Sparrow Sequential Batch Runner")]
pub struct BatchCli {
    #[clap(flatten)]
    pub main_args: MainCli,

    #[clap(long, default_value = "10")]
    pub start_qty: usize,

    #[clap(long, default_value = "50")]
    pub end_qty: usize,

    #[clap(long, default_value = "1")]
    pub step_qty: usize,

    // Đã bỏ parallel_tasks

    /// Ép cứng số core (nếu không muốn dùng hết 100% CPU)
    #[clap(long)]
    pub force_cores: Option<usize>,
}

fn main() -> Result<()> {
    // 1. KHỞI TẠO CƠ BẢN
    let total_cpu_cores = std::thread::available_parallelism().unwrap().get();
    let args = BatchCli::parse();
    
    fs::create_dir_all(OUTPUT_DIR)?;
    let log_file_path = format!("{}/log_master.txt", OUTPUT_DIR);
    
    match cfg!(debug_assertions) {
        true => io::init_logger(log::LevelFilter::Debug, Path::new(&log_file_path))?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?,
    }

    // 2. TÍNH TOÁN TÀI NGUYÊN (FULL POWER)
    // Vì chạy tuần tự, ta dùng toàn bộ số core cho task hiện tại
    let n_workers = args.force_cores.unwrap_or(total_cpu_cores);
    
    info!("[MASTER] Mode: SEQUENTIAL. Total Cores: {}. Workers per Task: {}.", 
        total_cpu_cores, n_workers);

    let input_file_path = &args.main_args.input;
    let base_ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    // 3. VÒNG LẶP TUẦN TỰ (SEQUENTIAL LOOP)
    let mut qty = args.start_qty;
    
    while qty <= args.end_qty {
        info!("\n========================================");
        info!("[MASTER] Starting Job: {} items", qty);
        info!("========================================");

        // Gọi hàm xử lý trực tiếp trên luồng chính
        if let Err(e) = solve_single_task(
            qty, 
            n_workers, 
            base_ext_instance.clone(),
            &args.main_args
        ) {
            error!("[MASTER] Job {} failed: {}", qty, e);
        } else {
            info!("[MASTER] Finished Job: {} items.", qty);
        }

        qty += args.step_qty;
    }

    info!("[MASTER] All jobs completed.");
    Ok(())
}

fn solve_single_task(
    target_qty: usize, 
    n_workers: usize, 
    mut ext_instance: ExtSPInstance,
    args: &MainCli
) -> Result<()> {
    
    // 1. CẬP NHẬT SỐ LƯỢNG ITEM
    if let Some(first_item) = ext_instance.items.first_mut() {
        first_item.demand = target_qty as u64;
    } else {
        bail!("Input file has no items!");
    }

    // tạo folder output
    let task_dir = format!("{}/qty_{}", OUTPUT_DIR, target_qty);
    fs::create_dir_all(&task_dir)?;

    // 2. THIẾT LẬP CONFIG "ULTRA" (HẠNG NẶNG)
    let mut config = DEFAULT_SPARROW_CONFIG;
    config.rng_seed = args.rng_seed
        .map(|s| s as usize)
        .or_else(|| Some(rand::rng().random::<u64>() as usize));
    let master_seed = config.rng_seed.unwrap() as u64;

    // A. Sử dụng tối đa luồng được cấp
    config.expl_cfg.separator_config.n_workers = n_workers;
    config.cmpr_cfg.separator_config.n_workers = n_workers;

    // B. Ultra Sampling
    let ultra_sample_config = SampleConfig {
        n_container_samples: 200, 
        n_focussed_samples: 100,  
        n_coord_descents: 10,     
    };
    config.expl_cfg.separator_config.sample_config = ultra_sample_config;
    config.cmpr_cfg.separator_config.sample_config = ultra_sample_config;

    // C. Persistence
    config.expl_cfg.separator_config.iter_no_imprv_limit = 500;
    config.expl_cfg.separator_config.strike_limit = 10;
    
    // D. Geometry Precision
    config.poly_simpl_tolerance = Some(0.0001);

    // E. Smart Shrink
    config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(0.98);

    let input_file_path = &args.input;
    
    let ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    
    let base_instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;
    let n = base_instance.item_qty(0) as f64;
    
    // Kích thước khởi tạo = Căn bậc 2 diện tích * 1.3 (để thuật toán Explore có chỗ "thở" lúc đầu rồi co dần)
    let start_size = (0.3 * n).sqrt();
    
    info!("[MAIN] Starting Square Size: {:.2}", start_size);

    // Setup instance khởi tạo là hình vuông
    let mut start_instance_snapshot = ext_instance.clone();
    start_instance_snapshot.strip_height = start_size;

    let rng = Xoshiro256PlusPlus::seed_from_u64(master_seed);
    let mut ctrlc_terminator = CtrlCTerminator::new();
    
    // --- 5. CHẠY OPTIMIZE (MỘT LẦN DUY NHẤT) ---
    // Không cần vòng lặp while dò tìm nữa
    
    let instance_struct = jagua_rs::probs::spp::io::import(&importer, &start_instance_snapshot)?;
    
    // Setup exporter
    let final_svg_path = Some(format!("{OUTPUT_DIR}/final_square.svg"));
    let mut exporter = SvgExporter::new(final_svg_path, None, None);

    info!("=== STARTING OPTIMIZATION (Square Constraint) ===");
    
    let final_solution = optimize(
        instance_struct.clone(),
        rng,
        &mut exporter,
        &mut ctrlc_terminator,
        &config.expl_cfg, 
        &config.cmpr_cfg
    );

    let final_size = final_solution.strip_width(); // Vì là hình vuông nên Width = Size
    info!("=== FINISHED. Best Square Side: {:.3} ===", final_size);

    // --- 6. XUẤT KẾT QUẢ ---
    let json_path = format!("{OUTPUT_DIR}/final_square_{:.2}.json", final_size);
    let output_struct = SPOutput {
        instance: start_instance_snapshot, // Lưu ý: Instance gốc này có thể chưa cập nhật size cuối cùng
        solution: jagua_rs::probs::spp::io::export(&instance_struct, &final_solution, *EPOCH)
    };
    io::write_json(&output_struct, Path::new(json_path.as_str()), Level::Info)?;
    
    let num_points = output_struct.solution.layout.placed_items.len();
    let csv_path = format!("{OUTPUT_DIR}/final_{num_points}.csv");
    if let Err(e) = io::write_csv(&output_struct.solution, Path::new(&csv_path)) {
        error!("Failed to write CSV: {}", e);
    } else {
        info!("Saved CSV result to {}", csv_path);
    }

    Ok(())
}