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
    pub start: usize,

    #[clap(long, default_value = "50")]
    pub end: usize,

    #[clap(long, default_value = "1")]
    pub step_qty: usize,

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
    
    info!("[MASTER] Mode: SEQUENTIAL BATCH (Square Constraint). Total Cores: {}. Workers per Task: {}.", 
        total_cpu_cores, n_workers);

    let input_file_path = &args.main_args.input;
    let base_ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    // 3. VÒNG LẶP TUẦN TỰ (SEQUENTIAL LOOP)
    let mut qty = args.start;
    
    while qty <= args.end {
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

    // 2. THIẾT LẬP CONFIG "ULTRA" (HẠNG NẶNG CHO BATCH RUN)
    let mut config = DEFAULT_SPARROW_CONFIG;
    config.rng_seed = args.rng_seed
        .map(|s| s as usize)
        .or_else(|| Some(rand::rng().random::<u64>() as usize));
    let master_seed = config.rng_seed.unwrap() as u64;

    // A. Sử dụng tối đa luồng được cấp
    config.expl_cfg.separator_config.n_workers = n_workers;
    config.cmpr_cfg.separator_config.n_workers = n_workers;

    // B. Ultra Sampling (Giữ nguyên cấu hình cao để tìm lời giải tốt nhất)
    let ultra_sample_config = SampleConfig {
        n_container_samples: 200, 
        n_focussed_samples: 100,  
        n_coord_descents: 20,     
    };
    config.expl_cfg.separator_config.sample_config = ultra_sample_config;
    config.cmpr_cfg.separator_config.sample_config = ultra_sample_config;

    // C. Persistence
    config.expl_cfg.separator_config.iter_no_imprv_limit = 1000;
    config.cmpr_cfg.separator_config.iter_no_imprv_limit = 1000;

    config.expl_cfg.separator_config.strike_limit = 20;

    config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(0.99);
    
    // D. Geometry Precision
    config.poly_simpl_tolerance = Some(0.00001);

    // E. Time Limits (Quan trọng: Đặt thời gian đủ lâu cho việc nén hình vuông)
    // Nếu args CLI có truyền time thì dùng, không thì dùng mặc định khá rộng rãi cho batch
    config.expl_cfg.time_limit = Duration::from_secs(180); // 2 phút explore
    config.cmpr_cfg.time_limit = Duration::from_secs(120);  // 1 phút compress
    if let Some(gt) = args.global_time {
        config.expl_cfg.time_limit = Duration::from_secs(gt).mul_f64(DEFAULT_EXPLORE_TIME_RATIO);
        config.cmpr_cfg.time_limit = Duration::from_secs(gt).mul_f64(DEFAULT_COMPRESS_TIME_RATIO);
    }

    // 3. CHUẨN BỊ DỮ LIỆU & TÍNH TOÁN DIỆN TÍCH
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    
    // Import để lấy thông tin hình học chính xác
    let base_instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;

    let n = target_qty as f64;
    // Kích thước khởi tạo an toàn: Căn bậc 2 diện tích * 1.3
    let start_size = (0.4 * n).sqrt();
    info!("[Job {}] Start Square Size: {:.2}", target_qty, start_size);

    // Set chiều cao/rộng khởi tạo cho instance
    let mut current_ext_instance = ext_instance.clone();
    current_ext_instance.strip_height = start_size;

    let instance_struct = jagua_rs::probs::spp::io::import(&importer, &current_ext_instance)?;

    // 4. CHẠY OPTIMIZE (SINGLE RUN - SQUARE CONSTRAINT)
    // Không dùng vòng lặp Binary Search nữa, để thuật toán tự co (shrink) hình vuông
    let rng = Xoshiro256PlusPlus::seed_from_u64(master_seed);
    let mut ctrlc_terminator = CtrlCTerminator::new(); 
    
    let final_svg_path = Some(format!("{}/result.svg", task_dir));
    let mut final_exporter = SvgExporter::new(final_svg_path, None, None);

    // Dùng catch_unwind để đảm bảo 1 job chết không kéo theo cả batch
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        optimize(
            instance_struct.clone(),
            rng,
            &mut final_exporter,
            &mut ctrlc_terminator,
            &config.expl_cfg,
            &config.cmpr_cfg
        )
    }));

    match result {
        Ok(final_solution) => {
            let final_size = final_solution.strip_width();
            let final_score = final_size * final_size / n;
            info!("[Job {}] SUCCESS.", target_qty);
            info!("Final Square Side: {:.10}", final_size);
            info!("Final Score: {:.10}", final_score);

            // Cập nhật lại snapshot instance để output JSON đúng kích thước
            let mut final_snapshot = current_ext_instance.clone();
            final_snapshot.strip_height = final_size;

            let json_path = format!("{}/result.json", task_dir);
            let output_struct = SPOutput {
                instance: final_snapshot,
                solution: jagua_rs::probs::spp::io::export(&instance_struct, &final_solution, *EPOCH)
            };
            // io::write_json(&output_struct, Path::new(&json_path), log::Level::Info)?;

            let csv_path = format!("output/result.csv");
            io::write_csv(&output_struct.solution, Path::new(&csv_path))?;
        }
        Err(_) => {
            error!("[Job {}] FAILED due to panic.", target_qty);
        }
    }

    Ok(())
}