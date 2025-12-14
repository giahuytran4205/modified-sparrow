extern crate core;

use clap::Parser as Clap;
use log::{info, warn, error, Level};
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use sparrow::config::*;
use sparrow::optimizer::optimize;
use sparrow::util::io;
use sparrow::util::io::{MainCli, SPOutput};
use std::fs::{self, File}; // Thêm File để dùng trong write_csv
use std::path::Path;
use std::io::Write; // Import Write trait cho write_csv
use std::time::Duration;
use jagua_rs::io::import::Importer;
use sparrow::EPOCH;

use anyhow::{bail, Result};
use rand_xoshiro::Xoshiro256PlusPlus;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, LOG_LEVEL_FILTER_RELEASE};
use sparrow::util::svg_exporter::SvgExporter;
use sparrow::util::ctrlc_terminator::CtrlCTerminator;
use std::panic;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub const OUTPUT_DIR: &str = "output";
pub const LIVE_DIR: &str = "data/live";

fn main() -> Result<()> {
    // --- 1. LẤY SỐ LUỒNG CPU & KHỞI TẠO ---
    let cpu_cores = std::thread::available_parallelism().unwrap().get();
    let mut config = DEFAULT_SPARROW_CONFIG;

    fs::create_dir_all(OUTPUT_DIR)?;
    let log_file_path = format!("{}/log.txt", OUTPUT_DIR);
    
    // Config Logger
    match cfg!(debug_assertions) {
        true => io::init_logger(log::LevelFilter::Debug, Path::new(&log_file_path))?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?,
    }

    let args = MainCli::parse();
    let input_file_path = &args.input;
    
    info!("[MAIN] Detected {} CPU cores. Strategy: Sequential Full-Power Retry.", cpu_cores);

    // --- 2. XỬ LÝ TIME LIMIT (CHO BƯỚC FINAL POLISH) ---
    let (explore_dur, compress_dur) = match (args.global_time, args.exploration, args.compression) {
        (Some(gt), None, None) => (Duration::from_secs(gt).mul_f64(DEFAULT_EXPLORE_TIME_RATIO), Duration::from_secs(gt).mul_f64(DEFAULT_COMPRESS_TIME_RATIO)),
        (None, Some(et), Some(ct)) => (Duration::from_secs(et), Duration::from_secs(ct)),
        (None, None, None) => (Duration::from_secs(600).mul_f64(DEFAULT_EXPLORE_TIME_RATIO), Duration::from_secs(600).mul_f64(DEFAULT_COMPRESS_TIME_RATIO)),
        _ => bail!("invalid cli pattern"),
    };
    
    // Config gốc (Thời gian dài) để dùng cho bước cuối cùng
    config.expl_cfg.time_limit = explore_dur;
    config.cmpr_cfg.time_limit = compress_dur;
    
    // [QUAN TRỌNG] Luôn dùng Max Core cho hiệu suất tốt nhất
    config.expl_cfg.separator_config.n_workers = cpu_cores;
    config.cmpr_cfg.separator_config.n_workers = cpu_cores;

    // Xử lý Early Termination
    if args.early_termination {
        config.expl_cfg.max_conseq_failed_attempts = Some(sparrow::consts::DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(sparrow::consts::DEFAULT_FAIL_DECAY_RATIO_CMPR);
        warn!("[MAIN] early termination enabled!");
    }

    // Xử lý Seed
    if let Some(arg_rng_seed) = args.rng_seed {
        config.rng_seed = Some(arg_rng_seed as usize);
    }
    let master_seed = match config.rng_seed {
        Some(seed) => seed as u64,
        None => rand::random(),
    };

    // --- 3. KHỞI TẠO SINGLETON TERMINATOR ---
    // Tạo duy nhất 1 lần ở đây để tránh lỗi panic
    let mut ctrlc_terminator = CtrlCTerminator::new();

    // --- 4. CHUẨN BỊ DỮ LIỆU ---
    let ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    
    // Tính toán cận dưới dựa trên diện tích
    let base_instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;

    let n_points= base_instance.item_qty(0) as f64;
    let max_h = (0.7 * n_points).sqrt();
    let min_h= (0.3 * n_points).sqrt();

    info!("Num points: {n_points}");
    
    info!("[PRE-CALC] Max height: {max_h:.2}, Min height: {min_h:.2}");

    let mut low = min_h; 
    let mut high = max_h;
    // Đảm bảo high luôn lớn hơn low một chút để thuật toán chạy được
    if high < low { high = low * 1.5; }

    let mut best_strip_height = high;
    let mut best_solution = None;
    let mut best_instance_snapshot = ext_instance.clone();
    best_instance_snapshot.strip_height = high;

    // --- 5. CẤU HÌNH "FAST CONFIG" CHO PROBE ---
    // Tăng thời gian lên để chống lại Load Average cao của Cloud VM
    let probe_explore_dur = Duration::from_secs(4); // 45s Explore
    let probe_compress_dur = Duration::from_secs(2); // 15s Compress
    
    let mut fast_config = config.clone();
    fast_config.expl_cfg.time_limit = probe_explore_dur;
    fast_config.cmpr_cfg.time_limit = probe_compress_dur;
    // Tắt bớt log để giảm I/O wait
    fast_config.expl_cfg.separator_config.log_level = log::Level::Warn;
    fast_config.cmpr_cfg.separator_config.log_level = log::Level::Warn;

    fast_config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(0.95);

    info!("[SQUARE SEARCH] Range: [{:.2} - {:.2}]", low, high);

    // --- 6. VÒNG LẶP SEQUENTIAL BINARY SEARCH ---
    while (high - low) > 0.0000001 { // Độ chính xác dừng 0.05
        let mid = (low + high) / 2.0;
        let mut success = false;
        
        // Số lần thử lại tại mỗi kích thước (Tăng lên 7 vì máy lag)
        const MAX_ATTEMPTS: usize = 5; 
        
        info!("\n=== Testing D = {:.2} (Max {} attempts) ===", mid, MAX_ATTEMPTS);

        for attempt in 0..MAX_ATTEMPTS {
            // Đổi seed cho mỗi lần thử
            let current_seed = master_seed.wrapping_add(attempt as u64 * 7777);
            let rng = Xoshiro256PlusPlus::seed_from_u64(current_seed);
            
            let mut current_ext_instance = ext_instance.clone();
            current_ext_instance.strip_height = mid;

            // Import instance với chiều cao mới
            if let Ok(instance) = jagua_rs::probs::spp::io::import(&importer, &current_ext_instance) {
                // Tắt SVG export khi đang dò tìm
                let mut no_svg = SvgExporter::new(None, None, None); 

                // Chạy Optimize an toàn (catch panic)
                let run_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    optimize(
                        instance.clone(), 
                        rng, 
                        &mut no_svg, 
                        &mut ctrlc_terminator, // Truyền tham chiếu đến terminator global
                        &fast_config.expl_cfg, 
                        &fast_config.cmpr_cfg
                    )
                }));

                if let Ok(solution) = run_result {
                    let w = solution.strip_width();
                    if w <= mid {
                        info!("   [Attempt {}] SUCCESS! Fits (L={:.2} <= D={:.2})", attempt+1, w, mid);
                        
                        best_strip_height = mid;
                        best_solution = Some(solution);
                        best_instance_snapshot = current_ext_instance;
                        
                        success = true;
                        break; // Tìm thấy rồi thì thoát vòng lặp attempt
                    }
                }
            }
        }

        if success {
            high = mid; // Thử kích thước nhỏ hơn
        } else {
            info!("   -> All {} attempts failed. Increasing size.", MAX_ATTEMPTS);
            low = mid; // Phải tăng kích thước
        }
    }

    info!("\n=== 6.5. LINEAR SQUEEZE (Vắt kiệt kết quả) ===");
    info!("Binary Search stopped at {:.2}. Trying to squeeze further...", best_strip_height);

    // Chiến thuật: Giảm dần độ cao từng chút một cho đến khi thất bại hoàn toàn
    let mut squeeze_step = best_strip_height * 0.01; // Bước giảm 1%
    let min_step = 0.1; // Độ chính xác tối thiểu
    
    // Config cho giai đoạn vắt kiệt: Cần lì lợm hơn
    let mut squeeze_config = fast_config.clone();
    squeeze_config.expl_cfg.time_limit = Duration::from_secs(10); // Tăng thời gian lên 10s
    squeeze_config.cmpr_cfg.time_limit = Duration::from_secs(5);

    loop {
        if squeeze_step < min_step { break; }
        
        let target_h = best_strip_height - squeeze_step;
        if target_h < low { break; } // Không xuống thấp hơn cận dưới lý thuyết

        info!("-> Squeezing: Testing H = {:.2} (Step {:.2})", target_h, squeeze_step);
        
        let mut success = false;
        // Tăng số lần thử lên cao (ví dụ 10 lần) để chắc chắn không bỏ sót
        const SQUEEZE_ATTEMPTS: usize = 10; 

        for attempt in 0..SQUEEZE_ATTEMPTS {
            let current_seed = master_seed.wrapping_add(attempt as u64 * 5555 + 123);
            let rng = Xoshiro256PlusPlus::seed_from_u64(current_seed);
            
            let mut current_ext_instance = ext_instance.clone();
            current_ext_instance.strip_height = target_h;

            if let Ok(instance) = jagua_rs::probs::spp::io::import(&importer, &current_ext_instance) {
                let mut no_svg = SvgExporter::new(None, None, None); 
                
                let run_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    optimize(
                        instance.clone(), rng, &mut no_svg, 
                        &mut ctrlc_terminator, 
                        &squeeze_config.expl_cfg, &squeeze_config.cmpr_cfg
                    )
                }));

                if let Ok(solution) = run_result {
                    let w = solution.strip_width();
                    if w <= target_h {
                        info!("   [Squeeze Success] Fits in {:.2}!", target_h);
                        
                        best_strip_height = target_h;
                        best_solution = Some(solution);
                        best_instance_snapshot = current_ext_instance;
                        
                        success = true;
                        break; 
                    }
                }
            }
        }

        if success {
            // Nếu thành công, giữ nguyên step để giảm tiếp
            info!("   -> Good! Going deeper...");
        } else {
            // Nếu thất bại, đừng dừng ngay! Hãy giảm bước nhảy nhỏ lại để dò kỹ hơn
            // Ví dụ: Đang giảm 10 đơn vị không được, thử giảm 5 đơn vị xem sao?
            info!("   -> Failed at {:.2}. Reducing step size.", target_h);
            squeeze_step /= 2.0; 
        }
    }

    // --- 7. FINAL POLISH (CHẠY KỸ LẦN CUỐI) ---
    info!("\n=== FINAL OPTIMIZATION (Using full config) ===");
    info!("Best square side found: {}. Refining...", best_strip_height);
    
    // Dùng config gốc (thời gian dài)
    let rng = Xoshiro256PlusPlus::seed_from_u64(master_seed);
    
    let final_svg_path = Some(format!("{OUTPUT_DIR}/final_square_{:.2}.svg", best_strip_height));
    let mut final_exporter = SvgExporter::new(final_svg_path, None, None);
    
    let best_instance_struct = jagua_rs::probs::spp::io::import(&importer, &best_instance_snapshot)?;
    
    // Chạy optimize lần cuối để lấy file SVG đẹp và JSON chi tiết
    let final_solution = optimize(
        best_instance_struct.clone(),
        rng,
        &mut final_exporter,
        &mut ctrlc_terminator,
        &config.expl_cfg, 
        &config.cmpr_cfg
    );

    // Xuất JSON
    let json_path = format!("{OUTPUT_DIR}/final_square_{:.2}.json", best_strip_height);
    let output_struct = SPOutput {
        instance: best_instance_snapshot,
        solution: jagua_rs::probs::spp::io::export(&best_instance_struct, &final_solution, *EPOCH)
    };
    io::write_json(&output_struct, Path::new(json_path.as_str()), Level::Info)?;
    
    // Xuất CSV (Gọi hàm phụ trợ bên dưới)
    let num_points = output_struct.solution.layout.placed_items.len();
    let csv_path = format!("{OUTPUT_DIR}/final_{num_points}.csv");
    if let Err(e) = io::write_csv(&output_struct.solution, Path::new(&csv_path)) {
        error!("Failed to write CSV: {}", e);
    } else {
        info!("Saved CSV result to {}", csv_path);
    }

    info!("Saved final JSON result to {}", json_path);

    Ok(())
}