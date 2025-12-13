extern crate core;

use clap::Parser as Clap;
use log::{Level, error, info, warn};
use rand::SeedableRng;
use sparrow::config::*;
use sparrow::optimizer::optimize;
use sparrow::util::io;
use sparrow::util::io::{MainCli, SPOutput};
use std::fs;
use std::path::Path;
use std::time::Duration;
use jagua_rs::io::import::Importer;
use sparrow::EPOCH;

use anyhow::{bail, Result};
use rand_xoshiro::Xoshiro256PlusPlus;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR, DEFAULT_MAX_CONSEQ_FAILS_EXPL, LOG_LEVEL_FILTER_DEBUG, LOG_LEVEL_FILTER_RELEASE};
use sparrow::util::svg_exporter::SvgExporter;
use sparrow::util::ctrlc_terminator::CtrlCTerminator;
use std::panic;

pub const OUTPUT_DIR: &str = "output";

pub const LIVE_DIR: &str = "data/live";

fn main() -> Result<()>{
    let mut config = DEFAULT_SPARROW_CONFIG;

    fs::create_dir_all(OUTPUT_DIR)?;
    let log_file_path = format!("{}/log.txt", OUTPUT_DIR);
    match cfg!(debug_assertions) {
        true => io::init_logger(LOG_LEVEL_FILTER_DEBUG, Path::new(&log_file_path))?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?,
    }

    let args = MainCli::parse();
    let input_file_path = &args.input;
    let (explore_dur, compress_dur) = match (args.global_time, args.exploration, args.compression) {
        (Some(gt), None, None) => {
            (Duration::from_secs(gt).mul_f32(DEFAULT_EXPLORE_TIME_RATIO), Duration::from_secs(gt).mul_f32(DEFAULT_COMPRESS_TIME_RATIO))
        },
        (None, Some(et), Some(ct)) => {
            (Duration::from_secs(et), Duration::from_secs(ct))
        },
        (None, None, None) => {
            warn!("[MAIN] no time limit specified");
            (Duration::from_secs(600).mul_f32(DEFAULT_EXPLORE_TIME_RATIO), Duration::from_secs(600).mul_f32(DEFAULT_COMPRESS_TIME_RATIO))
        },
        _ => bail!("invalid cli pattern (clap should have caught this)"),
    };
    config.expl_cfg.time_limit = explore_dur;
    config.cmpr_cfg.time_limit = compress_dur;
    if args.early_termination {
        config.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
        warn!("[MAIN] early termination enabled!");
    }
    if let Some(arg_rng_seed) = args.rng_seed {
        config.rng_seed = Some(arg_rng_seed as usize);
    }

    info!("[MAIN] configured to explore for {}s and compress for {}s", explore_dur.as_secs(), compress_dur.as_secs());

    let rng = match config.rng_seed {
        Some(seed) => {
            info!("[MAIN] using seed: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed as u64)
        },
        None => {
            let seed = rand::random();
            warn!("[MAIN] no seed provided, using: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed)
        }
    };

    info!("[MAIN] system time: {}", jiff::Timestamp::now());

    let ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    let instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;
    
    info!("[MAIN] loaded instance {} with #{} items", ext_instance.name, instance.total_item_qty());
    
    let mut svg_exporter = {
        let final_svg_path = Some(format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name));

        let intermediate_svg_dir = match cfg!(feature = "only_final_svg") {
            true => None,
            false => Some(format!("{OUTPUT_DIR}/sols_{}", ext_instance.name))
        };

        let live_svg_path = match cfg!(feature = "live_svg") {
            true => Some(format!("{LIVE_DIR}/.live_solution.svg")),
            false => None
        };
        
        SvgExporter::new(
            final_svg_path,
            intermediate_svg_dir,
            live_svg_path
        )
    };
    
    let mut ctrlc_terminator = CtrlCTerminator::new();

    // let solution = optimize(instance.clone(), rng, &mut svg_exporter, &mut ctrlc_terminator, &config.expl_cfg, &config.cmpr_cfg);

    // let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    // let json_output = SPOutput {
    //     instance: ext_instance,
    //     solution: jagua_rs::probs::spp::io::export(&instance, &solution, *EPOCH)
    // };
    // io::write_json(&json_output, Path::new(json_path.as_str()), Level::Info)?;

    // -------------------------------

    let total_area: f32 = instance.items.iter().map(|(item, _)| item.area()).sum();
    let min_theoretical_side = total_area.sqrt();

    // Cận dưới: 0 (hoặc kích thước item lớn nhất nếu bạn biết)
    let mut low = min_theoretical_side; 
    // Cận trên: strip_height gốc từ file JSON (đảm bảo file gốc là giá trị chạy được)
    let mut high = min_theoretical_side * 2.0;
    
    // Biến lưu kết quả tốt nhất tìm được
    let mut best_strip_height = high;
    let mut best_solution = None;
    let mut best_instance_snapshot = ext_instance.clone(); // Dùng để export JSON cuối cùng

    // Số bước lặp hoặc độ chính xác (epsilon)
    let iterations = 20; // Chạy 10 lần logN -> độ chính xác khá cao
    // Hoặc dùng while (high - low) > 1.0 ...

    info!("[BINARY SEARCH] Start searching min strip_height from {:.2} down to {:.2}", high, low);

    for i in 0..iterations {
        let mid = (low + high) / 2.0;
        info!("\n=== Iteration {}/{}: Testing strip_height = {:.2} ===", i + 1, iterations, mid);

        // A. Clone instance ra bản mới để sửa
        let mut current_ext_instance = ext_instance.clone();
        
        // B. Sửa trực tiếp strip_height
        // Lưu ý: mid là f64. Bạn cần ép kiểu về đúng kiểu của strip_height trong struct (thường là f64 hoặc u32)
        // Nếu struct dùng f64:
        current_ext_instance.strip_height = mid;
        
        // Nếu struct dùng u32 (hoặc i32), hãy comment dòng trên và dùng dòng này:
        // current_ext_instance.strip_height = mid.round() as u32; 

        // C. Import lại với instance đã sửa height
        let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
        
        // Truyền thẳng current_ext_instance vào (không cần JSON hack nữa)
        let import_result = jagua_rs::probs::spp::io::import(&importer, &current_ext_instance);

        let rng = match config.rng_seed {
            Some(seed) => {
                info!("[MAIN] using seed: {}", seed);
                Xoshiro256PlusPlus::seed_from_u64(seed as u64)
            },
            None => {
                let seed = rand::random();
                warn!("[MAIN] no seed provided, using: {}", seed);
                Xoshiro256PlusPlus::seed_from_u64(seed)
            }
        };

        // BỌC HÀM OPTIMIZE ĐỂ BẮT LỖI PANIC
        // AssertUnwindSafe cần thiết vì ta đang truyền tham chiếu mutable (&mut) qua ranh giới panic
        match import_result {
            Ok(instance) => {
                let mut svg_exporter = SvgExporter::new(None, None, None); 

                // 4. Chạy Optimize (Bắt Panic)
                let run_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    optimize(
                        instance.clone(), rng, &mut svg_exporter, &mut ctrlc_terminator, &config.expl_cfg, &config.cmpr_cfg
                    )
                }));

                match run_result {
                    Ok(solution) => {
                        // Optimize chạy thành công, ta có được chiều dài (width)
                        let resulting_width = solution.strip_width();
                        info!("-> Optimizer finished. Input Height: {:.2}, Result Width: {:.2}", mid, resulting_width);

                        // === KIỂM TRA ĐIỀU KIỆN HÌNH VUÔNG ===
                        if resulting_width <= mid {
                            // >>> THÀNH CÔNG: Nằm gọn trong hình vuông DxD <<<
                            info!("-> [SUCCESS] Fits in square! (Width {:.2} <= Height {:.2})", resulting_width, mid);
                            
                            // Cập nhật kết quả tốt nhất
                            best_strip_height = mid;
                            best_solution = Some(solution);
                            best_instance_snapshot = current_ext_instance; // Lưu snapshot với height = mid đúng
                            
                            // Thử tìm cạnh nhỏ hơn nữa
                            high = mid;
                        } else {
                            // >>> THẤT BẠI: Bị tràn ngang (Width > Height) <<<
                            warn!("-> [FAIL] Spilled horizontally (Width {:.2} > Height {:.2}). Need larger square.", resulting_width, mid);
                            // Kích thước này chưa đủ, cần tăng lên
                            low = mid;
                        }
                    },
                    Err(_) => {
                        // >>> THẤT BẠI: Panic (Item không nhét vừa chiều cao) <<<
                        warn!("-> [FAIL] Optimizer Panicked (Height {:.2} too small for items).", mid);
                        low = mid;
                    }
                }
            },
            Err(e) => {
                // >>> THẤT BẠI: Import lỗi (Item to hơn chiều cao) <<<
                warn!("-> [FAIL] Import Error at height {:.2}: {}", mid, e);
                low = mid;
            }
        }
    }

    // ==================================================================================
    // KẾT THÚC VÀ GHI FILE KẾT QUẢ TỐT NHẤT
    // ==================================================================================

    info!("\n=== BINARY SEARCH FINISHED ===");
    info!("Min valid strip_height found: {:.2}", best_strip_height);

    if let Some(sol) = best_solution {
        // Ghi lại file kết quả với best_strip_height
        
        // Cần tạo lại SvgExporter để ghi file (vì trong loop ta đã tắt)
        let final_svg_path = Some(format!("{OUTPUT_DIR}/final_{}_minH.svg", best_instance_snapshot.name));
        let final_exporter = SvgExporter::new(final_svg_path, None, None);
        
        let json_path = format!("{OUTPUT_DIR}/final_{}_minH.json", best_instance_snapshot.name);
        
        // Re-import instance chuẩn từ best snapshot để export
        let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
        let final_instance_struct = jagua_rs::probs::spp::io::import(&importer, &best_instance_snapshot)?;
        
        let solution = jagua_rs::probs::spp::io::export(&final_instance_struct, &sol, *EPOCH);
        let num_points = solution.layout.placed_items.len();
        let csv_path = format!("{OUTPUT_DIR}/final_{num_points}.csv");

        let json_output = SPOutput {
            instance: best_instance_snapshot,
            solution: solution
        };
        io::write_json(&json_output, Path::new(json_path.as_str()), Level::Info)?;
        io::write_csv(&json_output.solution, Path::new(csv_path.as_str()))?;
        info!("Saved best result to {}", json_path);
    } else {
        error!("Could not find any valid solution even at max height!");
    }

    Ok(())
}
