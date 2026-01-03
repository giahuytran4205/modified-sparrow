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
use sparrow::util::terminator::Terminator; // Import trait Terminator
use std::sync::{Mutex};
use jagua_rs::Instant;

// Import các struct config cần thiết
use sparrow::sample::search::SampleConfig; 
use sparrow::optimizer::separator::SeparatorConfig;

use anyhow::{bail, Result};
use rand_xoshiro::Xoshiro256PlusPlus;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, LOG_LEVEL_FILTER_RELEASE};
use sparrow::util::svg_exporter::SvgExporter;
use std::panic;
use rand::Rng;

// --- THƯ VIỆN CHO PARALLEL ---
use rayon::prelude::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub const OUTPUT_DIR: &str = "output";

// --- 1. ĐỊNH NGHĨA TERMINATOR CHO PARALLEL ---
#[derive(Clone)]
struct AtomicTerminator {
    /// Cờ dừng toàn cục (chia sẻ giữa tất cả các luồng)
    stop_flag: Arc<AtomicBool>,
    /// Timeout cục bộ (riêng cho từng task/luồng)
    timeout: Option<Instant>,
}

impl AtomicTerminator {
    fn new() -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        let r = flag.clone();
        
        // Đăng ký Ctrl-C handler 1 lần duy nhất
        // Lưu ý: Nếu chạy nhiều lần main logic, ctrlc::set_handler có thể trả về lỗi nếu set lại.
        // Trong thực tế nên xử lý Result, nhưng ở đây unwrap/expect cho đơn giản.
        let _ = ctrlc::set_handler(move || {
            warn!("[MAIN] Ctrl-C received! Signaling all workers to stop...");
            r.store(true, Ordering::SeqCst);
        });

        Self { 
            stop_flag: flag,
            timeout: None, // Mặc định chưa có timeout
        }
    }
}

// Implement đầy đủ trait Terminator
impl Terminator for AtomicTerminator {
    fn kill(&self) -> bool {
        // Dừng nếu:
        // 1. Có tín hiệu Ctrl-C toàn cục
        // HOẶC
        // 2. Đã quá thời gian timeout cục bộ (nếu có set timeout)
        let global_stop = self.stop_flag.load(Ordering::Relaxed);
        let local_timeout = self.timeout.map_or(false, |t| Instant::now() > t);
        
        global_stop || local_timeout
    }

    fn new_timeout(&mut self, timeout: Duration) {
        // Thiết lập timeout mới tính từ thời điểm hiện tại
        self.timeout = Some(Instant::now() + timeout);
    }

    fn timeout_at(&self) -> Option<Instant> {
        // Trả về thời điểm sẽ timeout
        self.timeout
    }
}

#[derive(Clap)]
#[clap(name = "Sparrow Parallel Batch Runner")]
pub struct BatchCli {
    #[clap(flatten)]
    pub main_args: MainCli,

    #[clap(long, default_value = "10")]
    pub start: usize,

    #[clap(long, default_value = "50")]
    pub end: usize,

    #[clap(long, default_value = "1")]
    pub step_qty: usize,

    /// Số core dành cho MỖI task (Ví dụ: 8)
    #[clap(long, default_value = "8")]
    pub cores_per_task: usize,
}

fn main() -> Result<()> {
    // 1. KHỞI TẠO
    let total_system_cores = std::thread::available_parallelism().unwrap().get();
    let args = BatchCli::parse();
    
    fs::create_dir_all(OUTPUT_DIR)?;
    let log_file_path = format!("{}/log_master.txt", OUTPUT_DIR);
    
    // Config Logger
    match cfg!(debug_assertions) {
        true => io::init_logger(log::LevelFilter::Debug, Path::new(&log_file_path))?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?,
    }

    // 2. TÍNH TOÁN PHÂN BỔ RESOURCE
    let cores_per_task = args.cores_per_task;
    // Số lượng task chạy song song tối đa = Tổng core hệ thống / Core mỗi task
    // Ví dụ: 32 / 8 = 4 tasks song song
    let max_parallel_tasks = std::cmp::max(1, total_system_cores / cores_per_task);

    info!("[MASTER] System Cores: {}. Cores/Task: {}. Parallel Tasks: {}.", 
        total_system_cores, cores_per_task, max_parallel_tasks);

    // Cấu hình ThreadPool toàn cục cho Rayon
    rayon::ThreadPoolBuilder::new()
        .num_threads(max_parallel_tasks) // Giới hạn số luồng xử lý song song
        .build_global()
        .unwrap();

    let input_file_path = &args.main_args.input;
    let base_ext_instance = io::read_spp_instance_json(Path::new(&input_file_path))?;

    // Khởi tạo Terminator chung (Thread-safe)
    let global_terminator = AtomicTerminator::new();
    let csv_file_mutex = Arc::new(Mutex::new(()));

    // 3. TẠO DANH SÁCH JOB
    // Tạo vector chứa các số lượng qty cần chạy: [10, 11, 12, ..., 50]
    let jobs: Vec<usize> = (args.start..=args.end).step_by(args.step_qty).collect();

    // 4. CHẠY SONG SONG (PARALLEL EXECUTION)
    // par_iter() sẽ tự động chia các job vào các luồng của Rayon
    jobs.par_iter().for_each(|&qty| {
        let worker_terminator = global_terminator.clone();
        
        // Kiểm tra nếu đã có tín hiệu dừng thì không chạy job mới
        if worker_terminator.kill() {
            return;
        }

        info!("[Job {}] Started on thread {:?}", qty, std::thread::current().id());

        match solve_single_task(
            qty,
            cores_per_task, // Truyền số core giới hạn cho mỗi task
            base_ext_instance.clone(),
            &args.main_args,
            worker_terminator,
            csv_file_mutex.clone()
        ) {
            Ok(_) => info!("[Job {}] Completed.", qty),
            Err(e) => error!("[Job {}] Failed: {}", qty, e),
        }
    });

    info!("[MASTER] All jobs completed.");
    Ok(())
}

fn solve_single_task(
    target_qty: usize, 
    n_workers: usize, // Đây là số core cho task này (cores_per_task)
    mut ext_instance: ExtSPInstance,
    args: &MainCli,
    mut terminator: AtomicTerminator, // Nhận Terminator dạng Clone
    csv_mutex: Arc<Mutex<()>>
) -> Result<()> {
    
    // 1. CẬP NHẬT SỐ LƯỢNG ITEM
    if let Some(first_item) = ext_instance.items.first_mut() {
        first_item.demand = target_qty as u64;
    } else {
        bail!("Input file has no items!");
    }

    // tạo folder output riêng
    let task_dir = format!("{}/qty_{}", OUTPUT_DIR, target_qty);
    fs::create_dir_all(&task_dir)?;

    // 2. CONFIG
    let mut config = DEFAULT_SPARROW_CONFIG;
    // Mỗi task tự random seed hoặc lấy từ args (cộng thêm qty để tránh trùng lặp giữa các task song song)
    config.rng_seed = args.rng_seed
        .map(|s| s as usize)
        .or_else(|| Some(rand::rng().random::<u64>() as usize));
    let master_seed = config.rng_seed.unwrap() as u64;

    // A. SET SỐ WORKER CHO TASK NÀY
    config.expl_cfg.separator_config.n_workers = n_workers;
    config.cmpr_cfg.separator_config.n_workers = n_workers;

    // B. Ultra Sampling 
    let ultra_sample_config = SampleConfig {
        n_container_samples: 200, 
        n_focussed_samples: 100,  
        n_coord_descents: 20,     
    };
    config.expl_cfg.separator_config.sample_config = ultra_sample_config.clone();
    config.cmpr_cfg.separator_config.sample_config = ultra_sample_config;

    // C. Persistence
    config.expl_cfg.separator_config.iter_no_imprv_limit = 1000;
    config.cmpr_cfg.separator_config.iter_no_imprv_limit = 1000;
    config.expl_cfg.separator_config.strike_limit = 20;
    config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(0.99);
    config.poly_simpl_tolerance = Some(0.00001);

    // E. Time Limits
    config.expl_cfg.time_limit = Duration::from_secs(180); 
    config.cmpr_cfg.time_limit = Duration::from_secs(120);
    if let Some(gt) = args.global_time {
        config.expl_cfg.time_limit = Duration::from_secs(gt).mul_f64(DEFAULT_EXPLORE_TIME_RATIO);
        config.cmpr_cfg.time_limit = Duration::from_secs(gt).mul_f64(DEFAULT_COMPRESS_TIME_RATIO);
    }

    // 3. CHUẨN BỊ DATA
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.min_item_separation, config.narrow_concavity_cutoff_ratio);
    let base_instance = jagua_rs::probs::spp::io::import(&importer, &ext_instance)?;

    let n = target_qty as f64;
    let start_size = (0.4 * n).sqrt() * 1.2; 
    
    let mut current_ext_instance = ext_instance.clone();
    current_ext_instance.strip_height = start_size;

    let instance_struct = jagua_rs::probs::spp::io::import(&importer, &current_ext_instance)?;

    // 4. OPTIMIZE
    let rng = Xoshiro256PlusPlus::seed_from_u64(master_seed);
    
    let final_svg_path = Some(format!("{}/result.svg", task_dir));
    let mut final_exporter = SvgExporter::new(final_svg_path, None, None);

    // Bắt panic để 1 thread chết không kéo sập cả main process
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        optimize(
            instance_struct.clone(),
            rng,
            &mut final_exporter,
            &mut terminator, // Pass &mut của local terminator clone
            &config.expl_cfg,
            &config.cmpr_cfg
        )
    }));

    match result {
        Ok(final_solution) => {
            let final_size = final_solution.strip_width();
            let final_score = final_size * final_size / n;
            
            // Log info: Dùng println! hoặc info! nhưng lưu ý log có thể bị trộn lẫn giữa các thread
            info!("[Job {}] SUCCESS. Side: {:.5}, Score: {:.5}", target_qty, final_size, final_score);

            let mut final_snapshot = current_ext_instance.clone();
            final_snapshot.strip_height = final_size;

            let json_path = format!("{}/result.json", task_dir);
            let output_struct = SPOutput {
                instance: final_snapshot,
                solution: jagua_rs::probs::spp::io::export(&instance_struct, &final_solution, *EPOCH)
            };
            // io::write_json(&output_struct, Path::new(&json_path), log::Level::Warn)?; // Giảm log level xuống warn để đỡ spam

            // Ghi CSV vào file tổng (Dùng Mutex nếu ghi chung file, nhưng ở đây ta ghi file riêng từng job để an toàn)
            // Hoặc ghi vào file trong thư mục con
            {
                // Khóa lại! Các luồng khác phải chờ ở dòng này cho đến khi luồng này ghi xong.
                let _guard = csv_mutex.lock().unwrap(); 
                
                let csv_path = format!("{}/result.csv", OUTPUT_DIR);
                // Gọi hàm write_csv (phiên bản append mà bạn đã sửa ở turn trước)
                io::write_csv(&output_struct.solution, Path::new(&csv_path))?;
            }
        }
        Err(_) => {
            error!("[Job {}] FAILED / PANIC.", target_qty);
        }
    }

    Ok(())
}