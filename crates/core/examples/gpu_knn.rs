use std::fs;
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use proxima_core::gpu::CudaKnn;
use proxima_core::{FlatIndex, Metric};

const CUDA_ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CUDA_ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;

extern "C" {
    #[link_name = "cudaSetDevice"]
    fn cuda_set_device(device: c_int) -> c_int;
    #[link_name = "cudaGetDeviceCount"]
    fn cuda_get_device_count(count: *mut c_int) -> c_int;
    #[link_name = "cudaGetDeviceProperties"]
    fn cuda_get_device_properties(prop: *mut u8, device: c_int) -> c_int;
    #[link_name = "cudaDeviceGetAttribute"]
    fn cuda_device_get_attribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    #[link_name = "cudaMemGetInfo"]
    fn cuda_mem_get_info(free: *mut usize, total: *mut usize) -> c_int;
    #[link_name = "cudaDriverGetVersion"]
    fn cuda_driver_get_version(version: *mut c_int) -> c_int;
    #[link_name = "cudaRuntimeGetVersion"]
    fn cuda_runtime_get_version(version: *mut c_int) -> c_int;
}

struct GpuInfo {
    index: c_int,
    count: Option<i32>,
    name: Option<String>,
    compute_capability: Option<String>,
    driver_version: Option<String>,
    runtime_version: Option<String>,
    total_vram_bytes: Option<u64>,
    select_error: Option<i32>,
}

impl GpuInfo {
    fn query(device: c_int) -> Self {
        let rc = unsafe { cuda_set_device(device) };
        let mut info = GpuInfo {
            index: device,
            count: cuda_int(|v| unsafe { cuda_get_device_count(v) }),
            name: None,
            compute_capability: None,
            driver_version: cuda_int(|v| unsafe { cuda_driver_get_version(v) })
                .map(format_cuda_version),
            runtime_version: cuda_int(|v| unsafe { cuda_runtime_get_version(v) })
                .map(format_cuda_version),
            total_vram_bytes: None,
            select_error: if rc == 0 { None } else { Some(rc) },
        };
        if rc != 0 {
            return info;
        }
        info.name = device_name(device);
        let major = cuda_int(|v| unsafe {
            cuda_device_get_attribute(v, CUDA_ATTR_COMPUTE_CAPABILITY_MAJOR, device)
        });
        let minor = cuda_int(|v| unsafe {
            cuda_device_get_attribute(v, CUDA_ATTR_COMPUTE_CAPABILITY_MINOR, device)
        });
        if let (Some(major), Some(minor)) = (major, minor) {
            info.compute_capability = Some(format!("{major}.{minor}"));
        }
        let (mut free, mut total) = (0usize, 0usize);
        if unsafe { cuda_mem_get_info(&mut free, &mut total) } == 0 {
            info.total_vram_bytes = Some(total as u64);
        }
        info
    }

    fn vram_display(&self) -> String {
        match self.total_vram_bytes {
            Some(bytes) => format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
            None => "unknown VRAM".to_string(),
        }
    }
}

fn cuda_int(call: impl Fn(*mut c_int) -> c_int) -> Option<i32> {
    let mut value: c_int = 0;
    if call(&mut value) == 0 {
        Some(value as i32)
    } else {
        None
    }
}

fn format_cuda_version(v: i32) -> String {
    format!("{}.{}", v / 1000, (v % 1000) / 10)
}

fn device_name(device: c_int) -> Option<String> {
    let mut prop = vec![0u8; 4096];
    if unsafe { cuda_get_device_properties(prop.as_mut_ptr(), device) } != 0 {
        return None;
    }
    let end = prop.iter().take(256).position(|&b| b == 0).unwrap_or(256);
    let name = String::from_utf8_lossy(&prop[..end]).trim().to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn hostname() -> Option<String> {
    for key in ["COMPUTERNAME", "HOSTNAME"] {
        if let Ok(value) = std::env::var(key) {
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    run("hostname", &[])
}

fn cpu_model() -> Option<String> {
    if cfg!(target_os = "linux") {
        if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if line.to_ascii_lowercase().starts_with("model name") {
                    if let Some((_, value)) = line.split_once(':') {
                        return Some(value.trim().to_string());
                    }
                }
            }
        }
    }
    if cfg!(target_os = "macos") {
        if let Some(brand) = run("sysctl", &["-n", "machdep.cpu.brand_string"]) {
            return Some(brand);
        }
    }
    if cfg!(target_os = "windows") {
        let key = "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
        if let Some(out) = run("reg", &["query", key, "/v", "ProcessorNameString"]) {
            if let Some(at) = out.find("REG_SZ") {
                let name = out[at + "REG_SZ".len()..].trim().to_string();
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
        if let Ok(value) = std::env::var("PROCESSOR_IDENTIFIER") {
            return Some(value);
        }
    }
    None
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn utc_timestamp() -> String {
    let secs = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => return "unknown".to_string(),
    };
    let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jopt(value: Option<&str>) -> String {
    match value {
        Some(v) => jstr(v),
        None => "null".to_string(),
    }
}

fn jnum(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

fn jint(value: Option<i64>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    }
}

fn jbool(value: Option<bool>) -> String {
    match value {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    }
}

fn jarray(values: &[String]) -> String {
    let items: Vec<String> = values.iter().map(|v| jstr(v)).collect();
    format!("[{}]", items.join(", "))
}

fn jobj(indent: usize, entries: &[(&str, String)]) -> String {
    let pad = " ".repeat(indent + 2);
    let inner: Vec<String> = entries
        .iter()
        .map(|(key, value)| format!("{pad}{}: {value}", jstr(key)))
        .collect();
    format!("{{\n{}\n{}}}", inner.join(",\n"), " ".repeat(indent))
}

#[allow(clippy::too_many_arguments)]
fn build_json(argv: &[String], gpu: &GpuInfo, cpu: &str, cores: usize, n: usize, dim: usize,
              nq: usize, k: usize, cpu1: f64, cpu_mt: f64, gpu_t: f64,
              agreement: Option<f64>) -> String {
    let dirty = git(&["status", "--porcelain"]).map(|s| !s.is_empty());
    let os = jobj(4, &[
        ("system", jstr(std::env::consts::OS)),
        ("machine", jstr(std::env::consts::ARCH)),
    ]);
    let cpu_obj = jobj(4, &[
        ("model", jstr(cpu)),
        ("cores_logical", jint(Some(cores as i64))),
    ]);
    let gpu_obj = jobj(4, &[
        ("device_index", jint(Some(gpu.index as i64))),
        ("device_count", jint(gpu.count.map(|v| v as i64))),
        ("name", jopt(gpu.name.as_deref())),
        ("compute_capability", jopt(gpu.compute_capability.as_deref())),
        ("cuda_driver_version", jopt(gpu.driver_version.as_deref())),
        ("cuda_runtime_version", jopt(gpu.runtime_version.as_deref())),
        ("total_vram_bytes", jint(gpu.total_vram_bytes.map(|v| v as i64))),
    ]);
    let git_obj = jobj(4, &[
        ("sha", jopt(git(&["rev-parse", "HEAD"]).as_deref())),
        ("branch", jopt(git(&["rev-parse", "--abbrev-ref", "HEAD"]).as_deref())),
        ("dirty", jbool(dirty)),
    ]);
    let versions = jobj(4, &[
        ("rustc", jopt(run("rustc", &["--version"]).as_deref())),
    ]);
    let provenance = jobj(2, &[
        ("timestamp_utc", jstr(&utc_timestamp())),
        ("hostname", jopt(hostname().as_deref())),
        ("argv", jarray(argv)),
        ("cwd", jopt(std::env::current_dir().ok()
            .and_then(|p| p.to_str().map(str::to_string)).as_deref())),
        ("os", os),
        ("cpu", cpu_obj),
        ("gpu", gpu_obj),
        ("git", git_obj),
        ("versions", versions),
    ]);
    let workload = jobj(2, &[
        ("n_base", jint(Some(n as i64))),
        ("dim", jint(Some(dim as i64))),
        ("n_queries", jint(Some(nq as i64))),
        ("k", jint(Some(k as i64))),
        ("metric", jstr("L2")),
    ]);
    let results = jobj(2, &[
        ("cpu_1t_seconds", jnum(cpu1)),
        ("cpu_1t_qps", jnum(nq as f64 / cpu1)),
        ("cpu_mt_seconds", jnum(cpu_mt)),
        ("cpu_mt_qps", jnum(nq as f64 / cpu_mt)),
        ("gpu_seconds", jnum(gpu_t)),
        ("gpu_qps", jnum(nq as f64 / gpu_t)),
        ("speedup_vs_cpu_1t", jnum(cpu1 / gpu_t)),
        ("speedup_vs_cpu_mt", jnum(cpu_mt / gpu_t)),
        ("topk_agreement", agreement.map(jnum).unwrap_or_else(|| "null".to_string())),
    ]);
    let doc = jobj(0, &[
        ("schema_version", "1".to_string()),
        ("provenance", provenance),
        ("workload", workload),
        ("results", results),
    ]);
    format!("{doc}\n")
}

fn write_artifact(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)
}

fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n * dim)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, def: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(def);
    let (n, dim, nq, k) = (arg(1, 1_000_000), arg(2, 128), arg(3, 2000), arg(4, 10));
    let device = arg(5, 0) as c_int;
    let metric = Metric::L2;

    let gpu_info = GpuInfo::query(device);
    let cpu = cpu_model().unwrap_or_else(|| "unknown".to_string());
    let cores = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0);

    println!("exact k-NN: N={n} dim={dim} queries={nq} k={k} metric=L2");
    println!("  host: {cpu} ({cores} logical processors)");
    if let Some(rc) = gpu_info.select_error {
        println!("  gpu : cudaSetDevice({device}) failed with CUDA error {rc}");
    }
    println!(
        "  gpu : {} (device {device} of {}, compute capability {}, {})",
        gpu_info.name.as_deref().unwrap_or("unknown"),
        gpu_info.count.map(|c| c.to_string()).unwrap_or_else(|| "?".to_string()),
        gpu_info.compute_capability.as_deref().unwrap_or("unknown"),
        gpu_info.vram_display()
    );
    println!(
        "  cuda: driver {} / runtime {}",
        gpu_info.driver_version.as_deref().unwrap_or("unknown"),
        gpu_info.runtime_version.as_deref().unwrap_or("unknown")
    );

    let base = gen(n, dim, 1);
    let queries = gen(nq, dim, 2);

    let mut flat = FlatIndex::new(dim, metric);
    flat.add(&base);
    let (mut cids, mut cd) = (vec![0i64; nq * k], vec![0f32; nq * k]);

    flat.search_batch(&queries, k, &mut cids, &mut cd, 1);
    let t = Instant::now();
    flat.search_batch(&queries, k, &mut cids, &mut cd, 1);
    let cpu1 = t.elapsed().as_secs_f64();

    flat.search_batch(&queries, k, &mut cids, &mut cd, 0);
    let t = Instant::now();
    flat.search_batch(&queries, k, &mut cids, &mut cd, 0);
    let cpu_mt = t.elapsed().as_secs_f64();

    let gpu = CudaKnn::new(&base, dim, metric).expect("GPU init");
    let (mut gids, mut gd) = (vec![0i64; nq * k], vec![0f32; nq * k]);
    gpu.search(&queries, k, &mut gids, &mut gd).expect("GPU warmup");
    let t = Instant::now();
    gpu.search(&queries, k, &mut gids, &mut gd).expect("GPU search");
    let gpu_t = t.elapsed().as_secs_f64();

    let mut agree = 0usize;
    for q in 0..nq {
        let cpu_set: std::collections::HashSet<i64> =
            cids[q * k..(q + 1) * k].iter().copied().collect();
        agree += gids[q * k..(q + 1) * k].iter().filter(|id| cpu_set.contains(id)).count();
    }
    println!("\n  CPU flat (1 thread) : {cpu1:8.3} s   ({:>8.0} q/s)", nq as f64 / cpu1);
    println!("  CPU flat (all cores): {cpu_mt:8.3} s   ({:>8.0} q/s)", nq as f64 / cpu_mt);
    println!("  GPU exact           : {gpu_t:8.3} s   ({:>8.0} q/s)", nq as f64 / gpu_t);
    println!("\n  speedup vs CPU 1-thread : {:.1}x", cpu1 / gpu_t);
    println!("  speedup vs CPU all-core : {:.1}x", cpu_mt / gpu_t);
    let agreement = if nq * k == 0 {
        println!("  GPU/CPU top-{k} agreement : n/a (nq={nq}, k={k}: nothing to compare)");
        None
    } else {
        let agreement = agree as f64 / (nq * k) as f64;
        println!("  GPU/CPU top-{k} agreement : {agreement:.4}  (expect ~1.0, both exact)");
        assert!(agreement > 0.99, "GPU results disagree with exact CPU: kernel bug");
        Some(agreement)
    };

    let artifact = repo_root().join("bench").join("results").join("gpu.json");
    let body = build_json(&a, &gpu_info, &cpu, cores, n, dim, nq, k, cpu1, cpu_mt, gpu_t,
                          agreement);
    match write_artifact(&artifact, &body) {
        Ok(()) => println!("\n  wrote {}", artifact.display()),
        Err(e) => println!("\n  could not write {}: {e}", artifact.display()),
    }

    println!("\n  OK: GPU exact search matches the CPU ground truth.");
}
