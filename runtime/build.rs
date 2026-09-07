use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").expect("Cargo must set TARGET");
    if target != "wasm32-wasip1-threads" {
        return;
    }

    // Rust cdylibs omit the executable/reactor CRT objects. A WASI threads
    // module still needs the reactor's `_initialize`, which sets the main
    // thread pointer before malloc, TLS, or synchronization can be used.
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let output = Command::new(rustc)
        .args(["--print", "target-libdir", "--target", &target])
        .output()
        .expect("query Rust target library directory");
    assert!(
        output.status.success(),
        "rustc failed to report the {target} library directory"
    );
    let target_libdir = String::from_utf8(output.stdout)
        .expect("Rust target library directory is not UTF-8")
        .trim()
        .to_owned();
    let reactor = PathBuf::from(target_libdir)
        .join("self-contained")
        .join("crt1-reactor.o");
    assert!(
        reactor.is_file(),
        "missing {}; install the {target} Rust target",
        reactor.display()
    );
    println!("cargo:rustc-link-arg={}", reactor.display());
}
