use anyhow::{anyhow, Result};
use std::path::PathBuf;
use std::process::Command;
use tracing::{debug, info, warn};

pub async fn clean(
    package: Option<String>,
    release: bool,
    target_dir: Option<PathBuf>,
) -> Result<()> {
    info!("Starting cargo clean operation");
    
    let mut cmd = Command::new("cargo");
    cmd.arg("clean");
    
    if let Some(pkg) = package {
        cmd.args(["--package", &pkg]);
    }
    
    if release {
        cmd.arg("--release");
    }
    
    if let Some(dir) = target_dir {
        cmd.args(["--target-dir", &dir.to_string_lossy()]);
    }
    
    debug!("Executing: {:?}", cmd);
    
    let output = cmd.output()?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("cargo clean failed: {}", stderr));
    }
    
    info!("Cargo clean completed successfully");
    Ok(())
}

pub fn get_cargo_version() -> Result<String> {
    let output = Command::new("cargo").arg("--version").output()?;
    
    if !output.status.success() {
        return Err(anyhow!("Failed to get cargo version"));
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn get_rustc_version() -> Result<String> {
    let output = Command::new("rustc").arg("--version").output()?;
    
    if !output.status.success() {
        return Err(anyhow!("Failed to get rustc version"));
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}