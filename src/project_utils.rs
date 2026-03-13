use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;

pub fn validate_project_name(name: &str) -> Result<()> {
    // Check for empty name
    if name.is_empty() {
        return Err(anyhow!("Project name cannot be empty"));
    }

    // Check for spaces
    if name.contains(' ') {
        return Err(anyhow!(
            "Project name cannot contain spaces. Use '_' or '-' instead"
        ));
    }

    // Check for leading digit
    if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        return Err(anyhow!("Project name cannot start with a digit"));
    }

    // Check for invalid characters
    let valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-";
    for c in name.chars() {
        if !valid_chars.contains(c) {
            return Err(anyhow!("Project name contains invalid character '{}'. Only letters, digits, '_' and '-' are allowed", c));
        }
    }

    // Check for uppercase letters and warn
    if name.chars().any(|c| c.is_ascii_uppercase()) {
        eprintln!("Warning: Project name contains uppercase letters. It's recommended to use lowercase for Rust projects.");
    }

    Ok(())
}

pub fn create_new_project(name: &str) -> Result<()> {
    // Validate project name first
    validate_project_name(name)?;

    let project_path = Path::new(name);

    // Check if directory already exists
    if project_path.exists() {
        return Err(anyhow!("Directory '{}' already exists", name));
    }

    // Create project directory
    fs::create_dir_all(project_path)?;

    // Create src directory
    let src_path = project_path.join("src");
    fs::create_dir_all(&src_path)?;

    // Create Cargo.toml
    let cargo_toml_content = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
"#,
        name
    );

    fs::write(project_path.join("Cargo.toml"), cargo_toml_content)?;

    // Create main.rs
    let main_rs_content = r#"fn main() {
    println!("Hello, world!");
}
"#;

    fs::write(src_path.join("main.rs"), main_rs_content)?;

    println!(
        "Created new project '{}' at: {}",
        name,
        project_path.display()
    );
    println!("To get started:");
    println!("   cd {}", name);
    println!("   cargpu build");
    println!("   cargpu run");

    Ok(())
}
