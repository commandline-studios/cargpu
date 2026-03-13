use anyhow::Result;
use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod cli;
mod compiler;
mod gpu;
mod cargo_compat;
mod error;
mod project_utils;

use compiler::{CarGPCompiler, BuildConfig, CheckConfig};

use cli::{Args, Commands};
use project_utils::create_new_project;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize logging with minimal output unless --logs is specified
    let log_level = if args.logs {
        "cargpu=debug,wgpu=info"
    } else if args.verbose {
        "cargpu=info,wgpu=warn"
    } else {
        "cargpu=warn,wgpu=error"
    };
    
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(log_level))
        )
        .init();

    if !args.quiet {
        info!("Starting cargpu v{}", env!("CARGO_PKG_VERSION"));
    }

    let quiet = args.quiet;
    match run_command(args).await {
        Ok(_) => {
            if !quiet {
                info!("Command completed successfully");
            }
            Ok(())
        }
        Err(e) => {
            error!("Command failed: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run_command(args: Args) -> Result<()> {
    match args.command {
        Commands::New { name } => {
            create_new_project(&name)
        }
        
        Commands::Build {
            release,
            package,
            bin,
            example,
            target,
            features,
            no_default_features,
        } => {
            let start_time = std::time::Instant::now();
            let mut compiler = CarGPCompiler::new_with_logs(args.verbose || args.logs, args.quiet, args.logs)?;
            
            let build_config = BuildConfig {
                release,
                package,
                bin,
                example,
                target,
                features,
                no_default_features,
            };
            
            let result = compiler.build(build_config).await;
            
            if !args.quiet {
                let duration = start_time.elapsed();
                println!("Compilation completed in {:?}", duration);
            }
            
            result
        }
        
        Commands::Run {
            release,
            package,
            bin,
            example,
            target,
            features,
            no_default_features,
            args: run_args,
        } => {
            let start_time = std::time::Instant::now();
            let mut compiler = CarGPCompiler::new_with_logs(args.verbose || args.logs, args.quiet, args.logs)?;
            
            let build_config = BuildConfig {
                release,
                package,
                bin,
                example,
                target,
                features,
                no_default_features,
            };
            
            let result = compiler.run(build_config, run_args).await;
            
            if !args.quiet {
                let duration = start_time.elapsed();
                println!("Compilation completed in {:?}", duration);
            }
            
            result
        }
        
        Commands::Check {
            package,
            bin,
            example,
            target,
            features,
            no_default_features,
        } => {
            let mut compiler = CarGPCompiler::new_with_logs(args.verbose || args.logs, args.quiet, args.logs)?;
            
            let check_config = CheckConfig {
                package,
                bin,
                example,
                target,
                features,
                no_default_features,
            };
            
            compiler.check(check_config).await
        }
        
        Commands::Clean {
            package,
            release,
            target_dir,
        } => {
            cargo_compat::clean(package, release, target_dir).await
        }
    }
}
