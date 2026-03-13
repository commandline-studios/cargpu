use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cargpu")]
#[command(about = "Drop-in cargo replacement with GPU acceleration", long_about = None)]
#[command(version = env!("CARGO_PKG_VERSION"))]
pub struct Args {
    #[arg(short = 'v', long, help = "Verbose output")]
    pub verbose: bool,

    #[arg(short = 'q', long, help = "Quiet output")]
    pub quiet: bool,

    #[arg(long, help = "Show detailed compilation logs")]
    pub logs: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Create a new cargpu project")]
    New {
        #[arg(help = "Project name")]
        name: String,
    },

    #[command(about = "Compile the current package")]
    Build {
        #[arg(short, long, help = "Build artifacts in release mode")]
        release: bool,

        #[arg(short, long, help = "Package to build")]
        package: Option<String>,

        #[arg(long, help = "Binary to build")]
        bin: Option<String>,

        #[arg(long, help = "Example to build")]
        example: Option<String>,

        #[arg(long, help = "Target triple to build for")]
        target: Option<String>,

        #[arg(short, long, help = "Space-separated list of features to activate")]
        features: Option<Vec<String>>,

        #[arg(long, help = "Do not activate the `default` feature")]
        no_default_features: bool,
    },

    #[command(about = "Run a binary or example of the local package")]
    Run {
        #[arg(short, long, help = "Build artifacts in release mode")]
        release: bool,

        #[arg(short, long, help = "Package to run")]
        package: Option<String>,

        #[arg(long, help = "Binary to run")]
        bin: Option<String>,

        #[arg(long, help = "Example to run")]
        example: Option<String>,

        #[arg(long, help = "Target triple to run for")]
        target: Option<String>,

        #[arg(short, long, help = "Space-separated list of features to activate")]
        features: Option<Vec<String>>,

        #[arg(long, help = "Do not activate the `default` feature")]
        no_default_features: bool,

        #[arg(last = true, help = "Arguments for the binary to run")]
        args: Vec<String>,
    },

    #[command(about = "Analyze the current package and report errors")]
    Check {
        #[arg(short, long, help = "Package to check")]
        package: Option<String>,

        #[arg(long, help = "Binary to check")]
        bin: Option<String>,

        #[arg(long, help = "Example to check")]
        example: Option<String>,

        #[arg(long, help = "Target triple to check for")]
        target: Option<String>,

        #[arg(short, long, help = "Space-separated list of features to activate")]
        features: Option<Vec<String>>,

        #[arg(long, help = "Do not activate the `default` feature")]
        no_default_features: bool,
    },

    #[command(about = "Remove artifacts that cargo has generated")]
    Clean {
        #[arg(short, long, help = "Package to clean")]
        package: Option<String>,

        #[arg(long, help = "Whether to remove release artifacts")]
        release: bool,

        #[arg(long, help = "Path to target directory")]
        target_dir: Option<PathBuf>,
    },
}
