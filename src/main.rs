use clap::{Parser, Subcommand};
use llmman::{cmd, daemon, ffi, hostgpu};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "llmman",
    about = "Run any agent on any model, models stored as OCI images",
    version = env!("LLMMAN_VERSION"),
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch an integration
    Launch(cmd::launch::LaunchArgs),
    /// Run a model interactively or with a one-shot prompt
    Run(Box<cmd::run::RunArgs>),
    /// Package model files into a local OCI image
    Build(cmd::build::BuildArgs),
    /// Log in to a container registry or HuggingFace
    Login(cmd::login::LoginArgs),
    /// Log out from a container registry or HuggingFace
    Logout(cmd::logout::LogoutArgs),
    /// Push a local image to a registry
    Push(cmd::push::PushArgs),
    /// Pull an image from a registry to the local store
    Pull(cmd::pull::PullArgs),
    /// Pull (if needed) and print a model's local path as JSON (internal,
    /// used by the vllm-llmman plugin)
    #[command(hide = true)]
    Resolve(cmd::resolve::ResolveArgs),
    /// Transfer an image directly from one location to another (e.g. HuggingFace to an OCI registry)
    Transfer(cmd::transfer::TransferArgs),
    /// Check a registry model's signatures against trusted public keys
    Verify(cmd::verify::VerifyArgs),
    /// List locally stored images
    #[command(alias = "ls")]
    List(cmd::list::ListArgs),
    /// List models currently loaded by a running `llmman serve`
    Ps(cmd::ps::PsArgs),
    /// Show the prompts `llmman serve` has seen, newest first (like `git log`)
    Log(cmd::log::LogArgs),
    /// List the hosted providers `--provider` can route to
    Providers(cmd::providers::ProvidersArgs),
    /// Read and write llmman.conf settings
    Config(cmd::config::ConfigArgs),
    /// Copy a local image to a new reference
    Cp(cmd::cp::CpArgs),
    /// Remove a local image, freeing its blobs and extracted cache no
    /// longer referenced by any other model (set LLMMAN_NOPRUNE to skip)
    Rm(cmd::rm::RmArgs),
    /// Stop (unload) a running model
    Stop(cmd::stop::StopArgs),
    /// Show a local model's architecture, parameters, license, and template
    Show(cmd::show::ShowArgs),
    /// Start an inference server (Ollama, OpenAI, Anthropic compatible APIs)
    Serve(cmd::serve::ServeArgs),
    /// Probe the local host's GPU/accelerator support (internal diagnostic)
    #[command(hide = true)]
    GpuDiscover(cmd::gpu_discover::GpuDiscoverArgs),
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    // Deliberately checked before anything else in this function — not a
    // documented subcommand (absent from `Commands`/`--help` on purpose)
    // and not routed through clap at all: this is `hostgpu::detect`'s own
    // internal re-exec target, isolating its real CUDA/HIP/Vulkan FFI
    // probing (see that module's doc comment) in a disposable child
    // process of exactly this same binary. See
    // `hostgpu::probe_subprocess_main`'s own doc comment for why that
    // isolation exists at all, and why this has to run before
    // `ffi::ensure_runtime_init`/`daemon::disable_std_handle_inheritance`
    // below: this child is meant to do nothing but the one raw probe and
    // exit, as fast and dependency-free as possible.
    if std::env::args_os().nth(1).as_deref()
        == Some(std::ffi::OsStr::new(hostgpu::PROBE_SUBPROCESS_ARG))
    {
        hostgpu::probe_subprocess_main();
    }

    // Must happen before any other call into the `ffi` module, from every
    // process that links the Go shim in — both this CLI's own process and
    // the detached `llmman serve` daemon it spawns (a separate process,
    // with its own copy of the shim's Go runtime to bootstrap) each reach
    // this same `main()`. See ffi::ensure_runtime_init's own doc comment
    // for why this is necessary on Windows specifically.
    ffi::ensure_runtime_init();
    // As early as possible, before this process (directly, or via
    // daemon::ensure_server) can spawn anything else on Windows — see
    // daemon::disable_std_handle_inheritance's own doc comment for the
    // real E2E hang this fixes.
    daemon::disable_std_handle_inheritance();

    let cli = Cli::parse_from(cmd::log::expand_count_shorthand(std::env::args_os()));
    let result = match &cli.command {
        Commands::Launch(a) => cmd::launch::run(a),
        Commands::Run(a) => cmd::run::run(a),
        Commands::Build(a) => cmd::build::run(a),
        Commands::Login(a) => cmd::login::run(a),
        Commands::Logout(a) => cmd::logout::run(a),
        Commands::Push(a) => cmd::push::run(a),
        Commands::Pull(a) => cmd::pull::run(a),
        Commands::Resolve(a) => cmd::resolve::run(a),
        Commands::Transfer(a) => cmd::transfer::run(a),
        Commands::Verify(a) => cmd::verify::run(a),
        Commands::List(a) => cmd::list::run(a),
        Commands::Ps(a) => cmd::ps::run(a),
        Commands::Log(a) => cmd::log::run(a),
        Commands::Providers(a) => cmd::providers::run(a),
        Commands::Config(a) => cmd::config::run(a),
        Commands::Cp(a) => cmd::cp::run(a),
        Commands::Rm(a) => cmd::rm::run(a),
        Commands::Stop(a) => cmd::stop::run(a),
        Commands::Show(a) => cmd::show::run(a),
        Commands::Serve(a) => cmd::serve::run(a),
        Commands::GpuDiscover(a) => cmd::gpu_discover::run(a),
    };
    if let Err(e) = result {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}
