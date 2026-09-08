//! Text generation backend on the llama.cpp release libraries: the same
//! `dlopen` route as `crate::mediagen`, serving llama-server's endpoints.

pub mod chat;
pub mod engine;
pub mod server;
