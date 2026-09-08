//! Chat template rendering: the model's own Jinja template, as llama-server
//! applies it, so the prompt tokens are the same.

use anyhow::{anyhow, Result};
use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

fn strftime_now(fmt: String) -> String {
    chrono::Local::now().format(&fmt).to_string()
}

/// Renders `messages` (OpenAI shape) with `template`, ending with the
/// assistant turn opener. `tools` is passed through for templates that
/// know about them; `thinking` is the template's `enable_thinking` knob.
pub fn render(template: &str, messages: &[Json], tools: Option<&Json>, bos: &str, eos: &str, thinking: bool) -> Result<String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    // the templates are written for Python's Jinja: dict.get, str.startswith, ...
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    minijinja_contrib::add_to_environment(&mut env);
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;
    // text-only content parts become the string llama-server would see
    let messages: Vec<Json> = messages.iter().map(flatten_content).collect();
    let out = tmpl
        .render(context! {
            messages => Value::from_serialize(&messages),
            tools => Value::from_serialize(&tools),
            add_generation_prompt => true,
            enable_thinking => thinking,
            bos_token => bos,
            eos_token => eos,
        })
        .map_err(|e| anyhow!("chat template: {e}"))?;
    Ok(out)
}

fn flatten_content(m: &Json) -> Json {
    let mut m = m.clone();
    if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
        let text: String = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("");
        m["content"] = Json::String(text);
    }
    if m.get("content").is_none() {
        m["content"] = Json::String(String::new());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn renders_chatml() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let out = render(CHATML, &msgs, None, "<s>", "</s>", true).unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn flattens_content_parts() {
        let msgs = vec![serde_json::json!({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})];
        let out = render(CHATML, &msgs, None, "", "", true).unwrap();
        assert!(out.contains("user\nab<|im_end|>"));
    }
}
