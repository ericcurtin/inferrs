//! What a model's chat template lets a request choose about thinking,
//! read off the template text: the `reasoning_effort` levels it compares
//! against, whether it reads `enable_thinking`, whether it thinks at
//! all. Qwen3.8 takes `low`/`medium`/`high`/`xhigh` and raises on
//! anything else; gpt-oss reads `reasoning_effort` but names only its
//! default; Qwen3.5 and Gemma 4 have just the switch; Llama 3 has none.
//! `llmman launch` offers an integration exactly these (see
//! `cmd::launch::opencode_variants`), never a level the model rejects.

/// Every `reasoning_effort` level any template or provider spells,
/// lowest first.
pub const EFFORT_LEVELS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// What a template that reads `reasoning_effort` without naming its
/// choices is taken to accept: gpt-oss's three, which every provider
/// takes too (see `cmd::serve::anthropic::portable_efforts`).
pub const DEFAULT_EFFORT_LEVELS: &[&str] = &["low", "medium", "high"];

/// A chat template's thinking controls.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ThinkingControls {
    /// The template mentions thinking or reasoning at all. llama-server
    /// can switch any such model off (`enable_thinking: false`, or an
    /// empty think block where the template has no switch).
    pub thinks: bool,
    /// The template reads `enable_thinking`, so a kwarg turns thinking on
    /// as well as off (Gemma 4 defaults it off).
    pub enable_thinking: bool,
    /// The `reasoning_effort` levels it distinguishes, lowest first; empty
    /// when it reads none.
    pub efforts: Vec<&'static str>,
}

impl ThinkingControls {
    /// The choices to offer, in cycle order: `none` for a model that
    /// thinks, then its levels, or `thinking` when a switch is all it has.
    pub fn choices(&self) -> Vec<&'static str> {
        if !self.thinks {
            return Vec::new();
        }
        let mut choices = vec!["none"];
        if !self.efforts.is_empty() {
            choices.extend(&self.efforts);
        } else if self.enable_thinking {
            choices.push("thinking");
        }
        choices
    }
}

/// Reads the thinking controls off `template`. Levels are the
/// [`EFFORT_LEVELS`] quoted in a Jinja block that names
/// `reasoning_effort` (or a variable derived from it); fewer than two
/// named — gpt-oss's `default("medium")` alone — means
/// [`DEFAULT_EFFORT_LEVELS`]. `none` is a switch, not a level.
pub fn thinking_controls(template: &str) -> ThinkingControls {
    let template = strip_comments(template);
    let lower = template.to_ascii_lowercase();
    let thinks = lower.contains("think") || lower.contains("reasoning");
    if !thinks {
        return ThinkingControls::default();
    }
    let enable_thinking = template.contains("enable_thinking");
    let efforts = if template.contains("reasoning_effort") {
        let named: Vec<&'static str> = EFFORT_LEVELS
            .iter()
            .copied()
            .filter(|level| {
                blocks(&template)
                    .filter(|block| block.contains("reasoning_effort"))
                    .any(|block| quoted_literals(block).any(|lit| lit == *level))
            })
            .collect();
        if named.len() >= 2 {
            named
        } else {
            DEFAULT_EFFORT_LEVELS.to_vec()
        }
    } else {
        Vec::new()
    };
    ThinkingControls {
        thinks,
        enable_thinking,
        efforts,
    }
}

/// `template` without its `{# ... #}` comments.
fn strip_comments(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{#") {
        out.push_str(&rest[..start]);
        match rest[start..].find("#}") {
            Some(end) => rest = &rest[start + end + 2..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Each `{% ... %}` and `{{ ... }}` of `template`, in order.
fn blocks(template: &str) -> impl Iterator<Item = &str> {
    let mut rest = template;
    std::iter::from_fn(move || {
        let start = rest.find("{%").into_iter().chain(rest.find("{{")).min()?;
        let close = if rest[start..].starts_with("{%") {
            "%}"
        } else {
            "}}"
        };
        let end = rest[start + 2..].find(close)?;
        let block = &rest[start + 2..start + 2 + end];
        rest = &rest[start + 2 + end + 2..];
        Some(block)
    })
}

/// The contents of every `'...'` and `"..."` in `text`, in order.
fn quoted_literals(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let start = rest.find(['\'', '"'])?;
        let quote = rest.as_bytes()[start] as char;
        let after = &rest[start + 1..];
        let end = after.find(quote)?;
        rest = &after[end + 1..];
        Some(&after[..end])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8 as shipped in `docker.io/ai/qwen3.8`: `high` is an alias
    /// for `xhigh`, so accepted too; the error message names levels only
    /// inside one long string.
    const QWEN3_8: &str = r#"
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort == 'high' %}
        {%- set resolved_reasoning_effort = 'xhigh' %}
    {%- endif %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
{%- endif %}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen3_8_names_its_levels_and_its_switch() {
        let controls = thinking_controls(QWEN3_8);
        assert_eq!(
            controls,
            ThinkingControls {
                thinks: true,
                enable_thinking: true,
                efforts: vec!["low", "medium", "high", "xhigh"],
            }
        );
        assert_eq!(
            controls.choices(),
            ["none", "low", "medium", "high", "xhigh"]
        );
        // Minified to one line, the same.
        let one_line = QWEN3_8.replace('\n', "");
        assert_eq!(thinking_controls(&one_line), controls);
    }

    /// gpt-oss reads `reasoning_effort` but names only its default.
    #[test]
    fn a_template_that_names_no_choices_gets_the_default_levels() {
        let gpt_oss = r#"
{%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\n\n" }}
"#;
        assert_eq!(
            thinking_controls(gpt_oss).efforts,
            DEFAULT_EFFORT_LEVELS.to_vec()
        );
    }

    /// Qwen3.5, GLM-5, Gemma 4, Nemotron 3: `enable_thinking`, no levels.
    #[test]
    fn a_switch_alone_offers_off_and_on() {
        let gemma4 = "{%- set enable_thinking = enable_thinking | default(false) -%}\
                      {%- if enable_thinking -%}<|think|>{%- endif -%}";
        let controls = thinking_controls(gemma4);
        assert!(controls.enable_thinking && controls.efforts.is_empty());
        assert_eq!(controls.choices(), ["none", "thinking"]);
    }

    /// Kimi K2.5, MiniMax M2.5: thinking with nothing a kwarg reads.
    #[test]
    fn thinking_without_a_kwarg_offers_only_off() {
        let kimi =
            "{%- if thinking is defined and thinking is false -%}<think></think>{%- endif -%}";
        assert_eq!(thinking_controls(kimi).choices(), ["none"]);
        let minimax = "{%- set reasoning_content = content.split('</think>')[0] -%}";
        assert_eq!(thinking_controls(minimax).choices(), ["none"]);
    }

    /// Llama 3.1, Phi-4: no thinking anywhere — a comment mentioning it
    /// does not count.
    #[test]
    fn a_plain_template_has_no_controls() {
        let llama = "{# no thinking here #}{{- '<|start_header_id|>' + message['role'] }}";
        assert_eq!(thinking_controls(llama), ThinkingControls::default());
        assert!(thinking_controls(llama).choices().is_empty());
        assert_eq!(thinking_controls(""), ThinkingControls::default());
    }

    /// Levels come out in cycle order, once each, without `none`, and
    /// only from blocks that name the variable.
    #[test]
    fn levels_are_ordered_deduplicated_and_scoped_to_the_variables_blocks() {
        let t = r#"{%- if reasoning_effort == "max" or reasoning_effort == "low" or reasoning_effort == "none" %}
{%- if reasoning_effort == 'low' %}{%- set unrelated = 'high' %}"#;
        assert_eq!(thinking_controls(t).efforts, vec!["low", "max"]);
    }

    #[test]
    fn blocks_and_literals_are_walked_in_order() {
        assert_eq!(
            blocks("a {% if x %}b{{ y }}c {% end").collect::<Vec<_>>(),
            [" if x ", " y "]
        );
        assert_eq!(
            quoted_literals(r#"x in ('a', "b c") ~ 'd"#).collect::<Vec<_>>(),
            ["a", "b c"]
        );
        assert_eq!(strip_comments("a{# b #}c{# open"), "ac");
    }
}
