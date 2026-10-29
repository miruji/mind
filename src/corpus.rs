use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

const HELPDIRS: &[&str] = &["manual", "command", "policy", "guide", "cpack_gen"];
const MAXTOKENSPERDOC: usize = 200;
const MINTOKENSPERDOC: usize = 20;

pub const MAXBAG: usize = 6;

const STOP: &[&str] = &[
    "the", "and", "you", "that", "have", "for", "not", "with", "are", "but", "was", "were", "this",
    "they", "them", "their", "there", "here", "what", "when", "where", "which", "who", "why", "how",
    "all", "any", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "did",
    "does", "doing", "has", "had", "been", "being", "into", "onto", "from", "than", "then", "too",
    "just", "get", "got", "goes", "going", "went", "say", "says", "said", "know", "knows", "think",
    "thinks", "thought", "want", "wants", "wanted", "need", "needs", "needed", "make", "makes",
    "made", "take", "takes", "took", "come", "comes", "came", "give", "gives", "gave", "really",
    "something", "someone", "thing", "things", "lot", "much", "more", "most", "other", "others",
    "such", "only", "also", "because", "about", "after", "before", "while", "both", "each", "few",
    "many", "some", "own", "same", "down", "over", "under", "again", "cannot", "let's", "gonna",
    "gotta", "yeah", "okay", "yes", "hello", "hi", "like", "well", "right", "sure", "never", "ever",
];

pub fn tokenize(path: &Path, limit: usize) -> Vec<String> {
    let raw = fs::read_to_string(path).unwrap_or_default().to_lowercase();
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in raw.chars() {
        if c.is_ascii_alphabetic() || c == '\'' || c == '-' { cur.push(c); }
        else if !cur.is_empty() { tokens.push(cur.clone()); cur.clear(); if tokens.len() >= limit { break; } }
    }
    if !cur.is_empty() && tokens.len() < limit { tokens.push(cur); }
    tokens
}

fn collectRst(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() { collectRst(&path, out); }
            else if path.extension().map(|e| e == "rst").unwrap_or(false) { out.push(path); }
        }
    }
}

pub fn loadDocs() -> HashMap<String, Vec<String>> {
    let mut docs = HashMap::new();
    let base = Path::new("data");
    for dir in HELPDIRS {
        let p = base.join(dir); if !p.exists() { continue; }
        let mut paths = Vec::new(); collectRst(&p, &mut paths);
        for path in paths {
            let tokens = tokenize(&path, MAXTOKENSPERDOC);
            if tokens.len() >= MINTOKENSPERDOC {
                let name = format!("{}/{}", dir, path.file_name().unwrap().to_string_lossy());
                docs.insert(name, tokens);
            }
        }
    }
    docs
}

/// Dialogue tokenizer: keeps punctuation as separate tokens so the model can
/// learn where a reply ends (".", "?", "!").
pub fn tokenizeDialogue(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_alphanumeric() || c == '\'' || c == '-' { cur.push(c); }
        else {
            if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            if c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' { out.push(c.to_string()); }
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

pub struct Dialogue { pub turns: Vec<Vec<String>>, pub acts: Vec<u8> }

fn parseDialogues(raw: &str, actsRaw: &[Vec<u8>]) -> Vec<Dialogue> {
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let mut turns = Vec::new();
        let mut turnActs = Vec::new();
        for (pi, part) in line.split("__eou__").enumerate() {
            let toks = tokenizeDialogue(part);
            if toks.is_empty() { continue; }
            turns.push(toks);
            turnActs.push(actsRaw.get(i).and_then(|a| a.get(pi)).copied().unwrap_or(1).clamp(1, 4));
        }
        if turns.len() < 2 { continue; }
        out.push(Dialogue { turns, acts: turnActs });
    }
    out
}

fn readActs(path: &Path) -> Vec<Vec<u8>> {
    fs::read_to_string(path).unwrap_or_default().lines()
        .map(|l| l.split_whitespace().filter_map(|t| t.parse::<u8>().ok()).collect()).collect()
}

/// Loads the DailyDialog corpus. Uses the official splits when the zip
/// archives have been extracted, otherwise a deterministic every-10th
/// holdout of the full corpus.
pub fn loadDialogues(root: &Path) -> (Vec<Dialogue>, Vec<Dialogue>) {
    let officialTrain = root.join("splits/train/train/dialogues_train.txt");
    let officialTest = root.join("splits/test/test/dialogues_test.txt");
    if officialTrain.exists() && officialTest.exists() {
        let train = parseDialogues(
            &fs::read_to_string(&officialTrain).unwrap_or_default(),
            &readActs(&root.join("splits/train/train/dialogues_act_train.txt")));
        let test = parseDialogues(
            &fs::read_to_string(&officialTest).unwrap_or_default(),
            &readActs(&root.join("splits/test/test/dialogues_act_test.txt")));
        return (train, test);
    }
    let raw = fs::read_to_string(root.join("dialogues_text.txt")).unwrap_or_default();
    let acts = readActs(&root.join("dialogues_act.txt"));
    let all = parseDialogues(&raw, &acts);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (i, d) in all.into_iter().enumerate() {
        if i % 10 == 0 { test.push(d); } else { train.push(d); }
    }
    (train, test)
}

/// Document frequency over prompts; used to keep only discriminative words of
/// the user message when it is used as conditioning context.
pub struct PromptModel { df: HashMap<String, u32>, n: u32 }

impl PromptModel {
    pub fn build(dialogs: &[Dialogue]) -> PromptModel {
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut n = 0u32;
        for d in dialogs {
            for i in 1..d.turns.len() {
                let mut seen = HashSet::new();
                for w in contentWords(&d.turns[i - 1]) {
                    if seen.insert(w.clone()) { *df.entry(w).or_insert(0) += 1; }
                }
                n += 1;
            }
        }
        PromptModel { df, n }
    }

    pub fn bag(&self, tokens: &[String]) -> Vec<String> {
        let mut scored: Vec<(f64, String)> = Vec::new();
        for w in contentWords(tokens) {
            let d = *self.df.get(&w).unwrap_or(&0);
            if d == 0 { continue; }
            let idf = (self.n as f64 + 1.0).ln() / (d as f64 + 1.0).ln();
            scored.push((idf, w));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        scored.into_iter().take(MAXBAG).map(|(_, w)| w).collect()
    }
}

pub fn contentWords(tokens: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in tokens {
        if t.len() < 3 { continue; }
        if !t.chars().all(|c| c.is_alphabetic()) { continue; }
        if STOP.contains(&t.as_str()) { continue; }
        if seen.insert(t.clone()) { out.push(t.clone()); }
    }
    out
}
