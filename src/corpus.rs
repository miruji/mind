use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::rc::Rc;

pub const MAXBAG: usize = 6;
/// Longest turn kept from the noisy support logs.
pub const MAXTURNTOKENS: usize = 40;

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

pub struct Dialogue { pub turns: Vec<Vec<String>> }

/// Document frequency over prompts; used to keep only discriminative words of
/// the user message when it is used as conditioning context.
pub struct PromptModel { df: HashMap<String, u32>, n: u32 }

impl PromptModel {
    pub fn build<I, D>(dialogs: I) -> PromptModel
    where I: Iterator<Item = D>, D: Borrow<Dialogue> {
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut n = 0u32;
        for d in dialogs {
            let d = d.borrow();
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

/// The Cornell Movie-Dialog Corpus lives in two plain text files:
/// `movie_lines.txt` keeps every utterance, `movie_conversations.txt` keeps
/// which utterances form one dialog. The fields are separated by ` +++$+++ `.
pub const MOVIESEP: &str = " +++$+++ ";

/// The whole corpus in memory: the utterances and the dialogs they form.
pub struct MovieCorpus {
    /// Text of every utterance, in the order of `movie_lines.txt`.
    texts: Vec<String>,
    /// `L1045` -> index in `texts`.
    index: HashMap<String, u32>,
    /// Dialogues: the utterances of each dialog, in order.
    pub conversations: Vec<Vec<u32>>,
    pub utterances: usize,
}

impl MovieCorpus {
    pub fn load(dir: &Path) -> std::io::Result<MovieCorpus> {
        let mut corpus = MovieCorpus {
            texts: Vec::new(),
            index: HashMap::new(),
            conversations: Vec::new(),
            utterances: 0,
        };
        // pass 1: the utterances
        // The files are read byte by byte: a few lines of the corpus hold
        // latin-1 characters, and they must not stop the reading.
        let mut reader = BufReader::with_capacity(1 << 20, fs::File::open(dir.join("movie_lines.txt"))?);
        let mut buf: Vec<u8> = Vec::with_capacity(256);
        loop {
            buf.clear();
            let n = reader.read_until(b'\n', &mut buf)?;
            if n == 0 { break; }
            let line = String::from_utf8_lossy(&buf);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.trim().is_empty() { continue; }
            let mut parts = line.splitn(5, MOVIESEP);
            let id = parts.next().unwrap_or("").trim().to_string();
            let _character = parts.next();
            let _movie = parts.next();
            let _name = parts.next();
            let text = parts.next().unwrap_or("").trim().to_string();
            if id.is_empty() || text.is_empty() { continue; }
            let idx = corpus.texts.len() as u32;
            corpus.texts.push(text);
            corpus.index.insert(id, idx);
        }
        corpus.utterances = corpus.texts.len();
        // pass 2: the dialogs
        let mut reader = BufReader::with_capacity(1 << 20, fs::File::open(dir.join("movie_conversations.txt"))?);
        let mut buf: Vec<u8> = Vec::with_capacity(256);
        loop {
            buf.clear();
            let n = reader.read_until(b'\n', &mut buf)?;
            if n == 0 { break; }
            let line = String::from_utf8_lossy(&buf);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.trim().is_empty() { continue; }
            let mut parts = line.splitn(4, MOVIESEP);
            parts.next();
            parts.next();
            parts.next();
            let list = parts.next().unwrap_or("");
            let mut conv: Vec<u32> = Vec::new();
            for raw in list.trim().trim_start_matches('[').trim_end_matches(']').split(',') {
                let id = raw.trim().trim_matches('\'').trim_matches('"');
                if id.is_empty() { continue; }
                if let Some(&i) = corpus.index.get(id) { conv.push(i); }
            }
            if conv.len() >= 2 { corpus.conversations.push(conv); }
        }
        Ok(corpus)
    }

    /// Text of one utterance.
    pub fn text(&self, idx: u32) -> &str {
        self.texts.get(idx as usize).map(|s| s.as_str()).unwrap_or("")
    }
}

/// Shared progress counters of a streamed corpus.
#[derive(Clone)]
pub struct CorpusStatus {
    pub files: Rc<Cell<u64>>,
    pub totalFiles: Rc<Cell<u64>>,
    pub turns: Rc<Cell<u64>>,
    /// Files that were already done when this chunk started.
    pub chunkStart: Rc<Cell<u64>>,
}

impl CorpusStatus {
    pub fn new() -> CorpusStatus {
        CorpusStatus {
            files: Rc::new(Cell::new(0)),
            totalFiles: Rc::new(Cell::new(0)),
            turns: Rc::new(Cell::new(0)),
            chunkStart: Rc::new(Cell::new(0)),
        }
    }
}

/// Counts the dialogs of the whole corpus.
pub fn countDialogs(corpus: &MovieCorpus) -> u64 {
    corpus.conversations.len() as u64
}

/// Streams the dialogs of the movie corpus.
pub struct MovieDialogs {
    corpus: Rc<MovieCorpus>,
    convIdx: usize,
    pub read: u64,
    pub turns: u64,
    pub status: CorpusStatus,
    limit: u64,
    skip: u64,
}

impl MovieDialogs {
    pub fn new(corpus: Rc<MovieCorpus>) -> MovieDialogs {
        MovieDialogs::withLimit(corpus, 0, 0)
    }

    /// `limit` caps the number of dialogs read from the start of the corpus
    /// (0 = all of them), `skip` ignores the dialogs that were already read in
    /// an earlier chunk. Both count from the same origin, so a chain of
    /// sessions with growing limits walks through the corpus once.
    pub fn withLimit(corpus: Rc<MovieCorpus>, limit: u64, skip: u64) -> MovieDialogs {
        let status = CorpusStatus::new();
        let total = countDialogs(&corpus);
        status.totalFiles.set(if limit > 0 { limit } else { total });
        MovieDialogs { corpus, convIdx: 0, read: 0, turns: 0, status, limit, skip }
    }
}

impl Iterator for MovieDialogs {
    type Item = Dialogue;
    fn next(&mut self) -> Option<Dialogue> {
        loop {
            if self.limit > 0 && self.read >= self.limit { return None; }
            if self.convIdx >= self.corpus.conversations.len() { return None; }
            let conv = &self.corpus.conversations[self.convIdx];
            self.convIdx += 1;
            self.read += 1;
            if self.skip > 0 {
                self.skip -= 1;
                self.status.files.set(self.read);
                continue;
            }
            let mut turns: Vec<Vec<String>> = Vec::new();
            for &line in conv {
                let tokens = tokenizeDialogue(self.corpus.text(line));
                if tokens.is_empty() { continue; }
                turns.push(tokens.into_iter().take(MAXTURNTOKENS).collect());
            }
            if turns.len() < 2 { continue; }
            self.turns += turns.len() as u64;
            self.status.files.set(self.read);
            self.status.turns.set(self.turns);
            return Some(Dialogue { turns });
        }
    }
}

/// Splits a stream into a training part and a small held out part.
pub struct SplitDialogs<I> {
    pub inner: I,
    pub counter: u64,
    holdEvery: u64,
    holdCap: usize,
    pub held: Vec<Dialogue>,
}

impl<I> SplitDialogs<I> {
    /// Continues an interrupted run: the holdout pattern stays the same.
    pub fn at(inner: I, counter: u64, holdEvery: u64, holdCap: usize) -> SplitDialogs<I> {
        SplitDialogs { inner, counter, holdEvery: holdEvery.max(1), holdCap, held: Vec::new() }
    }
}

impl<I: Iterator<Item = Dialogue>> Iterator for SplitDialogs<I> {
    type Item = Dialogue;
    fn next(&mut self) -> Option<Dialogue> {
        loop {
            let d = self.inner.next()?;
            self.counter += 1;
            if self.counter % self.holdEvery == 0 {
                if self.held.len() < self.holdCap { self.held.push(d); }
                continue;
            }
            return Some(d);
        }
    }
}
