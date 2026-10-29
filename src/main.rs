use rand::seq::SliceRandom;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Mask for 64-bit hash.
const MASK: u64 = u64::MAX;
/// World speed boundary.
const P1: f64 = 0.501;
/// Max retries for linear work.
const RETRYLIMIT: usize = 5000;
/// Weight increment.
const INCREMENT: f64 = 0.1;
/// Context window size.
const CONTEXTSIZE: usize = 2;
/// Number of proposals.
const PROPOSALCOUNT: usize = 3;
/// Training cycles.
const CYCLES: usize = 2;
/// Max tokens per document.
const MAXTOKENSPERDOC: usize = 150;
/// Min tokens to keep document.
const MINTOKENSPERDOC: usize = 20;
/// Labeled documents count.
const LABELEDDOCS: usize = 8;
/// Unlabeled documents count.
const UNLABELEDDOCS: usize = 8;
/// Test documents count.
const TESTDOCS: usize = 5;
/// Help subdirectories.
const HELPDIRS: &[&str] = &["manual", "command", "policy", "cpack_gen", "guide"];

/// Splitmix64-style unit from nanoseconds.
fn timeToUnit(ns: u64) -> f64 {
    let mut x: u64 = ns.wrapping_add(0x9E3779B97F4A7C15) & MASK;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9) & MASK;
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB) & MASK;
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// Current time in nanoseconds.
fn nowNs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

/// Physical linear work with clock binding.
struct LinearWork {
    p1: f64,
    lastNs: u64,
    calls: u64,
    successes: u64,
    exhausted: u64,
}

impl LinearWork {
    /// Create with given p1.
    fn new(p1: f64) -> Self {
        Self { p1, lastNs: 0, calls: 0, successes: 0, exhausted: 0 }
    }

    /// Wait until success or limit.
    fn waitForSuccess(&mut self) -> bool {
        let p1: f64 = self.p1;
        let low: f64 = 1.0 - p1;
        let mut lastNs: u64 = self.lastNs;
        for attempt in 1..=RETRYLIMIT {
            let mut ns: u64 = nowNs();
            if ns <= lastNs {
                ns = lastNs + 1;
            }
            lastNs = ns;
            let u: f64 = timeToUnit(ns);
            if u < p1 && u > low {
                self.lastNs = lastNs;
                self.calls += attempt as u64;
                self.successes += 1;
                return true;
            }
        }
        self.lastNs = lastNs;
        self.calls += RETRYLIMIT as u64;
        self.exhausted += 1;
        false
    }

    /// Single unit draw.
    fn unit(&mut self) -> f64 {
        let mut ns: u64 = nowNs();
        if ns <= self.lastNs {
            ns = self.lastNs + 1;
        }
        self.lastNs = ns;
        timeToUnit(ns)
    }
}

/// Fixed-size context.
type Context = [String; 2];

/// Build context from tokens.
fn makeContext(tokens: &[String], index: usize) -> Context {
    let mut ctx: Context = ["<BOS>".to_string(), "<BOS>".to_string()];
    let start: usize = index.saturating_sub(CONTEXTSIZE);
    let slice: &[String] = &tokens[start..index];
    let offset: usize = CONTEXTSIZE - slice.len();
    for (i, t) in slice.iter().enumerate() {
        ctx[offset + i] = t.clone();
    }
    ctx
}

/// Feature list from context.
fn contextFeatures(ctx: &Context) -> Vec<String> {
    let mut f: Vec<String> = vec![
        format!("prev:{}", ctx[1]),
        format!("pair:{}|{}", ctx[0], ctx[1]),
    ];
    if ctx[0] != "<BOS>" {
        f.push(format!("prev2:{}", ctx[0]));
    }
    f
}

/// Linear transition table for subconscious.
struct Subconscious {
    globalFreq: HashMap<String, u32>,
    commonTokens: Vec<String>,
    weights: HashMap<Context, HashMap<String, f64>>,
    updateCount: u64,
}

impl Subconscious {
    /// Build from frequency map.
    fn new(globalFreq: HashMap<String, u32>) -> Self {
        let mut common: Vec<(String, u32)> = globalFreq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self {
            globalFreq,
            commonTokens: common.into_iter().map(|(k, _)| k).collect(),
            weights: HashMap::new(),
            updateCount: 0,
        }
    }

    /// Top-k proposals for context.
    fn top(&self, ctx: &Context, k: usize) -> Vec<String> {
        let mut ranked: Vec<String> = if let Some(out) = self.weights.get(ctx) {
            let mut v: Vec<(String, f64)> = out.iter().map(|(t, w)| (t.clone(), *w)).collect();
            v.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap()
                    .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0)))
                    .then(a.0.cmp(&b.0))
            });
            v.into_iter().map(|(t, _)| t).collect()
        } else {
            Vec::new()
        };
        for t in &self.commonTokens {
            if ranked.len() >= k {
                break;
            }
            if !ranked.contains(t) {
                ranked.push(t.clone());
            }
        }
        ranked.truncate(k);
        ranked
    }

    /// Apply conscious action.
    fn receiveConsciousAction(&mut self, ctx: &Context, token: &str) {
        *self.weights.entry(ctx.clone()).or_default().entry(token.to_string()).or_insert(0.0) += INCREMENT;
        self.updateCount += 1;
    }
}

/// Non-linear choice network for consciousness.
struct Consciousness {
    globalFreq: HashMap<String, u32>,
    commonTokens: Vec<String>,
    edges: HashMap<String, HashMap<String, f64>>,
    totals: HashMap<String, f64>,
    updateCount: u64,
    actions: u64,
    failedUpdates: u64,
}

impl Consciousness {
    /// Build from frequency map.
    fn new(globalFreq: HashMap<String, u32>) -> Self {
        let mut common: Vec<(String, u32)> = globalFreq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self {
            globalFreq,
            commonTokens: common.into_iter().map(|(k, _)| k).collect(),
            edges: HashMap::new(),
            totals: HashMap::new(),
            updateCount: 0,
            actions: 0,
            failedUpdates: 0,
        }
    }

    /// Features for context and proposals.
    fn activeFeatures(&self, ctx: &Context, proposals: &[String]) -> Vec<String> {
        let mut f: Vec<String> = contextFeatures(ctx);
        for t in proposals {
            f.push(format!("proposal:{}", t));
        }
        f
    }

    /// Train on external target.
    fn receiveTrainingTrigger(
        &mut self,
        work: &mut LinearWork,
        ctx: &Context,
        proposals: &[String],
        target: &str,
    ) -> bool {
        if !work.waitForSuccess() {
            self.failedUpdates += 1;
            return false;
        }
        for feature in self.activeFeatures(ctx, proposals) {
            *self.edges.entry(feature.clone()).or_default().entry(target.to_string()).or_insert(0.0) += INCREMENT;
            *self.totals.entry(feature).or_insert(0.0) += INCREMENT;
        }
        self.updateCount += 1;
        true
    }

    /// Candidate tokens from features.
    fn candidatePool(&self, ctx: &Context, proposals: &[String], kPer: usize) -> Vec<String> {
        let mut pool: Vec<String> = proposals.to_vec();
        let mut seen: HashSet<String> = pool.iter().cloned().collect();
        for feature in self.activeFeatures(ctx, proposals) {
            if let Some(out) = self.edges.get(&feature) {
                let mut ranked: Vec<(String, f64)> = out.iter().map(|(t, w)| (t.clone(), *w)).collect();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap()
                        .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0)))
                        .then(a.0.cmp(&b.0))
                });
                for (t, _) in ranked.into_iter().take(kPer) {
                    if !seen.contains(&t) {
                        seen.insert(t.clone());
                        pool.push(t);
                    }
                }
            }
        }
        if pool.is_empty() {
            pool = self.commonTokens.iter().take(PROPOSALCOUNT).cloned().collect();
        }
        pool
    }

    /// Score candidates.
    fn scores(&self, ctx: &Context, proposals: &[String], candidates: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = candidates.iter().map(|t| (t.clone(), 0.0)).collect();
        for feature in self.activeFeatures(ctx, proposals) {
            let total: f64 = *self.totals.get(&feature).unwrap_or(&0.0);
            if total <= 0.0 {
                continue;
            }
            if let Some(out) = self.edges.get(&feature) {
                for t in candidates {
                    if let Some(w) = out.get(t) {
                        *scores.get_mut(t).unwrap() += w / total;
                    }
                }
            }
        }
        for (rank, t) in proposals.iter().enumerate() {
            *scores.entry(t.clone()).or_insert(0.0) += 0.12 / (rank as f64 + 1.0);
        }
        scores
    }

    /// Ranked top-k.
    fn rank(&self, ctx: &Context, proposals: &[String], k: usize) -> Vec<String> {
        let cands: Vec<String> = self.candidatePool(ctx, proposals, 3);
        let scores: HashMap<String, f64> = self.scores(ctx, proposals, &cands);
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap()
                .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0)))
                .then(a.0.cmp(&b.0))
        });
        ranked.into_iter().map(|(t, _)| t).take(k).collect()
    }

    /// Choose action via linear work.
    fn chooseAction(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String]) -> Option<String> {
        if !work.waitForSuccess() {
            self.failedUpdates += 1;
            return None;
        }
        let candidates: Vec<String> = self.candidatePool(ctx, proposals, 3);
        let scores: HashMap<String, f64> = self.scores(ctx, proposals, &candidates);
        let weights: Vec<f64> = candidates.iter().map(|t| scores.get(t).copied().unwrap_or(0.0).max(0.0)).collect();
        let total: f64 = weights.iter().sum();
        let chosen: String = if total <= 0.0 {
            proposals.first().cloned().or_else(|| candidates.first().cloned()).unwrap_or_default()
        } else {
            let threshold: f64 = work.unit() * total;
            let mut acc: f64 = 0.0;
            let mut ch: String = candidates.last().cloned().unwrap_or_default();
            for (t, w) in candidates.iter().zip(weights.iter()) {
                acc += *w;
                if threshold < acc {
                    ch = t.clone();
                    break;
                }
            }
            ch
        };
        self.actions += 1;
        Some(chosen)
    }
}

/// Tokenize rst file.
fn tokenize(path: &Path, limit: usize) -> Vec<String> {
    let raw: String = fs::read_to_string(path).unwrap_or_default();
    let mut kept: Vec<String> = Vec::new();
    let mut inCode: bool = false;
    let reCode: Regex = Regex::new(r"^\s*\.\.\s+(code-block|code::|sourcecode::|literalinclude)::").unwrap();
    let reDir: Regex = Regex::new(r"^\s*[=*#^~`:+-]{3,}\s*$").unwrap();
    let reRole: Regex = Regex::new(r":\w+:`([^`]+)`").unwrap();
    for line in raw.lines() {
        if reCode.is_match(line) {
            inCode = true;
            continue;
        }
        if inCode {
            if line.trim().is_empty() || line.starts_with(' ') || line.starts_with('\t') {
                continue;
            }
            inCode = false;
        }
        if line.trim_start().starts_with(".. ") {
            continue;
        }
        if reDir.is_match(line) {
            continue;
        }
        if line.starts_with("   ") || line.starts_with('\t') {
            continue;
        }
        let cleaned: String = reRole.replace_all(line, "$1").replace("``", " ").replace('`', " ");
        kept.push(cleaned);
    }
    let text: String = kept.join(" ").to_lowercase();
    let reTok: Regex = Regex::new(r"[a-z]+(?:[-'][a-z]+)*").unwrap();
    reTok.find_iter(&text).map(|m| m.as_str().to_string()).take(limit).collect()
}

/// Recursively collect rst paths.
fn collectRst(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path: PathBuf = entry.path();
            if path.is_dir() {
                collectRst(&path, out);
            } else if path.extension().map(|e| e == "rst").unwrap_or(false) {
                out.push(path);
            }
        }
    }
}

/// Load CMake help documents.
fn loadDocs() -> HashMap<String, Vec<String>> {
    let mut docs: HashMap<String, Vec<String>> = HashMap::new();
    let base: &Path = Path::new("data");
    for dir in HELPDIRS {
        let p: PathBuf = base.join(dir);
        if !p.exists() {
            continue;
        }
        let mut paths: Vec<PathBuf> = Vec::new();
        collectRst(&p, &mut paths);
        for path in paths {
            let tokens: Vec<String> = tokenize(&path, MAXTOKENSPERDOC);
            if tokens.len() >= MINTOKENSPERDOC {
                let name: String = format!("{}/{}", dir, path.file_name().unwrap().to_string_lossy());
                docs.insert(name, tokens);
            }
        }
    }
    docs
}

/// Context-target pairs from docs.
fn contextPairs(docs: &[Vec<String>]) -> Vec<(Context, String)> {
    let mut pairs: Vec<(Context, String)> = Vec::new();
    for tokens in docs {
        for i in 1..tokens.len() {
            pairs.push((makeContext(tokens, i), tokens[i].clone()));
        }
    }
    pairs
}

/// Evaluate subconscious metrics.
fn evaluateSub(sub: &Subconscious, pairs: &[(Context, String)]) -> (f64, f64, f64) {
    let n: usize = pairs.len().max(1);
    let mut top1: usize = 0;
    let mut top3: usize = 0;
    let mut seen: usize = 0;
    for (ctx, target) in pairs {
        let preds: Vec<String> = sub.top(ctx, 3);
        if preds.first().map(|p| p == target).unwrap_or(false) {
            top1 += 1;
        }
        if preds.iter().any(|p| p == target) {
            top3 += 1;
        }
        if sub.weights.contains_key(ctx) {
            seen += 1;
        }
    }
    (top1 as f64 / n as f64, top3 as f64 / n as f64, seen as f64 / n as f64)
}

/// Evaluate consciousness metrics.
fn evaluateConscious(sub: &Subconscious, cons: &Consciousness, pairs: &[(Context, String)]) -> (f64, f64, f64) {
    let n: usize = pairs.len().max(1);
    let mut top1: usize = 0;
    let mut top3: usize = 0;
    let mut inPool: usize = 0;
    for (ctx, target) in pairs {
        let proposals: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
        let ranked: Vec<String> = cons.rank(ctx, &proposals, 3);
        if ranked.first().map(|p| p == target).unwrap_or(false) {
            top1 += 1;
        }
        if ranked.iter().any(|p| p == target) {
            top3 += 1;
        }
        if cons.candidatePool(ctx, &proposals, 3).iter().any(|p| p == target) {
            inPool += 1;
        }
    }
    (top1 as f64 / n as f64, top3 as f64 / n as f64, inPool as f64 / n as f64)
}

fn main() {
    let docsMap: HashMap<String, Vec<String>> = loadDocs();
    println!("CORPUS=/data sample rst; usable documents={}", docsMap.len());
    println!("split: labeled={} unlabeled={} test={}; P1={}; cycles={}", LABELEDDOCS, UNLABELEDDOCS, TESTDOCS, P1, CYCLES);

    let mut names: Vec<String> = docsMap.keys().cloned().collect();
    names.sort();
    let mut rng = rand::thread_rng();
    names.shuffle(&mut rng);

    let n: usize = names.len();
    if n == 0 {
        println!("no documents loaded");
        return;
    }
    let labeledEnd: usize = LABELEDDOCS.min(n);
    let unlabeledEnd: usize = (LABELEDDOCS + UNLABELEDDOCS).min(n);
    let testEnd: usize = (LABELEDDOCS + UNLABELEDDOCS + TESTDOCS).min(n);
    let labeledNames: &[String] = &names[..labeledEnd];
    let unlabeledNames: &[String] = &names[labeledEnd..unlabeledEnd];
    let testNames: &[String] = &names[unlabeledEnd..testEnd];

    let labeled: Vec<Vec<String>> = labeledNames.iter().map(|n| docsMap[n].clone()).collect();
    let unlabeled: Vec<Vec<String>> = unlabeledNames.iter().map(|n| docsMap[n].clone()).collect();
    let test: Vec<Vec<String>> = testNames.iter().map(|n| docsMap[n].clone()).collect();

    let mut freq: HashMap<String, u32> = HashMap::new();
    for doc in &labeled {
        for t in doc {
            *freq.entry(t.clone()).or_insert(0) += 1;
        }
    }

    let testPairs: Vec<(Context, String)> = contextPairs(&test);
    let labeledPairs: Vec<(Context, String)> = contextPairs(&labeled);
    let unlabeledPairs: Vec<(Context, String)> = contextPairs(&unlabeled);

    let mut common: Vec<(String, u32)> = freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
    common.sort_by(|a, b| b.1.cmp(&a.1));
    let top3uni: Vec<String> = common.iter().take(3).map(|(k, _)| k.clone()).collect();
    let uniTop1: f64 = testPairs.iter().filter(|(_, t)| top3uni.first().map(|u| u == t).unwrap_or(false)).count() as f64 / testPairs.len().max(1) as f64;
    let uniTop3: f64 = testPairs.iter().filter(|(_, t)| top3uni.iter().any(|u| u == t)).count() as f64 / testPairs.len().max(1) as f64;

    let mut direct: Subconscious = Subconscious::new(freq.clone());
    for _ in 0..CYCLES {
        for (ctx, target) in &labeledPairs {
            direct.receiveConsciousAction(ctx, target);
        }
    }
    let (dirT1, dirT3, _): (f64, f64, f64) = evaluateSub(&direct, &testPairs);

    let mut work: LinearWork = LinearWork::new(P1);
    let mut sub: Subconscious = Subconscious::new(freq.clone());
    let mut cons: Consciousness = Consciousness::new(freq.clone());

    for _cycle in 0..CYCLES {
        for (ctx, target) in &labeledPairs {
            let props: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
            cons.receiveTrainingTrigger(&mut work, ctx, &props, target);
        }
        let mut stream: Vec<(Context, bool, String)> = labeledPairs.iter().map(|(c, t)| (c.clone(), true, t.clone())).collect();
        stream.extend(unlabeledPairs.iter().map(|(c, t)| (c.clone(), false, t.clone())));
        stream.shuffle(&mut rng);
        for (ctx, _, _) in &stream {
            let props: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
            if let Some(chosen) = cons.chooseAction(&mut work, ctx, &props) {
                sub.receiveConsciousAction(ctx, &chosen);
            }
        }
    }

    let (subT1, subT3, cov): (f64, f64, f64) = evaluateSub(&sub, &testPairs);
    let (consT1, consT3, pool): (f64, f64, f64) = evaluateConscious(&sub, &cons, &testPairs);

    let mut work2: LinearWork = LinearWork::new(P1);
    let mut consAlone: Consciousness = Consciousness::new(freq.clone());
    let subEmpty: Subconscious = Subconscious::new(freq.clone());
    for (ctx, target) in &labeledPairs {
        let props: Vec<String> = subEmpty.top(ctx, PROPOSALCOUNT);
        consAlone.receiveTrainingTrigger(&mut work2, ctx, &props, target);
    }
    let (aloneT1, aloneT3, _): (f64, f64, f64) = evaluateConscious(&subEmpty, &consAlone, &testPairs);

    println!();
    println!("labeledTokens={} unlabeledTokens={} testTokens={} vocabulary={}",
        labeled.iter().map(|d| d.len()).sum::<usize>(),
        unlabeled.iter().map(|d| d.len()).sum::<usize>(),
        testPairs.len(),
        freq.len());

    println!();
    println!("{:<48} {:>8} {:>8}", "model", "top1", "top3");
    println!("{}", "-".repeat(68));
    println!("{:<48} {:>7.2}% {:>7.2}%", "unigram (most frequent)", uniTop1 * 100.0, uniTop3 * 100.0);
    println!("{:<48} {:>7.2}% {:>7.2}%", "direct supervision of Subconscious", dirT1 * 100.0, dirT3 * 100.0);
    println!("{:<48} {:>7.2}% {:>7.2}%", "Subconscious after loop (withUnlabeled)", subT1 * 100.0, subT3 * 100.0);
    println!("{:<48} {:>7.2}% {:>7.2}%", "Consciousness without loop (Stage A only)", aloneT1 * 100.0, aloneT3 * 100.0);
    println!("{:<48} {:>7.2}% {:>7.2}%", "Consciousness after loop", consT1 * 100.0, consT3 * 100.0);

    println!();
    println!("DIAGNOSTICS");
    println!("test contexts known to Subconscious={:.1}% | target in candidate pool={:.1}%", cov * 100.0, pool * 100.0);
    println!("linearWork calls={} successes={} rate={:.6} exhausted={}", work.calls, work.successes, work.successes as f64 / work.calls.max(1) as f64, work.exhausted);
}
