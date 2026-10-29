use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MASK: u64 = u64::MAX;
const P1: f64 = 0.501;
const RETRYLIMIT: usize = 3000;
const INCREMENT: f64 = 0.1;
const CAP: f64 = 24.0;
const CONTEXTSIZE: usize = 2;
const PROPOSALCOUNT: usize = 3;
const MAXTOKENSPERDOC: usize = 200;
const MINTOKENSPERDOC: usize = 20;
const HELPDIRS: &[&str] = &["manual", "command", "policy", "guide", "cpack_gen"];
const FOCUSDECAY: f64 = 0.035;
const FOCUSGAIN: f64 = 0.28;
const FOCUSTHRESHOLD: f64 = 0.15;
const FOCUSMAX: f64 = 1.0;
const LIVESTEPS: u64 = 600;
const STEPDELAYMS: u64 = 20;

fn timeToUnit(ns: u64) -> f64 {
    let mut x = ns.wrapping_add(0x9E3779B97F4A7C15) & MASK;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9) & MASK;
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB) & MASK;
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / 9007199254740992.0)
}
fn nowNs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

struct LinearWork { p1: f64, lastNs: u64, calls: u64, successes: u64, exhausted: u64 }
impl LinearWork {
    fn new(p1: f64) -> Self { Self { p1, lastNs: 0, calls: 0, successes: 0, exhausted: 0 } }
    fn waitForSuccess(&mut self) -> bool {
        let p1 = self.p1; let low = 1.0 - p1; let mut lastNs = self.lastNs;
        for attempt in 1..=RETRYLIMIT {
            let mut ns = nowNs(); if ns <= lastNs { ns = lastNs + 1; } lastNs = ns;
            let u = timeToUnit(ns);
            if u < p1 && u > low { self.lastNs = lastNs; self.calls += attempt as u64; self.successes += 1; return true; }
        }
        self.lastNs = lastNs; self.calls += RETRYLIMIT as u64; self.exhausted += 1; false
    }
    fn unit(&mut self) -> f64 {
        let mut ns = nowNs(); if ns <= self.lastNs { ns = self.lastNs + 1; } self.lastNs = ns; timeToUnit(ns)
    }
}

type Context = [String; 2];
fn makeContext(tokens: &[String], index: usize) -> Context {
    let mut ctx = ["<BOS>".to_string(), "<BOS>".to_string()];
    let start = index.saturating_sub(CONTEXTSIZE);
    let slice = &tokens[start..index];
    let offset = CONTEXTSIZE - slice.len();
    for (i, t) in slice.iter().enumerate() { ctx[offset + i] = t.clone(); }
    ctx
}
fn contextFeatures(ctx: &Context) -> Vec<String> {
    let mut f = vec![format!("prev:{}", ctx[1]), format!("pair:{}|{}", ctx[0], ctx[1])];
    if ctx[0] != "<BOS>" { f.push(format!("prev2:{}", ctx[0])); }
    f
}

struct FeatureRow { weights: HashMap<String, f64>, total: f64, cap: Option<f64> }
impl FeatureRow {
    fn new(cap: Option<f64>) -> Self { Self { weights: HashMap::new(), total: 0.0, cap } }
    fn bump(&mut self, token: &str, amount: f64) {
        *self.weights.entry(token.to_string()).or_insert(0.0) += amount; self.total += amount;
        if let Some(cap) = self.cap {
            if self.total > cap * 1.5 {
                let scale = cap / self.total;
                for w in self.weights.values_mut() { *w *= scale; }
                let thr = cap * 0.0008;
                self.weights.retain(|_, w| *w >= thr);
                self.total = self.weights.values().sum();
            }
        }
    }
}

struct Subconscious {
    globalFreq: HashMap<String, u32>, commonTokens: Vec<String>,
    rows: HashMap<String, FeatureRow>, cap: Option<f64>,
    passiveWrites: u64, consciousWrites: u64,
}
impl Subconscious {
    fn new(globalFreq: HashMap<String, u32>, cap: Option<f64>) -> Self {
        let mut common: Vec<_> = globalFreq.iter().map(|(k,v)| (k.clone(),*v)).collect();
        common.sort_by(|a,b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self { globalFreq, commonTokens: common.into_iter().map(|(k,_)|k).collect(), rows: HashMap::new(), cap, passiveWrites: 0, consciousWrites: 0 }
    }
    fn row(&mut self, feature: &str) -> &mut FeatureRow {
        self.rows.entry(feature.to_string()).or_insert_with(|| FeatureRow::new(self.cap))
    }
    fn write(&mut self, ctx: &Context, token: &str, passive: bool) {
        for feature in contextFeatures(ctx) { self.row(&feature).bump(token, INCREMENT); }
        if passive { self.passiveWrites += 1; } else { self.consciousWrites += 1; }
    }
    fn receiveObservation(&mut self, ctx: &Context, token: &str) { self.write(ctx, token, true); }
    fn receiveConsciousAction(&mut self, ctx: &Context, token: &str) { self.write(ctx, token, false); }
    fn top(&self, ctx: &Context, k: usize) -> Vec<String> {
        let features = contextFeatures(ctx);
        let mut cand: HashSet<String> = HashSet::new();
        for feature in &features {
            if let Some(row) = self.rows.get(feature) {
                let mut ranked: Vec<_> = row.weights.iter().collect();
                ranked.sort_by(|a,b| b.1.partial_cmp(a.1).unwrap().then(a.0.cmp(b.0)));
                for (t,_) in ranked.into_iter().take(8) { cand.insert(t.clone()); }
            }
        }
        let mut scores: HashMap<String, f64> = HashMap::new();
        for t in &cand {
            let mut s = 0.0;
            for feature in &features {
                if let Some(row) = self.rows.get(feature) {
                    if let Some(w) = row.weights.get(t) { if row.total > 0.0 { s += w / row.total; } }
                }
            }
            scores.insert(t.clone(), s);
        }
        let mut ranked: Vec<String> = scores.keys().cloned().collect();
        ranked.sort_by(|a,b| scores[b].partial_cmp(&scores[a]).unwrap()
            .then(self.globalFreq.get(b).unwrap_or(&0).cmp(self.globalFreq.get(a).unwrap_or(&0))).then(a.cmp(b)));
        if ranked.len() < k {
            for t in &self.commonTokens {
                if !scores.contains_key(t) { ranked.push(t.clone()); }
                if ranked.len() >= k { break; }
            }
        }
        ranked.truncate(k); ranked
    }
}

struct Consciousness {
    globalFreq: HashMap<String, u32>, commonTokens: Vec<String>,
    edges: HashMap<String, HashMap<String, f64>>, totals: HashMap<String, f64>,
    actions: u64, failedUpdates: u64,
}
impl Consciousness {
    fn new(globalFreq: HashMap<String, u32>) -> Self {
        let mut common: Vec<_> = globalFreq.iter().map(|(k,v)| (k.clone(),*v)).collect();
        common.sort_by(|a,b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self { globalFreq, commonTokens: common.into_iter().map(|(k,_)|k).collect(), edges: HashMap::new(), totals: HashMap::new(), actions: 0, failedUpdates: 0 }
    }
    fn activeFeatures(&self, ctx: &Context, proposals: &[String]) -> Vec<String> {
        let mut f = contextFeatures(ctx);
        for t in proposals { f.push(format!("proposal:{}", t)); }
        f
    }
    fn receiveTrainingTrigger(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String], target: &str) -> bool {
        if !work.waitForSuccess() { self.failedUpdates += 1; return false; }
        for feature in self.activeFeatures(ctx, proposals) {
            *self.edges.entry(feature.clone()).or_default().entry(target.to_string()).or_insert(0.0) += INCREMENT;
            *self.totals.entry(feature).or_insert(0.0) += INCREMENT;
        }
        true
    }
    fn candidatePool(&self, ctx: &Context, proposals: &[String], kPer: usize) -> Vec<String> {
        let mut pool = proposals.to_vec();
        let mut seen: HashSet<String> = pool.iter().cloned().collect();
        for feature in self.activeFeatures(ctx, proposals) {
            if let Some(out) = self.edges.get(&feature) {
                let mut ranked: Vec<_> = out.iter().map(|(t,w)| (t.clone(),*w)).collect();
                ranked.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()
                    .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0))).then(a.0.cmp(&b.0)));
                for (t,_) in ranked.into_iter().take(kPer) {
                    if !seen.contains(&t) { seen.insert(t.clone()); pool.push(t); }
                }
            }
        }
        if pool.is_empty() { pool = self.commonTokens.iter().take(PROPOSALCOUNT).cloned().collect(); }
        pool
    }
    fn scores(&self, ctx: &Context, proposals: &[String], candidates: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = candidates.iter().map(|t| (t.clone(), 0.0)).collect();
        for feature in self.activeFeatures(ctx, proposals) {
            let total = *self.totals.get(&feature).unwrap_or(&0.0);
            if total <= 0.0 { continue; }
            if let Some(out) = self.edges.get(&feature) {
                for t in candidates {
                    if let Some(w) = out.get(t) { *scores.get_mut(t).unwrap() += w / total; }
                }
            }
        }
        for (rank, t) in proposals.iter().enumerate() {
            *scores.entry(t.clone()).or_insert(0.0) += 0.12 / (rank as f64 + 1.0);
        }
        scores
    }
    fn chooseAction(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String]) -> Option<String> {
        if !work.waitForSuccess() { self.failedUpdates += 1; return None; }
        let candidates = self.candidatePool(ctx, proposals, 3);
        let scores = self.scores(ctx, proposals, &candidates);
        let mut ranked = candidates;
        ranked.sort_by(|a,b| scores.get(b).unwrap_or(&0.0).partial_cmp(scores.get(a).unwrap_or(&0.0)).unwrap()
            .then(self.globalFreq.get(b).unwrap_or(&0).cmp(self.globalFreq.get(a).unwrap_or(&0))).then(a.cmp(b)));
        let chosen = if ranked.is_empty() {
            proposals.first().cloned().unwrap_or_default()
        } else if work.unit() < 0.10 && ranked.len() > 1 { ranked[1].clone() } else { ranked[0].clone() };
        self.actions += 1; Some(chosen)
    }
}

struct Focus {
    level: f64, consciousSteps: u64, totalSteps: u64, fades: u64, wasAwake: bool,
}
impl Focus {
    fn new() -> Self { Self { level: 0.55, consciousSteps: 0, totalSteps: 0, fades: 0, wasAwake: true } }
    fn isAwake(&self) -> bool { self.level >= FOCUSTHRESHOLD }
    fn probability(&self) -> f64 { self.level.clamp(0.0, 1.0) }
    fn onStep(&mut self, hadSurprise: bool) {
        self.totalSteps += 1;
        if hadSurprise { self.level = (self.level + FOCUSGAIN).min(FOCUSMAX); }
        else { self.level = (self.level - FOCUSDECAY).max(0.0); }
        let awake = self.isAwake();
        if awake { self.consciousSteps += 1; }
        if self.wasAwake && !awake { self.fades += 1; }
        self.wasAwake = awake;
    }
    fn avgPercent(&self) -> f64 {
        if self.totalSteps == 0 { 0.0 } else { self.consciousSteps as f64 / self.totalSteps as f64 * 100.0 }
    }
}

fn colorForFocus(level: f64) -> String {
    let t = level.clamp(0.0, 1.0);
    let (r, g) = if t < 0.5 { (255u8, (t * 2.0 * 220.0) as u8) } else { (((1.0 - t) * 2.0 * 220.0) as u8, 220u8) };
    format!("\x1b[38;2;{};{};40m", r, g)
}
fn resetColor() -> &'static str { "\x1b[0m" }
fn hitRate(hits: u64, steps: u64) -> f64 { if steps == 0 { 0.0 } else { hits as f64 / steps as f64 * 100.0 } }

fn tokenize(path: &Path, limit: usize) -> Vec<String> {
    let raw = fs::read_to_string(path).unwrap_or_default().to_lowercase();
    let mut tokens = Vec::new(); let mut cur = String::new();
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
fn loadDocs() -> HashMap<String, Vec<String>> {
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

fn main() {
    let docsMap = loadDocs();
    println!("CORPUS=data; documents={}", docsMap.len());
    println!("Live focus: decay={} gain={} threshold={}", FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD);
    println!("color: red=asleep → green=high focus | · passive  ! surprise  W will");
    println!("stats: t1/t3 = subconscious Top1/Top3 hit rate on the true next token");
    println!();

    let mut names: Vec<String> = docsMap.keys().cloned().collect();
    names.sort();
    let mut allTokens = Vec::new();
    for name in &names { allTokens.extend(docsMap[name].clone()); }
    if allTokens.len() < 10 { println!("not enough tokens"); return; }

    let mut freq: HashMap<String, u32> = HashMap::new();
    for t in &allTokens { *freq.entry(t.clone()).or_insert(0) += 1; }

    let mut pairs = Vec::new();
    for i in 1..allTokens.len() {
        pairs.push((makeContext(&allTokens, i), allTokens[i].clone()));
    }

    let mut work = LinearWork::new(P1);
    let mut sub = Subconscious::new(freq.clone(), Some(CAP));
    let mut cons = Consciousness::new(freq.clone());
    let mut focus = Focus::new();
    let mut stdout = io::stdout();
    let mut surpriseCount = 0u64;
    let mut willCount = 0u64;
    let mut top1Hits = 0u64;
    let mut top3Hits = 0u64;
    let mut scoredSteps = 0u64;
    let total = LIVESTEPS.min(pairs.len() as u64);

    println!("--- live stream ({} steps) ---\n", total);

    for step in 0..total {
        let idx = (step as usize) % pairs.len();
        let (ctx, target) = &pairs[idx];
        let props = sub.top(ctx, PROPOSALCOUNT);
        let isSurprise = !props.contains(target);
        scoredSteps += 1;
        if props.first().map_or(false, |t| t == target) { top1Hits += 1; }
        if props.contains(target) { top3Hits += 1; }
        sub.receiveObservation(ctx, target);

        let mut mark = "·";
        if isSurprise {
            surpriseCount += 1;
            mark = "!";
            let p = focus.probability();
            if work.unit() < p {
                cons.receiveTrainingTrigger(&mut work, ctx, &props, target);
                if let Some(chosen) = cons.chooseAction(&mut work, ctx, &props) {
                    sub.receiveConsciousAction(ctx, &chosen);
                    willCount += 1;
                    mark = "W";
                }
            }
        }
        focus.onStep(isSurprise);

        let col = colorForFocus(focus.level);
        print!("{}{}{}{} ", col, mark, target, resetColor());

        if (step + 1) % 40 == 0 || step + 1 == total {
            print!("\n  \x1b[90m[f={:.2} {} avg={:.1}% t1={:.1}% t3={:.1}%]\x1b[0m",
                focus.level,
                if focus.isAwake(){"Y"}else{"n"},
                focus.avgPercent(),
                hitRate(top1Hits, scoredSteps),
                hitRate(top3Hits, scoredSteps));
            println!();
            let _ = stdout.flush();
        }
        if STEPDELAYMS > 0 { thread::sleep(Duration::from_millis(STEPDELAYMS)); }
    }

    println!();
    println!("=== RESULT ===");
    println!("steps={} surprises={} willActions={}", total, surpriseCount, willCount);
    println!("top1={:.1}% top3={:.1}% (hits {}/{} and {}/{})",
        hitRate(top1Hits, scoredSteps), hitRate(top3Hits, scoredSteps),
        top1Hits, scoredSteps, top3Hits, scoredSteps);
    println!("focus final={:.3} fades={} avgConsciousness={:.1}%", focus.level, focus.fades, focus.avgPercent());
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites, sub.consciousWrites);
    println!("linearWork rate={:.6}", work.successes as f64 / work.calls.max(1) as f64);
}
