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
/// Cap for feature row.
const CAP: f64 = 24.0;
/// Context size.
const CONTEXTSIZE: usize = 2;
/// Proposals count.
const PROPOSALCOUNT: usize = 3;
/// Max tokens per doc.
const MAXTOKENSPERDOC: usize = 200;
/// Min tokens.
const MINTOKENSPERDOC: usize = 20;
/// Docs split.
const LABELEDDOCS: usize = 12;
const UNLABELEDDOCS: usize = 12;
const TESTDOCS: usize = 8;
const HELPDIRS: &[&str] = &["manual", "command", "policy", "guide", "cpack_gen"];

/// Focus decay per step without surprise.
const FOCUSDECAY: f64 = 0.04;
/// Focus gain on surprise / conscious action.
const FOCUSGAIN: f64 = 0.25;
/// Below this focus consciousness is off.
const FOCUSTHRESHOLD: f64 = 0.15;
/// Max focus.
const FOCUSMAX: f64 = 1.0;

fn timeToUnit(ns: u64) -> f64 {
    let mut x: u64 = ns.wrapping_add(0x9E3779B97F4A7C15) & MASK;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9) & MASK;
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB) & MASK;
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / 9007199254740992.0)
}

fn nowNs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

struct LinearWork {
    p1: f64,
    lastNs: u64,
    calls: u64,
    successes: u64,
    exhausted: u64,
}

impl LinearWork {
    fn new(p1: f64) -> Self {
        Self { p1, lastNs: 0, calls: 0, successes: 0, exhausted: 0 }
    }
    fn waitForSuccess(&mut self) -> bool {
        let p1: f64 = self.p1;
        let low: f64 = 1.0 - p1;
        let mut lastNs: u64 = self.lastNs;
        for attempt in 1..=RETRYLIMIT {
            let mut ns: u64 = nowNs();
            if ns <= lastNs { ns = lastNs + 1; }
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
    fn unit(&mut self) -> f64 {
        let mut ns: u64 = nowNs();
        if ns <= self.lastNs { ns = self.lastNs + 1; }
        self.lastNs = ns;
        timeToUnit(ns)
    }
}

type Context = [String; 2];

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

struct FeatureRow {
    weights: HashMap<String, f64>,
    total: f64,
    cap: Option<f64>,
}

impl FeatureRow {
    fn new(cap: Option<f64>) -> Self {
        Self { weights: HashMap::new(), total: 0.0, cap }
    }
    fn bump(&mut self, token: &str, amount: f64) {
        *self.weights.entry(token.to_string()).or_insert(0.0) += amount;
        self.total += amount;
        if let Some(cap) = self.cap {
            if self.total > cap * 1.5 {
                let scale: f64 = cap / self.total;
                for w in self.weights.values_mut() { *w *= scale; }
                let thr: f64 = cap * 0.0008;
                self.weights.retain(|_, w| *w >= thr);
                self.total = self.weights.values().sum();
            }
        }
    }
}

struct Subconscious {
    globalFreq: HashMap<String, u32>,
    commonTokens: Vec<String>,
    rows: HashMap<String, FeatureRow>,
    cap: Option<f64>,
    passiveWrites: u64,
    consciousWrites: u64,
}

impl Subconscious {
    fn new(globalFreq: HashMap<String, u32>, cap: Option<f64>) -> Self {
        let mut common: Vec<(String, u32)> = globalFreq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self {
            globalFreq,
            commonTokens: common.into_iter().map(|(k, _)| k).collect(),
            rows: HashMap::new(),
            cap,
            passiveWrites: 0,
            consciousWrites: 0,
        }
    }
    fn row(&mut self, feature: &str) -> &mut FeatureRow {
        self.rows.entry(feature.to_string()).or_insert_with(|| FeatureRow::new(self.cap))
    }
    fn write(&mut self, ctx: &Context, token: &str, passive: bool) {
        for feature in contextFeatures(ctx) {
            self.row(&feature).bump(token, INCREMENT);
        }
        if passive { self.passiveWrites += 1; } else { self.consciousWrites += 1; }
    }
    fn receiveObservation(&mut self, ctx: &Context, token: &str) {
        self.write(ctx, token, true);
    }
    fn receiveConsciousAction(&mut self, ctx: &Context, token: &str) {
        self.write(ctx, token, false);
    }
    fn top(&self, ctx: &Context, k: usize) -> Vec<String> {
        let features: Vec<String> = contextFeatures(ctx);
        let mut cand: HashSet<String> = HashSet::new();
        for feature in &features {
            if let Some(row) = self.rows.get(feature) {
                let mut ranked: Vec<(&String, &f64)> = row.weights.iter().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then(a.0.cmp(b.0)));
                for (t, _) in ranked.into_iter().take(8) {
                    cand.insert(t.clone());
                }
            }
        }
        let mut scores: HashMap<String, f64> = HashMap::new();
        for t in &cand {
            let mut s: f64 = 0.0;
            for feature in &features {
                if let Some(row) = self.rows.get(feature) {
                    if let Some(w) = row.weights.get(t) {
                        if row.total > 0.0 { s += w / row.total; }
                    }
                }
            }
            scores.insert(t.clone(), s);
        }
        let mut ranked: Vec<String> = scores.keys().cloned().collect();
        ranked.sort_by(|a, b| {
            scores[b].partial_cmp(&scores[a]).unwrap()
                .then(self.globalFreq.get(b).unwrap_or(&0).cmp(self.globalFreq.get(a).unwrap_or(&0)))
                .then(a.cmp(b))
        });
        if ranked.len() < k {
            for t in &self.commonTokens {
                if !scores.contains_key(t) { ranked.push(t.clone()); }
                if ranked.len() >= k { break; }
            }
        }
        ranked.truncate(k);
        ranked
    }
}

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
    fn activeFeatures(&self, ctx: &Context, proposals: &[String]) -> Vec<String> {
        let mut f: Vec<String> = contextFeatures(ctx);
        for t in proposals { f.push(format!("proposal:{}", t)); }
        f
    }
    fn receiveTrainingTrigger(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String], target: &str) -> bool {
        if !work.waitForSuccess() { self.failedUpdates += 1; return false; }
        for feature in self.activeFeatures(ctx, proposals) {
            *self.edges.entry(feature.clone()).or_default().entry(target.to_string()).or_insert(0.0) += INCREMENT;
            *self.totals.entry(feature).or_insert(0.0) += INCREMENT;
        }
        self.updateCount += 1;
        true
    }
    fn candidatePool(&self, ctx: &Context, proposals: &[String], kPer: usize) -> Vec<String> {
        let mut pool: Vec<String> = proposals.to_vec();
        let mut seen: HashSet<String> = pool.iter().cloned().collect();
        for feature in self.activeFeatures(ctx, proposals) {
            if let Some(out) = self.edges.get(&feature) {
                let mut ranked: Vec<(String, f64)> = out.iter().map(|(t, w)| (t.clone(), *w)).collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()
                    .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0)))
                    .then(a.0.cmp(&b.0)));
                for (t, _) in ranked.into_iter().take(kPer) {
                    if !seen.contains(&t) { seen.insert(t.clone()); pool.push(t); }
                }
            }
        }
        if pool.is_empty() {
            pool = self.commonTokens.iter().take(PROPOSALCOUNT).cloned().collect();
        }
        pool
    }
    fn scores(&self, ctx: &Context, proposals: &[String], candidates: &[String]) -> HashMap<String, f64> {
        let mut scores: HashMap<String, f64> = candidates.iter().map(|t| (t.clone(), 0.0)).collect();
        for feature in self.activeFeatures(ctx, proposals) {
            let total: f64 = *self.totals.get(&feature).unwrap_or(&0.0);
            if total <= 0.0 { continue; }
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
    fn rank(&self, ctx: &Context, proposals: &[String], k: usize) -> Vec<String> {
        let cands: Vec<String> = self.candidatePool(ctx, proposals, 3);
        let scores: HashMap<String, f64> = self.scores(ctx, proposals, &cands);
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()
            .then(self.globalFreq.get(&b.0).unwrap_or(&0).cmp(self.globalFreq.get(&a.0).unwrap_or(&0)))
            .then(a.0.cmp(&b.0)));
        ranked.into_iter().map(|(t, _)| t).take(k).collect()
    }
    fn chooseAction(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String]) -> Option<String> {
        if !work.waitForSuccess() { self.failedUpdates += 1; return None; }
        let candidates: Vec<String> = self.candidatePool(ctx, proposals, 3);
        let scores: HashMap<String, f64> = self.scores(ctx, proposals, &candidates);
        let mut ranked: Vec<String> = candidates;
        ranked.sort_by(|a, b| scores.get(b).unwrap_or(&0.0).partial_cmp(scores.get(a).unwrap_or(&0.0)).unwrap()
            .then(self.globalFreq.get(b).unwrap_or(&0).cmp(self.globalFreq.get(a).unwrap_or(&0)))
            .then(a.cmp(b)));
        let chosen: String = if ranked.is_empty() {
            proposals.first().cloned().unwrap_or_default()
        } else if work.unit() < 0.10 && ranked.len() > 1 {
            ranked[1].clone()
        } else {
            ranked[0].clone()
        };
        self.actions += 1;
        Some(chosen)
    }
}

/// Focus state for consciousness.
struct Focus {
    level: f64,
    consciousSteps: u64,
    subconsciousOnlySteps: u64,
    lastAwakeStep: u64,
    fades: u64,
}

impl Focus {
    fn new() -> Self {
        Self {
            level: 0.6, // start awake
            consciousSteps: 0,
            subconsciousOnlySteps: 0,
            lastAwakeStep: 0,
            fades: 0,
        }
    }
    fn isAwake(&self) -> bool {
        self.level >= FOCUSTHRESHOLD
    }
    fn onSurprise(&mut self, step: u64) {
        self.level = (self.level + FOCUSGAIN).min(FOCUSMAX);
        self.lastAwakeStep = step;
    }
    fn onStep(&mut self, step: u64, hadSurprise: bool) {
        if hadSurprise {
            self.onSurprise(step);
            self.consciousSteps += 1;
        } else {
            self.level = (self.level - FOCUSDECAY).max(0.0);
            if self.isAwake() {
                self.consciousSteps += 1;
            } else {
                self.subconsciousOnlySteps += 1;
            }
        }
        // detect fade
        if self.level < FOCUSTHRESHOLD && self.level + FOCUSDECAY >= FOCUSTHRESHOLD {
            self.fades += 1;
        }
    }
}

fn tokenize(path: &Path, limit: usize) -> Vec<String> {
    let raw: String = fs::read_to_string(path).unwrap_or_default().to_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in raw.chars() {
        if c.is_ascii_alphabetic() || c == '\'' || c == '-' {
            cur.push(c);
        } else if !cur.is_empty() {
            tokens.push(cur.clone());
            cur.clear();
            if tokens.len() >= limit { break; }
        }
    }
    if !cur.is_empty() && tokens.len() < limit {
        tokens.push(cur);
    }
    tokens
}

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

fn loadDocs() -> HashMap<String, Vec<String>> {
    let mut docs: HashMap<String, Vec<String>> = HashMap::new();
    let base: &Path = Path::new("data");
    for dir in HELPDIRS {
        let p: PathBuf = base.join(dir);
        if !p.exists() { continue; }
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

fn contextPairs(docs: &[Vec<String>]) -> Vec<(Context, String)> {
    let mut pairs: Vec<(Context, String)> = Vec::new();
    for tokens in docs {
        for i in 1..tokens.len() {
            pairs.push((makeContext(tokens, i), tokens[i].clone()));
        }
    }
    pairs
}

fn main() {
    let docsMap: HashMap<String, Vec<String>> = loadDocs();
    println!("CORPUS=data; documents={}", docsMap.len());
    println!("Focus: decay={FOCUSDECAY} gain={FOCUSGAIN} threshold={FOCUSTHRESHOLD}");

    let mut names: Vec<String> = docsMap.keys().cloned().collect();
    names.sort();
    names.reverse();

    let n: usize = names.len();
    if n == 0 {
        println!("no documents");
        return;
    }
    let labeledEnd: usize = LABELEDDOCS.min(n);
    let unlabeledEnd: usize = (LABELEDDOCS + UNLABELEDDOCS).min(n);
    let testEnd: usize = (LABELEDDOCS + UNLABELEDDOCS + TESTDOCS).min(n);
    let labeled: Vec<Vec<String>> = names[..labeledEnd].iter().map(|n| docsMap[n].clone()).collect();
    let unlabeled: Vec<Vec<String>> = names[labeledEnd..unlabeledEnd].iter().map(|n| docsMap[n].clone()).collect();
    let test: Vec<Vec<String>> = names[unlabeledEnd..testEnd].iter().map(|n| docsMap[n].clone()).collect();

    let mut trainDocs: Vec<Vec<String>> = labeled.clone();
    trainDocs.extend(unlabeled.clone());

    let mut freq: HashMap<String, u32> = HashMap::new();
    for doc in &trainDocs {
        for t in doc { *freq.entry(t.clone()).or_insert(0) += 1; }
    }

    let trainPairs: Vec<(Context, String)> = contextPairs(&trainDocs);
    let testPairs: Vec<(Context, String)> = contextPairs(&test);

    let mut work: LinearWork = LinearWork::new(P1);
    let mut sub: Subconscious = Subconscious::new(freq.clone(), Some(CAP));
    let mut cons: Consciousness = Consciousness::new(freq.clone());
    let mut focus: Focus = Focus::new();

    println!();
    println!("=== FOCUS CYCLE (train stream) ===");
    println!("{:<6} {:<8} {:<8} {:<10} {}", "step", "focus", "awake", "event", "token");

    let reportEvery: usize = (trainPairs.len() / 20).max(1);
    let mut step: u64 = 0;
    let mut surpriseCount: u64 = 0;
    let mut consciousActions: u64 = 0;

    for (ctx, target) in &trainPairs {
        step += 1;
        let props: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
        let isSurprise: bool = !props.contains(target);

        // always perceive
        sub.receiveObservation(ctx, target);

        let mut event: &str = "passive";
        if isSurprise {
            surpriseCount += 1;
            event = "surprise";
            if focus.isAwake() {
                // consciousness can react only while focused
                cons.receiveTrainingTrigger(&mut work, ctx, &props, target);
                if let Some(chosen) = cons.chooseAction(&mut work, ctx, &props) {
                    sub.receiveConsciousAction(ctx, &chosen);
                    consciousActions += 1;
                    event = "surprise+will";
                }
            } else {
                event = "surprise(asleep)";
            }
        }

        focus.onStep(step, isSurprise);

        if step == 1 || step as usize % reportEvery == 0 || !focus.isAwake() && focus.level + FOCUSDECAY >= FOCUSTHRESHOLD {
            let awakeStr = if focus.isAwake() { "YES" } else { "no" };
            println!("{:<6} {:<8.3} {:<8} {:<10} {}", step, focus.level, awakeStr, event, target);
        }
    }

    // final metrics
    let mut top1 = 0usize;
    let mut top3 = 0usize;
    for (ctx, target) in &testPairs {
        let preds = sub.top(ctx, 3);
        if preds.first().map(|p| p == target).unwrap_or(false) { top1 += 1; }
        if preds.iter().any(|p| p == target) { top3 += 1; }
    }
    let nTest = testPairs.len().max(1);

    println!();
    println!("=== RESULT ===");
    println!("steps={} surprises={} consciousActions={}", step, surpriseCount, consciousActions);
    println!("focus final={:.3} fades={} consciousSteps={} subOnlySteps={}",
        focus.level, focus.fades, focus.consciousSteps, focus.subconsciousOnlySteps);
    println!("test top1={:.2}% top3={:.2}%", top1 as f64 / nTest as f64 * 100.0, top3 as f64 / nTest as f64 * 100.0);
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites, sub.consciousWrites);
    println!("linearWork rate={:.6}", work.successes as f64 / work.calls.max(1) as f64);
}
