use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MASK: u64 = u64::MAX;
const P1: f64 = 0.501;
const RETRYLIMIT: usize = 5000;
const INCREMENT: f64 = 0.1;
const CAP: f64 = 24.0;
const CONTEXTSIZE: usize = 2;
const PROPOSALCOUNT: usize = 3;
const CYCLES: usize = 4;
const MAXTOKENSPERDOC: usize = 200;
const MINTOKENSPERDOC: usize = 20;
const LABELEDDOCS: usize = 12;
const UNLABELEDDOCS: usize = 12;
const TESTDOCS: usize = 8;
const HELPDIRS: &[&str] = &["manual", "command", "policy", "guide", "cpack_gen"];

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
    consciousWrites: u64,
    passiveWrites: u64,
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
            consciousWrites: 0,
            passiveWrites: 0,
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

fn evaluateSub(sub: &Subconscious, pairs: &[(Context, String)]) -> (f64, f64) {
    let n: usize = pairs.len().max(1);
    let mut top1: usize = 0;
    let mut top3: usize = 0;
    for (ctx, target) in pairs {
        let preds: Vec<String> = sub.top(ctx, 3);
        if preds.first().map(|p| p == target).unwrap_or(false) { top1 += 1; }
        if preds.iter().any(|p| p == target) { top3 += 1; }
    }
    (top1 as f64 / n as f64, top3 as f64 / n as f64)
}

fn evaluateConscious(sub: &Subconscious, cons: &Consciousness, pairs: &[(Context, String)]) -> (f64, f64) {
    let n: usize = pairs.len().max(1);
    let mut top1: usize = 0;
    let mut top3: usize = 0;
    for (ctx, target) in pairs {
        let proposals: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
        let ranked: Vec<String> = cons.rank(ctx, &proposals, 3);
        if ranked.first().map(|p| p == target).unwrap_or(false) { top1 += 1; }
        if ranked.iter().any(|p| p == target) { top3 += 1; }
    }
    (top1 as f64 / n as f64, top3 as f64 / n as f64)
}


fn evaluateNgram(train: &[Vec<String>], pairs: &[(Context, String)]) -> (f64, f64) {
    let mut prevToNext: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut uni: HashMap<String, u32> = HashMap::new();
    for tokens in train {
        for i in 0..tokens.len() {
            *uni.entry(tokens[i].clone()).or_insert(0) += 1;
            if i >= 1 {
                *prevToNext.entry(tokens[i-1].clone()).or_default().entry(tokens[i].clone()).or_insert(0) += 1;
            }
        }
    }
    let mut uniTop: Vec<(String, u32)> = uni.iter().map(|(k,v)| (k.clone(), *v)).collect();
    uniTop.sort_by(|a, b| b.1.cmp(&a.1));
    let uni3: Vec<String> = uniTop.iter().take(3).map(|(k,_)| k.clone()).collect();
    let n: usize = pairs.len().max(1);
    let mut top1: usize = 0;
    let mut top3: usize = 0;
    for (ctx, target) in pairs {
        let mut preds: Vec<String> = Vec::new();
        if let Some(m) = prevToNext.get(&ctx[1]) {
            let mut ranked: Vec<(String, u32)> = m.iter().map(|(t,c)| (t.clone(), *c)).collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (t, _) in ranked.into_iter().take(3) {
                preds.push(t);
            }
        }
        for t in &uni3 {
            if preds.len() >= 3 { break; }
            if !preds.contains(t) { preds.push(t.clone()); }
        }
        if preds.first().map(|p| p == target).unwrap_or(false) { top1 += 1; }
        if preds.iter().any(|p| p == target) { top3 += 1; }
    }
    (top1 as f64 / n as f64, top3 as f64 / n as f64)
}

fn main() {
    let docsMap: HashMap<String, Vec<String>> = loadDocs();
    println!("CORPUS=data sample rst; usable documents={}", docsMap.len());
    println!("split: labeled={} unlabeled={} test={}; P1={}; cycles={}", LABELEDDOCS, UNLABELEDDOCS, TESTDOCS, P1, CYCLES);

    let mut names: Vec<String> = docsMap.keys().cloned().collect();
    names.sort();
    names.reverse();

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

    // train pool for fair controls = labeled + unlabeled (never test)
    let mut trainDocs: Vec<Vec<String>> = labeled.clone();
    trainDocs.extend(unlabeled.clone());

    let mut freq: HashMap<String, u32> = HashMap::new();
    for doc in &trainDocs {
        for t in doc { *freq.entry(t.clone()).or_insert(0) += 1; }
    }

    let testPairs: Vec<(Context, String)> = contextPairs(&test);
    let labeledPairs: Vec<(Context, String)> = contextPairs(&labeled);
    let unlabeledPairs: Vec<(Context, String)> = contextPairs(&unlabeled);
    let trainPairs: Vec<(Context, String)> = contextPairs(&trainDocs);

    // unigram
    let mut common: Vec<(String, u32)> = freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
    common.sort_by(|a, b| b.1.cmp(&a.1));
    let top3uni: Vec<String> = common.iter().take(3).map(|(k, _)| k.clone()).collect();
    let uniTop1: f64 = testPairs.iter().filter(|(_, t)| top3uni.first().map(|u| u == t).unwrap_or(false)).count() as f64 / testPairs.len().max(1) as f64;
    let uniTop3: f64 = testPairs.iter().filter(|(_, t)| top3uni.iter().any(|u| u == t)).count() as f64 / testPairs.len().max(1) as f64;

    // n-gram baseline (train only)
    let (ngT1, ngT3): (f64, f64) = evaluateNgram(&trainDocs, &testPairs);

    // direct = perception on train only (fair)
    let mut direct: Subconscious = Subconscious::new(freq.clone(), Some(CAP));
    for _ in 0..CYCLES {
        for (ctx, target) in &trainPairs {
            direct.receiveObservation(ctx, target);
        }
    }
    let (dirT1, dirT3): (f64, f64) = evaluateSub(&direct, &testPairs);

    // perception only (same as direct now)
    let (percT1, percT3): (f64, f64) = (dirT1, dirT3);

    // full loop with surprise on train
    let mut work: LinearWork = LinearWork::new(P1);
    let mut sub: Subconscious = Subconscious::new(freq.clone(), Some(CAP));
    let mut cons: Consciousness = Consciousness::new(freq.clone());

    for _cycle in 0..CYCLES {
        for (ctx, target) in &trainPairs {
            let props: Vec<String> = sub.top(ctx, PROPOSALCOUNT);
            sub.receiveObservation(ctx, target);
            if !props.contains(target) {
                cons.receiveTrainingTrigger(&mut work, ctx, &props, target);
                if let Some(chosen) = cons.chooseAction(&mut work, ctx, &props) {
                    sub.receiveConsciousAction(ctx, &chosen);
                }
            }
        }
    }

    let (subT1, subT3): (f64, f64) = evaluateSub(&sub, &testPairs);
    let (consT1, consT3): (f64, f64) = evaluateConscious(&sub, &cons, &testPairs);

    println!();
    println!("labeledTokens={} unlabeledTokens={} testTokens={} vocabulary={}",
        labeled.iter().map(|d| d.len()).sum::<usize>(),
        unlabeled.iter().map(|d| d.len()).sum::<usize>(),
        testPairs.len(),
        freq.len());
    println!("NOTE: data highly repetitive (avg bigram jaccard ~0.22) — scores inflated");

    println!();
    println!("{:<52} {:>8} {:>8}", "model", "top1", "top3");
    println!("{}", "-".repeat(72));
    println!("{:<52} {:>7.2}% {:>7.2}%", "unigram (most frequent)", uniTop1 * 100.0, uniTop3 * 100.0);
    println!("{:<52} {:>7.2}% {:>7.2}%", "backoff n-gram (train only)", ngT1 * 100.0, ngT3 * 100.0);
    println!("{:<52} {:>7.2}% {:>7.2}%", "perception / direct (train only)", dirT1 * 100.0, dirT3 * 100.0);
    println!("{:<52} {:>7.2}% {:>7.2}%", "full loop perception+surprise", subT1 * 100.0, subT3 * 100.0);
    println!("{:<52} {:>7.2}% {:>7.2}%", "Consciousness after loop", consT1 * 100.0, consT3 * 100.0);

    println!();
    println!("DIAGNOSTICS");
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites, sub.consciousWrites);
    println!("linearWork calls={} successes={} rate={:.6} exhausted={}",
        work.calls, work.successes,
        work.successes as f64 / work.calls.max(1) as f64,
        work.exhausted);
}
