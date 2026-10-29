use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::binio::{readF64, readStr, readU32, readU64, writeF64, writeStr, writeU32, writeU64};

pub const MASK: u64 = u64::MAX;
pub const P1: f64 = 0.51; // Скорость симуляции, важно чтобы был 0.501 или как можно меньше, но для тестов можно и 0.51 для скорости, точность не падает сильно. но при большей скорости - нужно больше тестов.
pub const RETRYLIMIT: usize = 3000;
pub const INCREMENT: f64 = 0.1;
pub const CAP: f64 = 24.0;
pub const PROPOSALCOUNT: usize = 3;
/// How many entries of a feature row are taken into the candidate set.
pub const ROWTAKE: usize = 8;
/// How many continuations of a feature are offered to the conscious choice.
pub const EDGETAKE: usize = 3;
pub const BOS: &str = "<BOS>";
/// Window size used by the dialogue phase.
pub const DLGWINDOW: usize = 3;

pub const FOCUSDECAY: f64 = 0.035;
pub const FOCUSGAIN: f64 = 0.28;
pub const FOCUSTHRESHOLD: f64 = 0.15;
pub const FOCUSMAX: f64 = 1.0;

pub fn timeToUnit(ns: u64) -> f64 {
    let mut x = ns.wrapping_add(0x9E3779B97F4A7C15) & MASK;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9) & MASK;
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB) & MASK;
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / 9007199254740992.0)
}
pub fn nowNs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

pub struct LinearWork { p1: f64, lastNs: u64, calls: u64, successes: u64, exhausted: u64 }
impl LinearWork {
    pub fn new(p1: f64) -> Self { Self { p1, lastNs: 0, calls: 0, successes: 0, exhausted: 0 } }
    pub fn waitForSuccess(&mut self) -> bool {
        let p1 = self.p1; let low = 1.0 - p1; let mut lastNs = self.lastNs;
        for attempt in 1..=RETRYLIMIT {
            let mut ns = nowNs(); if ns <= lastNs { ns = lastNs + 1; } lastNs = ns;
            let u = timeToUnit(ns);
            if u < p1 && u > low { self.lastNs = lastNs; self.calls += attempt as u64; self.successes += 1; return true; }
        }
        self.lastNs = lastNs; self.calls += RETRYLIMIT as u64; self.exhausted += 1; false
    }
    pub fn unit(&mut self) -> f64 {
        let mut ns = nowNs(); if ns <= self.lastNs { ns = self.lastNs + 1; } self.lastNs = ns; timeToUnit(ns)
    }
    pub fn successes(&self) -> u64 { self.successes }
    pub fn calls(&self) -> u64 { self.calls }
    pub fn exhausted(&self) -> u64 { self.exhausted }
}

/// Context window. `win` holds the previous tokens ordered oldest -> newest,
/// `bag` holds the discriminative words of the prompt.
#[derive(Clone)]
pub struct Context {
    pub win: Vec<String>,
    pub bag: Vec<String>,
}

impl Context {
    pub fn withBag(win: Vec<String>, bag: Vec<String>) -> Context { Context { win, bag } }
}

/// Last `size` tokens of `history`, front-padded with <BOS>.
pub fn makeWindow(history: &[String], size: usize) -> Vec<String> {
    let mut win = vec![BOS.to_string(); size];
    let take = size.min(history.len());
    let offset = size - take;
    for i in 0..take { win[offset + i] = history[history.len() - take + i].clone(); }
    win
}

/// Feature extraction: the window of previous tokens plus the discriminative
/// words of the prompt. With a two token window this yields exactly the
/// features of the original engine (prev / pair / prev2); a longer window
/// adds prev3..prev5.
pub fn contextFeatures(ctx: &Context) -> Vec<String> {
    let n = ctx.win.len();
    let mut f = Vec::with_capacity(n + ctx.bag.len());
    if n >= 1 {
        f.push(format!("prev:{}", ctx.win[n - 1]));
        if n >= 2 {
            f.push(format!("pair:{}|{}", ctx.win[n - 2], ctx.win[n - 1]));
            if ctx.win[n - 2] != BOS { f.push(format!("prev2:{}", ctx.win[n - 2])); }
        }
        for d in 3..=5 {
            if n >= d && ctx.win[n - d] != BOS { f.push(format!("prev{}:{}", d, ctx.win[n - d])); }
        }
    }
    for w in &ctx.bag { f.push(format!("bag:{}", w)); }
    f
}

pub struct FeatureRow { weights: HashMap<String, f64>, total: f64, cap: Option<f64>, top: Vec<(String, f64)>, hits: u64 }

/// Order of a feature row: heavier first, then the token name.
fn rowBetter(a: &(String, f64), b: &(String, f64)) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0))
}

impl FeatureRow {
    fn new(cap: Option<f64>) -> Self { Self { weights: HashMap::new(), total: 0.0, cap, top: Vec::new(), hits: 0 } }
    fn bump(&mut self, token: &str, amount: f64) {
        self.hits += 1;
        let weight = {
            let w = self.weights.entry(token.to_string()).or_insert(0.0);
            *w += amount;
            *w
        };
        self.total += amount;
        self.updateTop(token, weight);
        if let Some(cap) = self.cap {
            if self.total > cap * 1.5 {
                let scale = cap / self.total;
                for w in self.weights.values_mut() { *w *= scale; }
                let thr = cap * 0.0008;
                self.weights.retain(|_, w| *w >= thr);
                self.total = self.weights.values().sum();
                self.rebuildTop();
            }
        }
    }
    /// Keeps the best ROWTAKE entries of the row up to date, so that the
    /// candidate set never needs a full sort of the row.
    fn updateTop(&mut self, token: &str, weight: f64) {
        if let Some(slot) = self.top.iter_mut().find(|(t, _)| t == token) {
            slot.1 = weight;
        } else if self.top.len() < ROWTAKE {
            self.top.push((token.to_string(), weight));
        } else {
            let weakest = self.top[self.top.len() - 1].clone();
            let candidate = (token.to_string(), weight);
            if rowBetter(&candidate, &weakest) != Ordering::Less { return; }
            self.top.pop();
            self.top.push(candidate);
        }
        self.top.sort_by(rowBetter);
    }
    fn rebuildTop(&mut self) {
        self.top.clear();
        let entries: Vec<(String, f64)> = self.weights.iter().map(|(t, w)| (t.clone(), *w)).collect();
        for (t, w) in entries { self.updateTop(&t, w); }
    }
    /// The best entries of the row, exactly as a full sort would give them.
    #[allow(dead_code)]
    fn bestEntries(&self) -> Vec<(String, f64)> {
        let mut all: Vec<(String, f64)> = self.weights.iter().map(|(t, w)| (t.clone(), *w)).collect();
        all.sort_by(rowBetter);
        all.truncate(ROWTAKE);
        all
    }
}

pub struct Subconscious {
    globalFreq: HashMap<String, u32>, commonTokens: Vec<String>,
    rows: HashMap<String, FeatureRow>, cap: Option<f64>, passiveWrites: u64, consciousWrites: u64,
    maxRows: usize, evicted: u64,
}
impl Subconscious {
    pub fn new(globalFreq: HashMap<String, u32>, cap: Option<f64>) -> Self {
        let mut common: Vec<_> = globalFreq.iter().map(|(k,v)| (k.clone(),*v)).collect();
        common.sort_by(|a,b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self { globalFreq, commonTokens: common.into_iter().map(|(k,_)|k).collect(), rows: HashMap::new(), cap, passiveWrites: 0, consciousWrites: 0, maxRows: 0, evicted: 0 }
    }
    /// Upper bound for the number of feature rows. When it is reached the
    /// least used rows are dropped, so the memory of the mind stays bounded
    /// on any corpus. 0 = no limit.
    pub fn setMaxRows(&mut self, maxRows: usize) { self.maxRows = maxRows; }
    pub fn evicted(&self) -> u64 { self.evicted }
    fn evictIfNeeded(&mut self) {
        if self.maxRows == 0 || self.rows.len() <= self.maxRows { return; }
        let target = self.maxRows * 9 / 10;
        let mut victims: Vec<String> = Vec::new();
        {
            let mut all: Vec<(u64, &String)> = self.rows.iter().map(|(k, r)| (r.hits, k)).collect();
            let dropCount = all.len().saturating_sub(target);
            if dropCount == 0 { return; }
            all.select_nth_unstable_by(dropCount, |a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
            for i in 0..dropCount { victims.push(all[i].1.clone()); }
        }
        let dropped = victims.len() as u64;
        for key in victims { self.rows.remove(&key); }
        self.evicted += dropped;
    }
    fn row(&mut self, feature: &str) -> &mut FeatureRow {
        self.rows.entry(feature.to_string()).or_insert_with(|| FeatureRow::new(self.cap))
    }
    pub fn write(&mut self, ctx: &Context, token: &str, passive: bool) {
        for feature in contextFeatures(ctx) { self.row(&feature).bump(token, INCREMENT); }
        if passive { self.passiveWrites += 1; } else { self.consciousWrites += 1; }
        self.evictIfNeeded();
    }
    pub fn receiveObservation(&mut self, ctx: &Context, token: &str) { self.write(ctx, token, true); }
    pub fn receiveConsciousAction(&mut self, ctx: &Context, token: &str) { self.write(ctx, token, false); }
    fn scoreToken(&self, features: &[String], token: &str) -> f64 {
        let mut s = 0.0;
        for feature in features {
            if let Some(row) = self.rows.get(feature) {
                if let Some(w) = row.weights.get(token) { if row.total > 0.0 { s += w / row.total; } }
            }
        }
        s
    }
    pub fn top(&self, ctx: &Context, k: usize) -> Vec<String> {
        let (ranked, _) = self.topWithCandidates(ctx, k);
        ranked
    }

    /// The proposals and the full candidate set behind them: a token outside
    /// of that set can never be hit, no matter how good the ranking is.
    pub fn topWithCandidates(&self, ctx: &Context, k: usize) -> (Vec<String>, HashSet<String>) {
        let features = contextFeatures(ctx);
        let mut cand: HashSet<String> = HashSet::new();
        for feature in &features {
            if let Some(row) = self.rows.get(feature) {
                for (t, _) in &row.top { cand.insert(t.clone()); }
            }
        }
        let mut scores: HashMap<String, f64> = HashMap::new();
        for t in &cand { scores.insert(t.clone(), self.scoreToken(&features, t)); }
        let mut ranked: Vec<String> = scores.keys().cloned().collect();
        ranked.sort_by(|a,b| scores[b].partial_cmp(&scores[a]).unwrap()
            .then(self.globalFreq.get(b).unwrap_or(&0).cmp(self.globalFreq.get(a).unwrap_or(&0))).then(a.cmp(b)));
        if ranked.len() < k {
            for t in &self.commonTokens {
                if !scores.contains_key(t) { ranked.push(t.clone()); }
                if ranked.len() >= k { break; }
            }
        }
        ranked.truncate(k);
        (ranked, cand)
    }
    /// Score of an arbitrary candidate set under the same rules as `top`.
    pub fn scoreCandidates(&self, ctx: &Context, cands: &[String]) -> Vec<(String, f64)> {
        let features = contextFeatures(ctx);
        cands.iter().map(|t| (t.clone(), self.scoreToken(&features, t))).collect()
    }
    /// Additive vocabulary growth. A later phase may teach new words without
    /// touching the statistics collected so far: only the global frequency
    /// table and the common token fallback are extended.
    pub fn observeVocabulary(&mut self, freq: &HashMap<String, u32>) {
        for (token, count) in freq {
            *self.globalFreq.entry(token.clone()).or_insert(0) += count;
        }
        self.rebuildCommon();
    }
    /// Diagnostics: (rows, entries, biggest row).
    pub fn rowStats(&self) -> (usize, usize, usize) {
        let mut entries = 0usize;
        let mut biggest = 0usize;
        for row in self.rows.values() {
            let n = row.weights.len();
            entries += n;
            if n > biggest { biggest = n; }
        }
        (self.rows.len(), entries, biggest)
    }
    /// Writes the whole table to a stream (checkpoint format).
    pub fn save(&self, out: &mut dyn Write) -> std::io::Result<()> {
        writeU64(out, self.globalFreq.len() as u64)?;
        for (token, count) in &self.globalFreq {
            writeStr(out, token)?;
            writeU64(out, *count as u64)?;
        }
        writeU64(out, self.rows.len() as u64)?;
        for (feature, row) in &self.rows {
            writeStr(out, feature)?;
            writeF64(out, row.total)?;
            writeU64(out, row.hits)?;
            writeU32(out, row.weights.len() as u32)?;
            for (token, weight) in &row.weights {
                writeStr(out, token)?;
                writeF64(out, *weight)?;
            }
        }
        Ok(())
    }
    /// Reads the table back; the derived top-k cache is rebuilt.
    pub fn load(&mut self, input: &mut dyn Read) -> std::io::Result<()> {
        self.globalFreq.clear();
        let tokens = readU64(input)?;
        for _ in 0..tokens {
            let token = readStr(input)?;
            let count = readU64(input)? as u32;
            self.globalFreq.insert(token, count);
        }
        self.rows.clear();
        let rows = readU64(input)?;
        for _ in 0..rows {
            let feature = readStr(input)?;
            let total = readF64(input)?;
            let hits = readU64(input)?;
            let entries = readU32(input)?;
            let mut row = FeatureRow::new(self.cap);
            row.total = total;
            row.hits = hits;
            for _ in 0..entries {
                let token = readStr(input)?;
                let weight = readF64(input)?;
                row.weights.insert(token, weight);
            }
            row.rebuildTop();
            self.rows.insert(feature, row);
        }
        self.rebuildCommon();
        Ok(())
    }
    fn rebuildCommon(&mut self) {
        let mut common: Vec<_> = self.globalFreq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.commonTokens = common.into_iter().map(|(k, _)| k).collect();
    }
    pub fn passiveWrites(&self) -> u64 { self.passiveWrites }
    pub fn consciousWrites(&self) -> u64 { self.consciousWrites }
}

pub struct Consciousness {
    globalFreq: HashMap<String, u32>, commonTokens: Vec<String>,
    edges: HashMap<String, HashMap<String, f64>>, totals: HashMap<String, f64>,
    edgeTop: HashMap<String, Vec<(String, f64)>>,
    maxFeatures: usize, evicted: u64,
    actions: u64, failedUpdates: u64,
}

/// Order of the conscious edges: heavier first, then the global frequency,
/// then the token name.
fn edgeBetter(a: &(String, f64), b: &(String, f64), freq: &HashMap<String, u32>) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap()
        .then(freq.get(&a.0).unwrap_or(&0).cmp(freq.get(&b.0).unwrap_or(&0)))
        .then(a.0.cmp(&b.0))
}

impl Consciousness {
    pub fn new(globalFreq: HashMap<String, u32>) -> Self {
        let mut common: Vec<_> = globalFreq.iter().map(|(k,v)| (k.clone(),*v)).collect();
        common.sort_by(|a,b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Self { globalFreq, commonTokens: common.into_iter().map(|(k,_)|k).collect(), edges: HashMap::new(), totals: HashMap::new(), edgeTop: HashMap::new(), maxFeatures: 0, evicted: 0, actions: 0, failedUpdates: 0 }
    }
    /// Upper bound for the number of features that keep conscious edges.
    /// 0 = no limit.
    pub fn setMaxFeatures(&mut self, maxFeatures: usize) { self.maxFeatures = maxFeatures; }
    pub fn evicted(&self) -> u64 { self.evicted }
    /// Drops the least used features so that the memory stays bounded.
    fn evictIfNeeded(&mut self) {
        if self.maxFeatures == 0 || self.edges.len() <= self.maxFeatures { return; }
        let target = self.maxFeatures * 9 / 10;
        let mut victims: Vec<String> = Vec::new();
        {
            // the total of a feature is proportional to the number of times
            // it was updated, so it is the usage counter of the feature
            let mut all: Vec<(u64, &String)> = self.edges.keys()
                .map(|k| (*self.totals.get(k).unwrap_or(&0.0) as u64, k)).collect();
            let dropCount = all.len().saturating_sub(target);
            if dropCount == 0 { return; }
            all.select_nth_unstable_by(dropCount, |a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));
            for i in 0..dropCount { victims.push(all[i].1.clone()); }
        }
        for key in victims {
            self.edges.remove(&key);
            self.totals.remove(&key);
            self.edgeTop.remove(&key);
            self.evicted += 1;
        }
    }
    fn activeFeatures(&self, ctx: &Context, proposals: &[String]) -> Vec<String> {
        let mut f = contextFeatures(ctx);
        for t in proposals { f.push(format!("proposal:{}", t)); }
        f
    }
    pub fn receiveTrainingTrigger(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String], target: &str) -> bool {
        if !work.waitForSuccess() { self.failedUpdates += 1; return false; }
        for feature in self.activeFeatures(ctx, proposals) {
            let weight = {
                let out = self.edges.entry(feature.clone()).or_default();
                let w = out.entry(target.to_string()).or_insert(0.0);
                *w += INCREMENT;
                *w
            };
            *self.totals.entry(feature.clone()).or_insert(0.0) += INCREMENT;
            self.updateEdgeTop(&feature, target, weight);
        }
        self.evictIfNeeded();
        true
    }
    /// Keeps the best EDGETAKE continuations of a feature up to date.
    fn updateEdgeTop(&mut self, feature: &str, token: &str, weight: f64) {
        let freq = &self.globalFreq;
        let top = self.edgeTop.entry(feature.to_string()).or_default();
        if let Some(slot) = top.iter_mut().find(|(t, _)| t == token) {
            slot.1 = weight;
        } else if top.len() < EDGETAKE {
            top.push((token.to_string(), weight));
        } else {
            let weakest = top[top.len() - 1].clone();
            let candidate = (token.to_string(), weight);
            if edgeBetter(&candidate, &weakest, freq) != Ordering::Less { return; }
            top.pop();
            top.push(candidate);
        }
        top.sort_by(|a, b| edgeBetter(a, b, freq));
    }
    fn candidatePool(&self, ctx: &Context, proposals: &[String], kPer: usize) -> Vec<String> {
        let mut pool = proposals.to_vec();
        let mut seen: HashSet<String> = pool.iter().cloned().collect();
        for feature in self.activeFeatures(ctx, proposals) {
            if let Some(top) = self.edgeTop.get(&feature) {
                for (t, _) in top.iter().take(kPer) {
                    if !seen.contains(t) { seen.insert(t.clone()); pool.push(t.clone()); }
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
    pub fn chooseAction(&mut self, work: &mut LinearWork, ctx: &Context, proposals: &[String]) -> Option<String> {
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
    /// Conscious score of an arbitrary candidate set (edges + proposal prior).
    pub fn scoreCandidates(&self, ctx: &Context, proposals: &[String], cands: &[String]) -> Vec<(String, f64)> {
        let scores = self.scores(ctx, proposals, cands);
        cands.iter().map(|t| (t.clone(), *scores.get(t).unwrap_or(&0.0))).collect()
    }
    /// Additive vocabulary growth, same semantics as in `Subconscious`.
    pub fn observeVocabulary(&mut self, freq: &HashMap<String, u32>) {
        for (token, count) in freq {
            *self.globalFreq.entry(token.clone()).or_insert(0) += count;
        }
        self.rebuildCommon();
        // The order of the edges depends on the global frequency, so the
        // cached best continuations have to be rebuilt.
        self.edgeTop.clear();
    }
    /// Diagnostics: (features, edges, biggest feature).
    pub fn edgeStats(&self) -> (usize, usize, usize) {
        let mut edges = 0usize;
        let mut biggest = 0usize;
        for out in self.edges.values() {
            edges += out.len();
            if out.len() > biggest { biggest = out.len(); }
        }
        (self.edges.len(), edges, biggest)
    }
    /// Writes the conscious edges to a stream (checkpoint format).
    pub fn save(&self, out: &mut dyn Write) -> std::io::Result<()> {
        writeU64(out, self.globalFreq.len() as u64)?;
        for (token, count) in &self.globalFreq {
            writeStr(out, token)?;
            writeU64(out, *count as u64)?;
        }
        writeU64(out, self.edges.len() as u64)?;
        for (feature, targets) in &self.edges {
            writeStr(out, feature)?;
            writeF64(out, *self.totals.get(feature).unwrap_or(&0.0))?;
            writeU32(out, targets.len() as u32)?;
            for (target, weight) in targets {
                writeStr(out, target)?;
                writeF64(out, *weight)?;
            }
        }
        Ok(())
    }
    /// Reads the conscious edges back; the cached best continuations are rebuilt.
    pub fn load(&mut self, input: &mut dyn Read) -> std::io::Result<()> {
        self.globalFreq.clear();
        let tokens = readU64(input)?;
        for _ in 0..tokens {
            let token = readStr(input)?;
            let count = readU64(input)? as u32;
            self.globalFreq.insert(token, count);
        }
        self.edges.clear();
        self.totals.clear();
        self.edgeTop.clear();
        let features = readU64(input)?;
        for _ in 0..features {
            let feature = readStr(input)?;
            let total = readF64(input)?;
            let entries = readU32(input)?;
            let mut targets: HashMap<String, f64> = HashMap::with_capacity(entries as usize);
            for _ in 0..entries {
                let target = readStr(input)?;
                let weight = readF64(input)?;
                targets.insert(target, weight);
            }
            self.totals.insert(feature.clone(), total);
            for (target, weight) in &targets {
                self.updateEdgeTop(&feature, target, *weight);
            }
            self.edges.insert(feature, targets);
        }
        self.rebuildCommon();
        Ok(())
    }
    fn rebuildCommon(&mut self) {
        let mut common: Vec<_> = self.globalFreq.iter().map(|(k, v)| (k.clone(), *v)).collect();
        common.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.commonTokens = common.into_iter().map(|(k, _)| k).collect();
    }
    pub fn actions(&self) -> u64 { self.actions }
    pub fn failedUpdates(&self) -> u64 { self.failedUpdates }
}

pub struct Focus {
    level: f64, consciousSteps: u64, totalSteps: u64, fades: u64, wasAwake: bool,
}
impl Focus {
    pub fn new() -> Self { Self { level: 0.55, consciousSteps: 0, totalSteps: 0, fades: 0, wasAwake: true } }
    pub fn isAwake(&self) -> bool { self.level >= FOCUSTHRESHOLD }
    pub fn probability(&self) -> f64 { self.level.clamp(0.0, 1.0) }
    pub fn onStep(&mut self, hadSurprise: bool) {
        self.totalSteps += 1;
        if hadSurprise { self.level = (self.level + FOCUSGAIN).min(FOCUSMAX); }
        else { self.level = (self.level - FOCUSDECAY).max(0.0); }
        let awake = self.isAwake();
        if awake { self.consciousSteps += 1; }
        if self.wasAwake && !awake { self.fades += 1; }
        self.wasAwake = awake;
    }
    pub fn level(&self) -> f64 { self.level }
    pub fn fades(&self) -> u64 { self.fades }
    pub fn avgPercent(&self) -> f64 {
        if self.totalSteps == 0 { 0.0 } else { self.consciousSteps as f64 / self.totalSteps as f64 * 100.0 }
    }
}

pub fn colorForFocus(level: f64) -> String {
    let t = level.clamp(0.0, 1.0);
    let (r, g) = if t < 0.5 { (255u8, (t * 2.0 * 220.0) as u8) } else { (((1.0 - t) * 2.0 * 220.0) as u8, 220u8) };
    format!("\x1b[38;2;{};{};40m", r, g)
}

/// Quality ramp: green when the model is doing well, red when it is not.
pub fn colorForQuality(q: f64) -> String {
    let t = q.clamp(0.0, 1.0);
    let r = (255.0 * (1.0 - t) + 40.0 * t) as u8;
    let g = (60.0 * (1.0 - t) + 220.0 * t) as u8;
    format!("\x1b[38;2;{};{};40m", r, g)
}
pub fn resetColor() -> &'static str { "\x1b[0m" }
pub fn hitRate(hits: u64, steps: u64) -> f64 { if steps == 0 { 0.0 } else { hits as f64 / steps as f64 * 100.0 } }

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo random generator, no external crates needed.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn pick(&mut self, len: usize) -> usize { (self.next() as usize) % len }
    }

    /// The cached best entries of a row must be exactly what a full sort of
    /// the row gives, otherwise the candidate set of `top` would change.
    #[test]
    fn rowCacheMatchesFullSort() {
        let mut row = FeatureRow::new(Some(CAP));
        let mut rng = Rng(12345);
        for step in 0..4000 {
            let token = format!("t{}", rng.pick(40));
            row.bump(&token, INCREMENT);
            if step % 97 == 0 {
                assert_eq!(row.top, row.bestEntries(), "row cache diverged at step {}", step);
            }
        }
    }

    /// Same for the conscious edges.
    #[test]
    fn edgeCacheMatchesFullSort() {
        let mut freq: HashMap<String, u32> = HashMap::new();
        for i in 0..20 { freq.insert(format!("t{}", i), i); }
        let mut cons = Consciousness::new(freq);
        let mut rng = Rng(777);
        let mut work = LinearWork::new(P1);
        let ctx = Context::withBag(vec!["a".to_string(), "b".to_string()], Vec::new());
        for step in 0..300 {
            let proposals = vec![format!("p{}", rng.pick(5))];
            let target = format!("t{}", rng.pick(30));
            cons.receiveTrainingTrigger(&mut work, &ctx, &proposals, &target);
            if step % 31 == 0 {
                for (feature, out) in &cons.edges {
                    let mut all: Vec<(String, f64)> = out.iter().map(|(t, w)| (t.clone(), *w)).collect();
                    all.sort_by(|a, b| edgeBetter(a, b, &cons.globalFreq));
                    all.truncate(EDGETAKE);
                    let cached = cons.edgeTop.get(feature).cloned().unwrap_or_default();
                    assert_eq!(cached, all, "edge cache diverged for {}", feature);
                }
            }
        }
    }
}
