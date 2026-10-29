use std::borrow::Borrow;
use std::io::{self, Write};
use std::time::Instant;

use crate::corpus::{tokenizeDialogue, CorpusStatus, Dialogue, PromptModel};
use crate::mind::{
    colorForFocus, colorForQuality, hitRate, makeWindow, resetColor, Consciousness, Context, Focus,
    LinearWork, Subconscious, DLGWINDOW, PROPOSALCOUNT,
};

/// How often the training reports its progress.
pub const PROGRESSSTEP: u64 = 20000;
/// How often the live stream prints its block.
pub const LIVEBLOCK: u64 = 40;

/// Rolling quality of the live stream: 1.0 for a Top1 hit, 0.5 for a Top3 only
/// hit, 0.0 for a miss. It drives the green -> red colour of the [ ] block.
pub struct Quality { ewma: f64, alpha: f64 }

impl Quality {
    pub fn new(alpha: f64) -> Quality { Quality { ewma: 0.5, alpha } }
    pub fn observe(&mut self, top1: bool, top3: bool) -> f64 {
        let s = if top1 { 1.0 } else if top3 { 0.5 } else { 0.0 };
        self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * s;
        self.ewma
    }
}

/// Slide the window by one token.
fn slide(history: &[String], token: &str) -> Vec<String> {
    let mut h = history.to_vec();
    if h.is_empty() { return vec![token.to_string()]; }
    h.remove(0);
    h.push(token.to_string());
    h
}

#[derive(Default, Clone)]
pub struct ReplyStats {
    pub steps: u64, pub top1: u64, pub top3: u64, pub surprises: u64, pub wills: u64,
    pub turns: u64, pub turnTop1: u64, pub turnTop3: u64,
    /// How often the true token was reachable at all (it was one of the
    /// candidates of some feature). This is the ceiling of top1/top3.
    pub cand: u64,
    pub candSize: u64,
}

impl ReplyStats {
    /// Adds the counters of another chunk of the same corpus.
    pub fn merge(&mut self, other: &ReplyStats) {
        self.steps += other.steps;
        self.top1 += other.top1;
        self.top3 += other.top3;
        self.surprises += other.surprises;
        self.wills += other.wills;
        self.turns += other.turns;
        self.turnTop1 += other.turnTop1;
        self.turnTop3 += other.turnTop3;
        self.cand += other.cand;
        self.candSize += other.candSize;
    }
}

/// Stop conditions of a training run.
#[derive(Clone, Copy)]
pub struct TrainLimits {
    /// Wall clock deadline.
    pub deadline: Option<Instant>,
    /// Stop when the process grows above this many megabytes (0 = no limit).
    pub maxMb: u64,
    /// Memory budget of the mind in megabytes (0 = no budget). The number of
    /// remembered features follows it, so the run can go on for ever.
    pub ramBudgetMb: u64,
}

impl Default for TrainLimits {
    fn default() -> TrainLimits { TrainLimits { deadline: None, maxMb: 0, ramBudgetMb: 0 } }
}

/// How the beginning of the stage is shown live.
#[derive(Clone, Copy)]
pub struct LiveOptions {
    /// How many tokens are printed one by one (0 = no live output).
    pub steps: u64,
    /// Pause between the printed tokens, milliseconds.
    pub delayMs: u64,
}

impl Default for LiveOptions {
    fn default() -> LiveOptions { LiveOptions { steps: 0, delayMs: 0 } }
}

/// The one and only training stage: the mind reads the corpus, learns to
/// continue every message and optionally shows the first steps live.
pub fn runReplies<I, D>(
    dialogs: I,
    promptModel: &PromptModel,
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    work: &mut LinearWork,
    focus: &mut Focus,
    train: bool,
    limits: TrainLimits,
    status: Option<&CorpusStatus>,
    live: LiveOptions,
) -> ReplyStats
where I: Iterator<Item = D>, D: Borrow<Dialogue> {
    let mut st = ReplyStats::default();
    let started = Instant::now();
    let mut nextProgress = PROGRESSSTEP;
    let mut stopped = false;
    let mut quality = Quality::new(0.12);
    'corpus: for d in dialogs {
        let d = d.borrow();
        for i in 1..d.turns.len() {
            let prompt = &d.turns[i - 1];
            let bag = promptModel.bag(prompt);
            let mut history = makeWindow(prompt, DLGWINDOW);
            let mut first = true;
            for t in &d.turns[i] {
                let ctx = Context::withBag(history.clone(), bag.clone());
                let (props, cand) = sub.topWithCandidates(&ctx, PROPOSALCOUNT);
                st.steps += 1;
                st.candSize += cand.len() as u64;
                let reachable = cand.contains(t);
                if reachable { st.cand += 1; }
                let hit1 = props.first().map_or(false, |p| p == t);
                let hit3 = props.contains(t);
                if hit1 { st.top1 += 1; }
                if hit3 { st.top3 += 1; }
                if first {
                    st.turns += 1;
                    if hit1 { st.turnTop1 += 1; }
                    if hit3 { st.turnTop3 += 1; }
                    first = false;
                }
                let surprise = !hit3;
                if surprise { st.surprises += 1; }
                if train {
                    sub.receiveObservation(&ctx, t);
                    if surprise && work.unit() < focus.probability() {
                        cons.receiveTrainingTrigger(work, &ctx, &props, t);
                        if let Some(chosen) = cons.chooseAction(work, &ctx, &props) {
                            sub.receiveConsciousAction(&ctx, &chosen);
                            st.wills += 1;
                        }
                    }
                    focus.onStep(surprise);
                }
                if st.steps <= live.steps {
                    let q = quality.observe(hit1, hit3);
                    let mark = if surprise { "!" } else { "·" };
                    print!(
                        "{}{}{}{} ",
                        colorForFocus(focus.level()),
                        mark,
                        t,
                        resetColor());
                    if st.steps % LIVEBLOCK == 0 {
                        print!(
                            "\n  {}[f={:.2} {} q={:.2} t1={:.1}% t3={:.1}%]{}",
                            colorForQuality(q),
                            focus.level(),
                            if focus.isAwake() { "Y" } else { "n" },
                            q,
                            hitRate(st.top1, st.steps),
                            hitRate(st.top3, st.steps),
                            resetColor());
                        println!();
                        let _ = io::stdout().flush();
                    }
                    if live.delayMs > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(live.delayMs));
                    }
                }
                history = slide(&history, t);
            }
            if let Some(dl) = limits.deadline {
                if Instant::now() >= dl { stopped = true; break 'corpus; }
            }
        }
        if train && st.steps >= nextProgress {
            fitMemory(sub, cons, limits.ramBudgetMb);
            let rss = residentMb();
            println!(
                "  ... reply steps={} t1={:.1}% t3={:.1}% reach={:.1}% cands={:.0} ({:.1}s, {}/{}Mb){}",
                st.steps,
                hitRate(st.top1, st.steps),
                hitRate(st.top3, st.steps),
                hitRate(st.cand, st.steps),
                st.candSize as f64 / st.steps.max(1) as f64,
                started.elapsed().as_secs_f64(),
                rss,
                limits.ramBudgetMb,
                corpusProgress(status, started));
            let _ = io::stdout().flush();
            nextProgress += PROGRESSSTEP;
        }
        if limits.maxMb > 0 && residentMb() >= limits.maxMb { stopped = true; break 'corpus; }
    }
    if stopped {
        println!(
            "  ... training stopped early: steps={} ({:.1}s, {}Mb)",
            st.steps, started.elapsed().as_secs_f64(), residentMb());
    }
    st
}

/// Generate a reply for a user prompt, token by token, exactly the way the
/// training loop consumes the corpus.
pub fn generateReply(
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    work: &mut LinearWork,
    focus: &Focus,
    promptModel: &PromptModel,
    prompt: &[String],
    maxTokens: usize,
) -> String {
    let bag = promptModel.bag(prompt);
    let mut history = makeWindow(prompt, DLGWINDOW);
    let mut out = String::new();
    for _ in 0..maxTokens {
        let ctx = Context::withBag(history.clone(), bag.clone());
        let props = sub.top(&ctx, PROPOSALCOUNT);
        if props.is_empty() { break; }
        let mut chosen = props[0].clone();
        if work.unit() < focus.probability() {
            if let Some(pick) = cons.chooseAction(work, &ctx, &props) { chosen = pick; }
        }
        sub.receiveConsciousAction(&ctx, &chosen);
        let stop = chosen == "." || chosen == "?" || chosen == "!";
        if !out.is_empty() && !isPunctuation(&chosen) { out.push(' '); }
        out.push_str(&chosen);
        if stop { break; }
        history = slide(&history, &chosen);
    }
    out
}

fn isPunctuation(token: &str) -> bool {
    token == "." || token == "," || token == "?" || token == "!" || token == ";" || token == ":"
}

/// Keeps the mind inside the memory budget: when the process grows too big
/// the least used features are dropped, when it shrinks too much the mind is
/// allowed to remember more.
fn fitMemory(sub: &mut Subconscious, cons: &mut Consciousness, budgetMb: u64) {
    if budgetMb == 0 { return; }
    let rss = residentMb();
    let target = budgetMb * 85 / 100;
    let rows = sub.rowStats().0;
    let features = cons.edgeStats().0;
    if rss > target {
        let factor = (target as f64 / rss as f64).max(0.5);
        sub.setMaxRows(((rows as f64 * factor) as usize).max(1000));
        cons.setMaxFeatures(((features as f64 * factor) as usize).max(1000));
    } else if rss < budgetMb * 60 / 100 {
        sub.setMaxRows(rows + rows / 10 + 1000);
        cons.setMaxFeatures(features + features / 10 + 1000);
    }
}

/// Resident memory of the process, for the progress line of long runs.
pub fn residentMb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
                return kb / 1024;
            }
        }
    }
    0
}

/// "files 12345/1850268 (0.7%), eta 41m" for the streamed corpus.
fn corpusProgress(status: Option<&CorpusStatus>, started: Instant) -> String {
    let status = match status { Some(s) => s, None => return String::new() };
    let files = status.files.get();
    let total = status.totalFiles.get();
    if total == 0 { return String::new(); }
    let done = files as f64 / total as f64 * 100.0;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    let inChunk = files.saturating_sub(status.chunkStart.get()).max(1);
    let eta = (total.saturating_sub(files)) as f64 / (inChunk as f64 / elapsed);
    format!(
        " | files {}/{} ({:.1}%), eta {:.0}m",
        files, total, done, eta / 60.0)
}

/// Interactive chat: the user writes a line, the mind answers on enter.
pub fn chatLoop(
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    work: &mut LinearWork,
    focus: &mut Focus,
    promptModel: &PromptModel,
    maxTokens: usize,
) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lastPrompt: Vec<String> = Vec::new();
    println!();
    println!("=== CHAT ===");
    println!("write a line and press enter - the mind continues with its own reply");
    println!("commands: /quit  /help  /stats  /top");
    loop {
        print!("\nyou> ");
        let _ = stdout.flush();
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let text = line.trim().to_string();
        if text.is_empty() { continue; }
        if text.starts_with('/') {
            let mut parts = text.split_whitespace();
            let cmd = parts.next().unwrap_or("").to_string();
            match cmd.as_str() {
                "/quit" | "/exit" => { println!("bye"); break; }
                "/help" => { printHelp(); continue; }
                "/stats" => { printStats(sub, cons, focus, work); continue; }
                "/top" => { printTop(sub, cons, promptModel, &lastPrompt); continue; }
                other => { println!("unknown command: {}", other); continue; }
            }
        }
        let prompt = tokenizeDialogue(&text);
        if prompt.is_empty() { continue; }
        lastPrompt = prompt.clone();
        let reply = generateReply(sub, cons, work, focus, promptModel, &prompt, maxTokens);
        println!("mind> {}", reply);
        let _ = stdout.flush();
    }
}

fn printHelp() {
    println!();
    println!("/quit          leave the chat");
    println!("/stats         mind statistics");
    println!("/top           proposals for the last prompt");
}

fn printStats(sub: &Subconscious, cons: &Consciousness, focus: &Focus, work: &LinearWork) {
    println!();
    println!("focus={:.3} awake={} avgConsciousness={:.1}% fades={}",
        focus.level(), if focus.isAwake() { "Y" } else { "n" }, focus.avgPercent(), focus.fades());
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites(), sub.consciousWrites());
    println!("consciousActions={} failedUpdates={}", cons.actions(), cons.failedUpdates());
    println!("linearWork calls={} successes={} exhausted={}",
        work.calls(), work.successes(), work.exhausted());
}

fn printTop(sub: &Subconscious, cons: &Consciousness, promptModel: &PromptModel, prompt: &[String]) {
    println!();
    if prompt.is_empty() { println!("(no prompt yet)"); return; }
    let bag = promptModel.bag(prompt);
    println!("bag   : {}", bag.join(", "));
    let ctx = Context::withBag(makeWindow(prompt, DLGWINDOW), bag);
    let props = sub.top(&ctx, 5);
    let subScores = sub.scoreCandidates(&ctx, &props);
    let conScores = cons.scoreCandidates(&ctx, &props, &props);
    println!("reply : {}", props.join(", "));
    for (i, token) in props.iter().enumerate() {
        println!("  {:>12}  sub={:.4} con={:.4}", token, subScores[i].1, conScores[i].1);
    }
}
