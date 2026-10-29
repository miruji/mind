use std::borrow::Borrow;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

use crate::corpus::{tokenizeDialogue, Dialogue, PromptModel};
use crate::mind::{
    makeWindow, Consciousness, Context, Focus, LinearWork, Subconscious, DLGWINDOW, MODE_ACT,
    MODE_REPLY, PROPOSALCOUNT,
};

/// Dialog act labels used by the act mind (isolated tokens, they never mix
/// with the ordinary vocabulary of the text mind).
pub const ACT_LABELS: [&str; 4] = ["act1", "act2", "act3", "act4"];
/// DailyDialog acts: 1 inform, 2 question, 3 directive, 4 commissive.
pub const ACT_NAMES: [&str; 4] = ["inform", "question", "directive", "commissive"];
/// How often the reply training reports its progress.
pub const PROGRESSSTEP: u64 = 20000;

pub fn actLabel(act: u8) -> &'static str { ACT_LABELS[(act.clamp(1, 4) - 1) as usize] }
pub fn actNumber(label: &str) -> u8 {
    for (i, l) in ACT_LABELS.iter().enumerate() { if *l == label { return (i + 1) as u8; } }
    1
}
pub fn actName(act: u8) -> &'static str { ACT_NAMES[(act.clamp(1, 4) - 1) as usize] }

/// Separate mind for the dialog act classifier. It uses the very same
/// subconscious / consciousness machinery, but keeps its own vocabulary so
/// act labels can never leak into the reply candidates.
pub struct ActMind { pub sub: Subconscious, pub cons: Consciousness }

impl ActMind {
    pub fn new(dialogs: &[Dialogue]) -> ActMind {
        let mut freq: HashMap<String, u32> = HashMap::new();
        for l in ACT_LABELS.iter() { freq.insert(l.to_string(), 0); }
        for d in dialogs {
            for a in &d.acts { *freq.entry(actLabel(*a).to_string()).or_insert(0) += 1; }
        }
        ActMind { sub: Subconscious::new(freq.clone(), None), cons: Consciousness::new(freq) }
    }

    /// Act context: the prompt turn is the window, its content words are the bag.
    pub fn context(&self, prompt: &[String], promptModel: &PromptModel) -> Context {
        Context::withBag(makeWindow(prompt, DLGWINDOW), promptModel.bag(prompt), MODE_ACT)
    }

    /// Predicted act and the raw proposals behind it.
    pub fn predict(&self, ctx: &Context) -> (u8, Vec<String>) {
        let props = self.sub.top(ctx, PROPOSALCOUNT);
        let act = props.first().map(|t| actNumber(t)).unwrap_or(1);
        (act, props)
    }
}

/// Token frequency over a set of dialogues (used to grow the vocabulary of
/// the mind without touching the statistics collected so far).
pub fn dialogueFrequency(dialogs: &[Dialogue]) -> HashMap<String, u32> {
    let mut freq: HashMap<String, u32> = HashMap::new();
    for d in dialogs {
        for turn in &d.turns {
            for t in turn { *freq.entry(t.clone()).or_insert(0) += 1; }
        }
    }
    freq
}

/// Slide the window by one token.
fn slide(history: &[String], token: &str) -> Vec<String> {
    let mut h = history.to_vec();
    if h.is_empty() { return vec![token.to_string()]; }
    h.remove(0);
    h.push(token.to_string());
    h
}

#[derive(Default)]
pub struct ActStats { pub steps: u64, pub top1: u64, pub top3: u64, pub surprises: u64, pub wills: u64 }

/// Where the `act:` feature of the reply model comes from.
#[derive(Clone, Copy, PartialEq)]
pub enum ActSource {
    /// The gold act of the corpus (teacher forcing).
    Gold,
    /// The act predicted by the act mind (what the chat really uses).
    Predicted,
}

/// Stop conditions of a training run.
#[derive(Clone, Copy)]
pub struct TrainLimits {
    /// Wall clock deadline.
    pub deadline: Option<Instant>,
    /// Stop when the process grows above this many megabytes (0 = no limit).
    pub maxMb: u64,
}

impl Default for TrainLimits {
    fn default() -> TrainLimits { TrainLimits { deadline: None, maxMb: 0 } }
}

/// Train (or only evaluate) the act classifier.
pub fn runActs<I, D>(
    dialogs: I,
    mind: &mut ActMind,
    promptModel: &PromptModel,
    work: &mut LinearWork,
    focus: &mut Focus,
    train: bool,
) -> ActStats
where I: Iterator<Item = D>, D: Borrow<Dialogue> {
    let mut st = ActStats::default();
    for d in dialogs {
        let d = d.borrow();
        if d.acts.len() != d.turns.len() { continue; }
        for i in 1..d.turns.len() {
            let ctx = mind.context(&d.turns[i - 1], promptModel);
            let target = actLabel(d.acts[i]);
            let props = mind.sub.top(&ctx, PROPOSALCOUNT);
            let hit1 = props.first().map_or(false, |t| t == target);
            let hit3 = props.iter().any(|t| t == target);
            st.steps += 1;
            if hit1 { st.top1 += 1; }
            if hit3 { st.top3 += 1; }
            let surprise = !hit3;
            if surprise { st.surprises += 1; }
            if train {
                mind.sub.receiveObservation(&ctx, target);
                if surprise && work.unit() < focus.probability() {
                    mind.cons.receiveTrainingTrigger(work, &ctx, &props, target);
                    if let Some(chosen) = mind.cons.chooseAction(work, &ctx, &props) {
                        mind.sub.receiveConsciousAction(&ctx, &chosen);
                        st.wills += 1;
                    }
                }
                focus.onStep(surprise);
            }
        }
    }
    st
}

#[derive(Default)]
pub struct ReplyStats {
    pub steps: u64, pub top1: u64, pub top3: u64, pub surprises: u64, pub wills: u64,
    pub turns: u64, pub turnTop1: u64, pub turnTop3: u64,
    /// How often the true token was reachable at all (it was one of the
    /// candidates of some feature). This is the ceiling of top1/top3.
    pub cand: u64,
    pub candSize: u64,
}

/// Train (or only evaluate) the reply model.
///
/// `actSource` selects what the `act:` feature carries: the gold act of the
/// training corpus (teacher forcing), the act predicted by the act mind (the
/// honest pipeline that the chat uses) or no act at all.
pub fn runReplies<I, D>(
    dialogs: I,
    promptModel: &PromptModel,
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    work: &mut LinearWork,
    focus: &mut Focus,
    actMind: &ActMind,
    train: bool,
    actSource: ActSource,
    limits: TrainLimits,
) -> ReplyStats
where I: Iterator<Item = D>, D: Borrow<Dialogue> {
    let mut st = ReplyStats::default();
    let started = Instant::now();
    let mut nextProgress = PROGRESSSTEP;
    let mut stopped = false;
    'corpus: for d in dialogs {
        let d = d.borrow();
        for i in 1..d.turns.len() {
            let prompt = &d.turns[i - 1];
            let bag = promptModel.bag(prompt);
            let act = match actSource {
                ActSource::Gold => d.acts.get(i).copied().unwrap_or(0),
                ActSource::Predicted => {
                    let actCtx = actMind.context(prompt, promptModel);
                    actMind.predict(&actCtx).0
                }
            };
            let mut history = makeWindow(prompt, DLGWINDOW);
            let mut first = true;
            for t in &d.turns[i] {
                let ctx = Context::withBag(history.clone(), bag.clone(), MODE_REPLY).withAct(act);
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
                history = slide(&history, t);
            }
            if let Some(dl) = limits.deadline {
                if Instant::now() >= dl { stopped = true; break 'corpus; }
            }
        }
        if train && st.steps >= nextProgress {
            println!(
                "  ... reply steps={} t1={:.1}% t3={:.1}% reach={:.1}% cands={:.0} ({:.1}s, {}Mb)",
                st.steps,
                crate::mind::hitRate(st.top1, st.steps),
                crate::mind::hitRate(st.top3, st.steps),
                crate::mind::hitRate(st.cand, st.steps),
                st.candSize as f64 / st.steps.max(1) as f64,
                started.elapsed().as_secs_f64(),
                residentMb());
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
    actMind: &ActMind,
    prompt: &[String],
    maxTokens: usize,
    forcedAct: Option<u8>,
) -> (String, u8, Vec<String>) {
    let bag = promptModel.bag(prompt);
    let actCtx = actMind.context(prompt, promptModel);
    let (predicted, actProps) = actMind.predict(&actCtx);
    let act = forcedAct.unwrap_or(predicted);
    let mut history = makeWindow(prompt, DLGWINDOW);
    let mut out = String::new();
    for _ in 0..maxTokens {
        let ctx = Context::withBag(history.clone(), bag.clone(), MODE_REPLY).withAct(act);
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
    (out, act, actProps)
}

fn isPunctuation(token: &str) -> bool {
    token == "." || token == "," || token == "?" || token == "!" || token == ";" || token == ":"
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

/// Interactive chat: the user writes a line, the mind answers on enter.
pub fn chatLoop(
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    work: &mut LinearWork,
    focus: &mut Focus,
    promptModel: &PromptModel,
    actMind: &ActMind,
    maxTokens: usize,
) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lastPrompt: Vec<String> = Vec::new();
    let mut forcedAct: Option<u8> = None;
    println!();
    println!("=== CHAT ===");
    println!("write a line and press enter - the mind continues with its own reply");
    println!("commands: /quit  /help  /stats  /act <1-4>  /top");
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
                "/top" => { printTop(sub, cons, promptModel, actMind, &lastPrompt); continue; }
                "/act" => {
                    match parts.next().and_then(|v| v.parse::<u8>().ok()) {
                        Some(a) if (1..=4).contains(&a) => {
                            forcedAct = Some(a);
                            println!("act forced: {} ({})", a, actName(a));
                        }
                        _ => println!("usage: /act <1-4>"),
                    }
                    continue;
                }
                other => { println!("unknown command: {}", other); continue; }
            }
        }
        let prompt = tokenizeDialogue(&text);
        if prompt.is_empty() { continue; }
        lastPrompt = prompt.clone();
        let (reply, act, actProps) = generateReply(
            sub, cons, work, focus, promptModel, actMind, &prompt, maxTokens, forcedAct);
        println!(
            "mind[{}:{}]> {}",
            actName(act),
            actProps.first().map(|t| t.to_string()).unwrap_or_default(),
            reply);
        let _ = stdout.flush();
    }
}

fn printHelp() {
    println!();
    println!("/quit          leave the chat");
    println!("/stats         mind statistics");
    println!("/top           proposals for the last prompt");
    println!("/act <1-4>     force the dialog act of the reply (inform/question/directive/commissive)");
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

fn printTop(
    sub: &Subconscious,
    cons: &Consciousness,
    promptModel: &PromptModel,
    actMind: &ActMind,
    prompt: &[String],
) {
    println!();
    if prompt.is_empty() { println!("(no prompt yet)"); return; }
    let actCtx = actMind.context(prompt, promptModel);
    let (act, actProps) = actMind.predict(&actCtx);
    println!("act   : {} -> {}", actName(act), actProps.join(", "));
    let bag = promptModel.bag(prompt);
    println!("bag   : {}", bag.join(", "));
    let ctx = Context::withBag(makeWindow(prompt, DLGWINDOW), bag, MODE_REPLY).withAct(act);
    let props = sub.top(&ctx, 5);
    let subScores = sub.scoreCandidates(&ctx, &props);
    let conScores = cons.scoreCandidates(&ctx, &props, &props);
    println!("reply : {}", props.join(", "));
    for (i, token) in props.iter().enumerate() {
        let s = subScores[i].1;
        let c = conScores[i].1;
        println!("  {:>12}  sub={:.4} con={:.4}", token, s, c);
    }
}
