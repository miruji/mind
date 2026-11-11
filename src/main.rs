// The project keeps the camelCase naming of the original single file engine.
#![allow(non_snake_case)]

mod binio;
mod checkpoint;
mod corpus;
mod dialogue;
mod mind;

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

use corpus::{countDialogs, MovieCorpus, MovieDialogs, PromptModel, SplitDialogs};
use dialogue::{LiveOptions, ReplyStats, TrainLimits};
use mind::{
    hitRate, Focus, LinearWork, Subconscious, CAP, FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD, P1,
};

const MOVIE_ROOT: &str = "data/data";
const MAXREPLYTOKENS: usize = 24;
const MOVIE_SAMPLE: usize = 4000;
const MOVIE_HOLDOUT: u64 = 10;
const MOVIE_HOLDCAP: usize = 2000;
/// Memory budget of the mind in gigabytes.
const RAMBUDGETGB: f64 = 8.0;
const LIVESTEPS: u64 = 600;
/// How many previous turns of the dialogue give the mind its topic.
const CHATCONTEXT: usize = 3;
const STEPDELAYMS: u64 = 20;

struct Options {
    steps: u64,
    minutes: u64,
    files: u64,
    sample: usize,
    repeat: u32,
    maxMb: u64,
    ramGb: f64,
    delay: u64,
    maxReply: usize,
    save: Option<String>,
    load: Option<String>,
    log: Option<String>,
    chat: bool,
    chatOnly: bool,
    chatContext: usize,
}

impl Options {
    fn parse() -> Options {
        let mut opts = Options {
            steps: LIVESTEPS,
            minutes: 1,
            files: 0,
            sample: MOVIE_SAMPLE,
            repeat: 1,
            maxMb: 0,
            ramGb: RAMBUDGETGB,
            delay: STEPDELAYMS,
            maxReply: MAXREPLYTOKENS,
            save: None,
            load: None,
            log: None,
            chat: true,
            chatOnly: false,
            chatContext: CHATCONTEXT,
        };
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--steps" => {
                    i += 1;
                    opts.steps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(LIVESTEPS);
                }
                "--minutes" => {
                    i += 1;
                    opts.minutes = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1);
                }
                "--files" | "--ubuntu-files" => {
                    i += 1;
                    opts.files = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--repeat" => {
                    i += 1;
                    opts.repeat = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
                }
                "--sample" | "--ubuntu-sample" => {
                    i += 1;
                    opts.sample = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(MOVIE_SAMPLE);
                }
                "--max-mb" => {
                    i += 1;
                    opts.maxMb = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--ram" => {
                    i += 1;
                    opts.ramGb = args.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(RAMBUDGETGB);
                }
                "--delay" => {
                    i += 1;
                    opts.delay = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(STEPDELAYMS);
                }
                "--max-reply" => {
                    i += 1;
                    opts.maxReply = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(MAXREPLYTOKENS);
                }
                "--log" => {
                    i += 1;
                    opts.log = args.get(i).cloned();
                }
                "--window" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse::<usize>().ok()) { mind::WINDOW.store(v.clamp(1, 12), std::sync::atomic::Ordering::Relaxed); }
                }
                "--joint" => {
                    i += 1;
                    mind::JOINT.store(parseJoint(args.get(i).map(|x| x.as_str()).unwrap_or("")), std::sync::atomic::Ordering::Relaxed);
                }
                "--skip" => {
                    i += 1;
                    let _ = mind::SKIPS.set(parseSkips(args.get(i).map(|x| x.as_str()).unwrap_or("")));
                }
                "--confK" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse::<f64>().ok()) { mind::CONFK.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed); }
                }
                "--proposals" => {
                    i += 1;
                    mind::PROPOSALS.store(args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
                }
                "--rowtake" => {
                    i += 1;
                    if let Some(v) = args.get(i).and_then(|v| v.parse::<usize>().ok()) { mind::ROWTAKEN.store(v.clamp(1, 256), std::sync::atomic::Ordering::Relaxed); }
                }
                "--save" => {
                    i += 1;
                    opts.save = args.get(i).cloned();
                }
                "--load" => {
                    i += 1;
                    opts.load = args.get(i).cloned();
                }
                "--no-chat" => opts.chat = false,
                "--chat-only" => { opts.chatOnly = true; opts.chat = true; }
                "--chat-context" => {
                    i += 1;
                    opts.chatContext = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(CHATCONTEXT);
                }
                other => println!("unknown option: {}", other),
            }
            i += 1;
        }
        opts
    }
}

fn printReplyStats(label: &str, st: &ReplyStats) {
    println!(
        "{}: steps={} top1={:.1}% top3={:.1}% (hits {}/{} and {}/{})",
        label,
        st.steps,
        hitRate(st.top1, st.steps),
        hitRate(st.top3, st.steps),
        st.top1, st.steps, st.top3, st.steps);
    println!(
        "{}: reply starts top1={:.1}% top3={:.1}% (hits {}/{} and {}/{})",
        label,
        hitRate(st.turnTop1, st.turns),
        hitRate(st.turnTop3, st.turns),
        st.turnTop1, st.turns, st.turnTop3, st.turns);
    println!(
        "{}: reachable={:.1}% (ceiling for top1/top3), candidates per step={:.0}",
        label,
        hitRate(st.cand, st.steps),
        st.candSize as f64 / st.steps.max(1) as f64);
}

fn parseJoint(list: &str) -> u32 {
    let mut mask: u32 = 0;
    for k in list.split(',') { if let Ok(k) = k.trim().parse::<u32>() { if (3..=8).contains(&k) { mask |= 1u32 << k; } } }
    mask
}

fn parseSkips(list: &str) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for code in list.split(',') {
        let c: Vec<char> = code.trim().chars().collect();
        if c.len() == 2 { if let (Some(a), Some(b)) = (c[0].to_digit(10), c[1].to_digit(10)) { v.push((a as usize, b as usize)); } }
    }
    v
}

/// The feature set decides what the rows mean, so it is stored next to a checkpoint
/// (`<file>.cfg`) and restored on --load: a chain of sessions must keep one set.
fn featureConfig() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let joint: Vec<String> = (3..=8u32).filter(|k| mind::JOINT.load(Relaxed) & (1u32 << k) != 0).map(|k| k.to_string()).collect();
    let skips: Vec<String> = mind::SKIPS.get().map(|v| v.iter().map(|(a, b)| format!("{}{}", a, b)).collect()).unwrap_or_default();
    format!("window={} joint={} skip={} confK={} rowtake={} proposals={}",
        mind::dlgWindow(), joint.join(","), skips.join(","), f64::from_bits(mind::CONFK.load(Relaxed)), mind::rowTake(), mind::proposalsLimit())
}

fn applyFeatureConfig(text: &str) {
    use std::sync::atomic::Ordering::Relaxed;
    for part in text.split_whitespace() {
        let mut kv = part.splitn(2, '=');
        let (k, v) = (kv.next().unwrap_or(""), kv.next().unwrap_or(""));
        match k {
            "window" => { if let Ok(x) = v.parse::<usize>() { mind::WINDOW.store(x, Relaxed); } }
            "joint" => mind::JOINT.store(parseJoint(v), Relaxed),
            "skip" => { let _ = mind::SKIPS.set(parseSkips(v)); }
            "confK" => { if let Ok(x) = v.parse::<f64>() { mind::CONFK.store(x.to_bits(), Relaxed); } }
            "rowtake" => { if let Ok(x) = v.parse::<usize>() { mind::ROWTAKEN.store(x, Relaxed); } }
            "proposals" => { if let Ok(x) = v.parse::<usize>() { mind::PROPOSALS.store(x, Relaxed); } }
            _ => {}
        }
    }
}

/// One line per pass in the session log.
fn writeSessionRow(
    path: Option<&str>,
    pass: u32,
    stats: &ReplyStats,
    hold: &ReplyStats,
    sub: &mind::Subconscious,
    cons: &mind::Consciousness,
    focus: &mind::Focus,
    from: u64,
    to: u64,
    seconds: f64,
) {
    let path = match path { Some(p) => p, None => return };
    use std::io::Write;
    let isNew = !Path::new(path).exists();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        if isNew {
            let _ = writeln!(f, "dialogsDone,dialogsThisPass,stepsThisPass,trainTop1,trainTop3,trainReach,surprisePct,willPct,holdTop1,holdTop3,holdReach,holdSurprisePct,rows,entries,features,edges,rssMb,tablesMb,evictedRows,focusFinal,seconds,features,pass");
        }
        let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 };
        let (rows, entries, _) = sub.rowStats();
        let (feats, edges, _) = cons.edgeStats();
        let _ = writeln!(f, "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{},{},{},{},{:.3},{:.0},{},{}",
            to, to.saturating_sub(from), stats.steps,
            pct(stats.top1, stats.steps), pct(stats.top3, stats.steps), pct(stats.cand, stats.steps),
            pct(stats.surprises, stats.steps), pct(stats.wills, stats.steps),
            pct(hold.top1, hold.steps), pct(hold.top3, hold.steps), pct(hold.cand, hold.steps),
            pct(hold.surprises, hold.steps), rows, entries, feats, edges,
            dialogue::residentMb(), dialogue::mindBytes(sub, cons) / (1024 * 1024), sub.evicted(),
            focus.level(), seconds, featureConfig().replace(' ', ";"), pass);
    }
}

fn main() {
    let opts = Options::parse();
    let started = Instant::now();

    let root = Path::new(MOVIE_ROOT);
    if !root.exists() {
        println!("corpus not found: {}", MOVIE_ROOT);
        return;
    }
    let corpus = match MovieCorpus::load(root) {
        Ok(c) => Rc::new(c),
        Err(err) => {
            println!("cannot read corpus {}: {}", MOVIE_ROOT, err);
            return;
        }
    };

    println!("CORPUS={}", MOVIE_ROOT);
    println!("Live focus: decay={} gain={} threshold={}", FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD);
    println!("color: red=asleep → green=high focus | · passive  ! surprise  W will");
    println!("stats: t1/t3 = subconscious Top1/Top3 hit rate on the true next token");
    let totalDialogs = if opts.files > 0 { opts.files } else { countDialogs(&corpus) };
    println!("dialogs in corpus: {} (utterances {})", totalDialogs, corpus.utterances);
    if opts.minutes > 0 {
        println!("budget: {} min", opts.minutes);
    } else {
        println!("budget: none, the whole corpus will be processed");
    }

    // ================= the mind ============================================
    let mut work = LinearWork::new(P1);
    let mut sub = Subconscious::new(HashMap::new(), Some(CAP));
    let mut cons = mind::Consciousness::new(HashMap::new());
    let ramBudgetMb = (opts.ramGb * 1024.0) as u64;
    if opts.ramGb > 0.0 {
        println!("memory budget: {} Gb for the mind (the weakest features are forgotten)", opts.ramGb);
    }
    let mut focus = Focus::new();
    let mut cp = checkpoint::Checkpoint::default();

    if let Some(path) = &opts.load {
        let cfgPath = format!("{}.cfg", path);
        match std::fs::read_to_string(&cfgPath) {
            Ok(text) => {
                let before = featureConfig();
                applyFeatureConfig(&text);
                println!("features restored from {}: {}", cfgPath, featureConfig());
                if before != featureConfig() { println!("(command line features were replaced: a session chain must keep one feature set)"); }
            }
            Err(_) => println!("warning: {} not found, features come from the command line: {}", cfgPath, featureConfig()),
        }
    }
    println!("features: {}", featureConfig());
    if let Some(path) = &opts.load {
        match checkpoint::load(Path::new(path), &mut sub, &mut cons) {
            Ok(loaded) => {
                cp = loaded;
                println!();
                println!("--- resumed from {} ---", path);
                println!("dialogs done={} seen={} steps done={}",
                    cp.dialogsDone, cp.dialogsSeen, cp.train.steps);
                println!("mind: rows={} features={} edges={}",
                    sub.rowStats().0, cons.edgeStats().0, cons.edgeStats().1);
            }
            Err(err) => {
                println!("cannot read checkpoint {}: {}", path, err);
                return;
            }
        }
    }

    // The prompt model only needs the discriminative words of the corpus, it
    // is collected in one pass over a small sample and then thrown away.
    let sample = MovieDialogs::new(corpus.clone()).take(opts.sample);
    let promptModel = PromptModel::build(sample);
    println!("prompt model: {} dialogs sampled", opts.sample);

    // ================= one common stage on the corpus =======================
    println!();
    println!("--- stage: learning the corpus ---");
    let deadline = if opts.minutes > 0 {
        Some(Instant::now() + Duration::from_secs(opts.minutes * 60))
    } else {
        None
    };
    let filesAtStart = cp.dialogsDone;
    let stageStart = Instant::now();
    // The range of the session is [start, end). With --repeat the same range
    // is read again and again: the holdout stays the same every pass, so the
    // passes can be compared with each other.
    let rangeStart = cp.dialogsDone;
    let rangeEnd = if opts.files > 0 { opts.files } else { countDialogs(&corpus) };
    let counterBase = cp.dialogsSeen;
    let mut position = rangeStart;
    let mut counter = counterBase;
    for pass in 1..=if opts.chatOnly { 0 } else { opts.repeat } {
        let mut corpusStream = SplitDialogs::at(
            MovieDialogs::withLimit(corpus.clone(), rangeEnd, position),
            counter, MOVIE_HOLDOUT, MOVIE_HOLDCAP);
        let status = corpusStream.inner.status.clone();
        status.chunkStart.set(position);
        let stats = dialogue::runReplies(
            &mut corpusStream, &promptModel, &mut sub, &mut cons, &mut work, &mut focus,
            true, mind::proposalsLimit(),
            TrainLimits { deadline, maxMb: opts.maxMb, ramBudgetMb },
            Some(&status),
            LiveOptions { steps: opts.steps, delayMs: opts.delay });
        cp.train.merge(&stats);
        let read = corpusStream.inner.read;
        let seen = corpusStream.counter;
        println!();
        printReplyStats(&format!("reply (pass {})", pass), &stats);
        println!(
            "dialogs read={} seen={} turns={} ({:.1}s)",
            read, seen, corpusStream.inner.turns, started.elapsed().as_secs_f64());

        // ================= quality on the held out part ====================
        let held = std::mem::take(&mut corpusStream.held);
        let mut holdChunk = ReplyStats::default();
        if !held.is_empty() {
            let evalStats = dialogue::runReplies(
                held.iter(), &promptModel, &mut sub, &mut cons, &mut work, &mut focus,
                false, mind::proposalsLimit(), TrainLimits::default(), None, LiveOptions::default());
            cp.holdout.merge(&evalStats);
            println!();
            printReplyStats(&format!("reply holdout (pass {})", pass), &evalStats);
            holdChunk = evalStats;
        }
        writeSessionRow(
            opts.log.as_deref(), pass, &stats, &holdChunk, &sub, &cons, &focus,
            position, read, stageStart.elapsed().as_secs_f64());

        if read < rangeEnd {
            // the pass was cut short (time or memory): keep the place
            position = read;
            counter = seen;
            break;
        }
        position = rangeEnd;
        counter = seen;
        if pass < opts.repeat {
            position = rangeStart;
            counter = counterBase;
        }
    }
    cp.dialogsDone = position;
    cp.dialogsSeen = counter;

    // The vocabulary of the mind is taken from its own tables: no separate
    // frequency map is kept in memory during the run.
    let freqs = sub.tokenFrequencies();
    sub.observeVocabulary(&freqs);
    cons.observeVocabulary(&freqs);

    println!();
    println!("=== RESULT ===");
    printReplyStats("reply total", &cp.train);
    printReplyStats("reply holdout total", &cp.holdout);
    println!("dialogs done={} of {}", cp.dialogsDone, totalDialogs);
    println!(
        "focus final={:.3} fades={} avgConsciousness={:.1}%",
        focus.level(), focus.fades(), focus.avgPercent());
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites(), sub.consciousWrites());
    println!("linearWork rate={:.6}", work.successes() as f64 / work.calls().max(1) as f64);
    {
        let (rows, entries, biggest) = sub.rowStats();
        let (feats, edges, biggestEdge) = cons.edgeStats();
        println!("mind: rows={} entries={} biggestRow={} | features={} edges={} biggestFeature={}",
            rows, entries, biggest, feats, edges, biggestEdge);
    }
    println!("stage took {:.1}s", started.elapsed().as_secs_f64());
    println!("memory: {} Mb rss, {} Mb of tables", dialogue::residentMb(), dialogue::mindBytes(&sub, &cons) / (1024 * 1024));
    println!("forgotten: rows={} features={}", sub.evicted(), cons.evicted());
    {
        let elapsed = stageStart.elapsed().as_secs_f64().max(0.001);
        let rate = (cp.dialogsDone.saturating_sub(filesAtStart)) as f64 / elapsed;
        let remaining = totalDialogs.saturating_sub(cp.dialogsDone);
        let eta = remaining as f64 / rate.max(0.0001);
        println!("eta to the end of the corpus: {:.0} min ({} dialogs left, {:.0} dialogs/s)",
            eta / 60.0, remaining, rate);
    }

    // ================= checkpoint ===========================================
    if let Some(path) = &opts.save {
        match checkpoint::save(Path::new(path), &sub, &cons, &cp) {
            Ok(()) => {
                let size = std::fs::metadata(path).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                println!();
                println!("checkpoint saved: {} ({} Mb), dialogs done={}", path, size, cp.dialogsDone);
                let _ = std::fs::write(format!("{}.cfg", path), featureConfig());
            }
            Err(err) => println!("cannot save checkpoint {}: {}", path, err),
        }
    }

    // ================= chat =================================================
    if opts.chat {
        dialogue::chatLoop(
            &mut sub, &mut cons, &mut work, &mut focus, &promptModel, opts.maxReply,
            opts.chatContext);
    }
}
