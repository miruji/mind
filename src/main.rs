// The project keeps the camelCase naming of the original single file engine.
#![allow(non_snake_case)]

mod binio;
mod checkpoint;
mod corpus;
mod dialogue;
mod mind;

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use corpus::{countFiles, CountingDialogs, PromptModel, SplitDialogs, UbuntuDialogs};
use dialogue::{LiveOptions, ReplyStats, TrainLimits};
use mind::{
    hitRate, Focus, LinearWork, Subconscious, CAP, FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD, P1,
};

const UBUNTU_ROOT: &str = "data/ubuntu_dialogs/dialogs";
const MAXREPLYTOKENS: usize = 24;
const UBUNTU_SAMPLE: usize = 4000;
const UBUNTU_HOLDOUT: u64 = 10;
const UBUNTU_HOLDCAP: usize = 2000;
/// Upper bound for the number of feature rows the mind keeps.
const MAXFEATURES: usize = 1_000_000;
const LIVESTEPS: u64 = 600;
const STEPDELAYMS: u64 = 20;

struct Options {
    steps: u64,
    minutes: u64,
    ubuntuFiles: u64,
    ubuntuSample: usize,
    maxMb: u64,
    maxFeatures: usize,
    delay: u64,
    maxReply: usize,
    save: Option<String>,
    load: Option<String>,
    chat: bool,
}

impl Options {
    fn parse() -> Options {
        let mut opts = Options {
            steps: LIVESTEPS,
            minutes: 1,
            ubuntuFiles: 0,
            ubuntuSample: UBUNTU_SAMPLE,
            maxMb: 0,
            maxFeatures: MAXFEATURES,
            delay: STEPDELAYMS,
            maxReply: MAXREPLYTOKENS,
            save: None,
            load: None,
            chat: true,
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
                "--ubuntu-files" => {
                    i += 1;
                    opts.ubuntuFiles = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--ubuntu-sample" => {
                    i += 1;
                    opts.ubuntuSample = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(UBUNTU_SAMPLE);
                }
                "--max-mb" => {
                    i += 1;
                    opts.maxMb = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--max-features" => {
                    i += 1;
                    opts.maxFeatures = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(MAXFEATURES);
                }
                "--delay" => {
                    i += 1;
                    opts.delay = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(STEPDELAYMS);
                }
                "--max-reply" => {
                    i += 1;
                    opts.maxReply = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(MAXREPLYTOKENS);
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

fn main() {
    let opts = Options::parse();
    let started = Instant::now();

    let root = Path::new(UBUNTU_ROOT);
    if !root.exists() {
        println!("corpus not found: {}", UBUNTU_ROOT);
        return;
    }

    println!("CORPUS={}", UBUNTU_ROOT);
    println!("Live focus: decay={} gain={} threshold={}", FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD);
    println!("color: red=asleep → green=high focus | · passive  ! surprise  W will");
    println!("stats: t1/t3 = subconscious Top1/Top3 hit rate on the true next token");
    let totalFiles = if opts.ubuntuFiles > 0 { opts.ubuntuFiles } else { countFiles(root) };
    println!("files in corpus: {}", totalFiles);
    if opts.minutes > 0 {
        println!("budget: {} min", opts.minutes);
    } else {
        println!("budget: none, the whole corpus will be processed");
    }

    // ================= the mind ============================================
    let mut work = LinearWork::new(P1);
    let mut sub = Subconscious::new(HashMap::new(), Some(CAP));
    let mut cons = mind::Consciousness::new(HashMap::new());
    sub.setMaxRows(opts.maxFeatures);
    cons.setMaxFeatures(opts.maxFeatures);
    if opts.maxFeatures > 0 {
        println!("mind limit: {} features (the weakest are forgotten)", opts.maxFeatures);
    }
    let mut focus = Focus::new();
    let mut cp = checkpoint::Checkpoint::default();

    if let Some(path) = &opts.load {
        match checkpoint::load(Path::new(path), &mut sub, &mut cons) {
            Ok(loaded) => {
                cp = loaded;
                println!();
                println!("--- resumed from {} ---", path);
                println!("files done={} dialogues done={} steps done={}",
                    cp.filesDone, cp.dialoguesDone, cp.train.steps);
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
    let sample = UbuntuDialogs::new(root).take(opts.ubuntuSample);
    let promptModel = PromptModel::build(sample);
    println!("prompt model: {} files sampled", opts.ubuntuSample);

    // ================= one common stage on the corpus =======================
    println!();
    println!("--- stage: learning the corpus ---");
    let deadline = if opts.minutes > 0 {
        Some(Instant::now() + Duration::from_secs(opts.minutes * 60))
    } else {
        None
    };
    let mut ubuntu = CountingDialogs::new(SplitDialogs::at(
        UbuntuDialogs::withLimit(root, opts.ubuntuFiles, cp.filesDone),
        cp.dialoguesDone, UBUNTU_HOLDOUT, UBUNTU_HOLDCAP));
    let status = ubuntu.inner.inner.status.clone();
    status.chunkStart.set(cp.filesDone);
    let stats = dialogue::runReplies(
        &mut ubuntu, &promptModel, &mut sub, &mut cons, &mut work, &mut focus,
        true,
        TrainLimits { deadline, maxMb: opts.maxMb },
        Some(&status),
        LiveOptions { steps: opts.steps, delayMs: opts.delay });
    cp.train.merge(&stats);
    cp.filesDone = ubuntu.inner.inner.read;
    cp.dialoguesDone = ubuntu.inner.counter;
    println!();
    printReplyStats("reply (this chunk)", &stats);
    println!(
        "files read={} dialogues={} turns={} ({:.1}s)",
        ubuntu.inner.inner.read, ubuntu.inner.counter, ubuntu.inner.inner.turns,
        started.elapsed().as_secs_f64());

    // ================= quality on the held out part ========================
    let held = std::mem::take(&mut ubuntu.inner.held);
    if !held.is_empty() {
        let evalStats = dialogue::runReplies(
            held.iter(), &promptModel, &mut sub, &mut cons, &mut work, &mut focus,
            false, TrainLimits::default(), None, LiveOptions::default());
        cp.holdout.merge(&evalStats);
        println!();
        printReplyStats("reply holdout (this chunk)", &evalStats);
    }

    sub.observeVocabulary(&ubuntu.freq);
    cons.observeVocabulary(&ubuntu.freq);

    println!();
    println!("=== RESULT ===");
    printReplyStats("reply total", &cp.train);
    printReplyStats("reply holdout total", &cp.holdout);
    println!("files done={} of {}", cp.filesDone, totalFiles);
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
    println!("memory: {} Mb", dialogue::residentMb());
    println!("forgotten: rows={} features={}", sub.evicted(), cons.evicted());

    // ================= checkpoint ===========================================
    if let Some(path) = &opts.save {
        match checkpoint::save(Path::new(path), &sub, &cons, &cp) {
            Ok(()) => {
                let size = std::fs::metadata(path).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                println!();
                println!("checkpoint saved: {} ({} Mb), files done={}", path, size, cp.filesDone);
            }
            Err(err) => println!("cannot save checkpoint {}: {}", path, err),
        }
    }

    // ================= chat =================================================
    if opts.chat {
        dialogue::chatLoop(
            &mut sub, &mut cons, &mut work, &mut focus, &promptModel, opts.maxReply);
    }
}
