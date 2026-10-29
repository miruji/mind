// The project keeps the camelCase naming of the original single file engine.
#![allow(non_snake_case)]

mod corpus;
mod dialogue;
mod mind;

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use corpus::{loadDialogues, loadDocs, PromptModel};
use dialogue::{ActMind, ActStats, ReplyStats};
use mind::{
    colorForFocus, colorForQuality, hitRate, makeContext, resetColor, Consciousness, Focus,
    LinearWork, Subconscious, CAP, FOCUSDECAY, FOCUSGAIN, FOCUSTHRESHOLD, P1, PROPOSALCOUNT,
};

const LIVESTEPS: u64 = 600;
const STEPDELAYMS: u64 = 20;
const DIALOGUE_ROOT: &str = "data2/ijcnlp_dailydialog";
const MAXREPLYTOKENS: usize = 24;
const QUALITYALPHA: f64 = 0.12;

/// Rolling quality of the live stream: 1.0 for a Top1 hit, 0.5 for a Top3
/// only hit, 0.0 for a miss. Drives the green -> red colour of the [ ] block.
struct Quality { ewma: f64, alpha: f64 }
impl Quality {
    fn new(alpha: f64) -> Quality { Quality { ewma: 0.5, alpha } }
    fn observe(&mut self, top1: bool, top3: bool) -> f64 {
        let s = if top1 { 1.0 } else if top3 { 0.5 } else { 0.0 };
        self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * s;
        self.ewma
    }
}

struct Options {
    steps: u64,
    limit: usize,
    chat: bool,
    delay: u64,
    maxReply: usize,
}

impl Options {
    fn parse() -> Options {
        let mut opts = Options {
            steps: LIVESTEPS,
            limit: 0,
            chat: true,
            delay: STEPDELAYMS,
            maxReply: MAXREPLYTOKENS,
        };
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--steps" => {
                    i += 1;
                    opts.steps = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(LIVESTEPS);
                }
                "--limit" => {
                    i += 1;
                    opts.limit = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "--delay" => {
                    i += 1;
                    opts.delay = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(STEPDELAYMS);
                }
                "--max-reply" => {
                    i += 1;
                    opts.maxReply = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(MAXREPLYTOKENS);
                }
                "--no-chat" => opts.chat = false,
                other => println!("unknown option: {}", other),
            }
            i += 1;
        }
        opts
    }
}

fn printActStats(label: &str, st: &ActStats) {
    println!(
        "{}: steps={} top1={:.1}% top3={:.1}% (hits {}/{} and {}/{})",
        label,
        st.steps,
        hitRate(st.top1, st.steps),
        hitRate(st.top3, st.steps),
        st.top1, st.steps, st.top3, st.steps);
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
}

fn main() {
    let opts = Options::parse();
    let started = Instant::now();

    // ================= phase 1: cmake docs, same live stream as before =====
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
    let mut quality = Quality::new(QUALITYALPHA);
    let total = opts.steps.min(pairs.len() as u64);

    println!("--- live stream ({} steps) ---\n", total);

    for step in 0..total {
        let idx = (step as usize) % pairs.len();
        let (ctx, target) = &pairs[idx];
        let props = sub.top(ctx, PROPOSALCOUNT);
        let isSurprise = !props.contains(target);
        scoredSteps += 1;
        let hit1 = props.first().map_or(false, |t| t == target);
        if hit1 { top1Hits += 1; }
        if props.contains(target) { top3Hits += 1; }
        let q = quality.observe(hit1, props.contains(target));
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

        let col = colorForFocus(focus.level());
        print!("{}{}{}{} ", col, mark, target, resetColor());

        if (step + 1) % 40 == 0 || step + 1 == total {
            print!(
                "\n  {}[f={:.2} {} q={:.2} t1={:.1}% t3={:.1}%]{}",
                colorForQuality(q),
                focus.level(),
                if focus.isAwake() { "Y" } else { "n" },
                q,
                hitRate(top1Hits, scoredSteps),
                hitRate(top3Hits, scoredSteps),
                resetColor());
            println!();
            let _ = stdout.flush();
        }
        if opts.delay > 0 { thread::sleep(Duration::from_millis(opts.delay)); }
    }

    println!();
    println!("=== RESULT (docs) ===");
    println!("steps={} surprises={} willActions={}", total, surpriseCount, willCount);
    println!(
        "top1={:.1}% top3={:.1}% (hits {}/{} and {}/{})",
        hitRate(top1Hits, scoredSteps),
        hitRate(top3Hits, scoredSteps),
        top1Hits, scoredSteps, top3Hits, scoredSteps);
    println!(
        "focus final={:.3} fades={} avgConsciousness={:.1}%",
        focus.level(), focus.fades(), focus.avgPercent());
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites(), sub.consciousWrites());

    // ================= phase 2: DailyDialog ================================
    let root = Path::new(DIALOGUE_ROOT);
    let (mut train, test) = loadDialogues(root);
    if opts.limit > 0 && train.len() > opts.limit { train.truncate(opts.limit); }
    let trainTurns: usize = train.iter().map(|d| d.turns.len()).sum();
    let testTurns: usize = test.iter().map(|d| d.turns.len()).sum();
    println!();
    println!("CORPUS={}; trainDialogs={} trainTurns={} testDialogs={} testTurns={}",
        DIALOGUE_ROOT, train.len(), trainTurns, test.len(), testTurns);

    let promptModel = PromptModel::build(&train);
    // The mind keeps everything it learned on the docs corpus, the dialogue
    // phase only adds the new words to its vocabulary.
    let dlgFreq = dialogue::dialogueFrequency(&train);
    sub.observeVocabulary(&dlgFreq);
    cons.observeVocabulary(&dlgFreq);

    let mut actMind = ActMind::new(&train);
    println!();
    println!("--- phase 2: learning the dialogs ---");
    let actTrain = dialogue::runActs(&train, &mut actMind, &promptModel, &mut work, &mut focus, true);
    printActStats("acts  train", &actTrain);
    let replyTrain = dialogue::runReplies(
        &train, &promptModel, &mut sub, &mut cons, &mut work, &mut focus, &actMind, true, false);
    printReplyStats("reply train", &replyTrain);
    {
        let (rows, entries, biggest) = sub.rowStats();
        let (feats, edges, biggestEdge) = cons.edgeStats();
        println!("mind: rows={} entries={} biggestRow={} | features={} edges={} biggestFeature={}",
            rows, entries, biggest, feats, edges, biggestEdge);
    }
    println!("phase 2 took {:.1}s", started.elapsed().as_secs_f64());

    // ================= phase 3: quality of the prediction ===================
    println!();
    println!("--- phase 3: evaluation on the test split ---");
    let actEval = dialogue::runActs(&test, &mut actMind, &promptModel, &mut work, &mut focus, false);
    printActStats("acts  test", &actEval);
    let replyEval = dialogue::runReplies(
        &test, &promptModel, &mut sub, &mut cons, &mut work, &mut focus, &actMind, false, true);
    printReplyStats("reply test (predicted act)", &replyEval);
    let replyOracle = dialogue::runReplies(
        &test, &promptModel, &mut sub, &mut cons, &mut work, &mut focus, &actMind, false, false);
    printReplyStats("reply test (gold act)", &replyOracle);
    println!("phase 3 took {:.1}s", started.elapsed().as_secs_f64());

    println!();
    println!("=== RESULT (dialogues) ===");
    println!(
        "acts  train top1={:.1}% top3={:.1}% | test top1={:.1}% top3={:.1}%",
        hitRate(actTrain.top1, actTrain.steps),
        hitRate(actTrain.top3, actTrain.steps),
        hitRate(actEval.top1, actEval.steps),
        hitRate(actEval.top3, actEval.steps));
    println!(
        "reply train top1={:.1}% top3={:.1}% | test(pipeline) top1={:.1}% top3={:.1}% | test(gold act) top1={:.1}% top3={:.1}%",
        hitRate(replyTrain.top1, replyTrain.steps),
        hitRate(replyTrain.top3, replyTrain.steps),
        hitRate(replyEval.top1, replyEval.steps),
        hitRate(replyEval.top3, replyEval.steps),
        hitRate(replyOracle.top1, replyOracle.steps),
        hitRate(replyOracle.top3, replyOracle.steps));
    println!(
        "focus final={:.3} fades={} avgConsciousness={:.1}%",
        focus.level(), focus.fades(), focus.avgPercent());
    println!("passiveWrites={} consciousWrites={}", sub.passiveWrites(), sub.consciousWrites());
    println!("linearWork rate={:.6}", work.successes() as f64 / work.calls().max(1) as f64);

    // ================= phase 4: interactive chat ===========================
    if opts.chat {
        dialogue::chatLoop(
            &mut sub, &mut cons, &mut work, &mut focus, &promptModel, &actMind, opts.maxReply);
    }
}
