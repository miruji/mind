> ./target/release/mind --load data/runs/m.bin --chat-only
CORPUS=data/data
Live focus: decay=0.035 gain=0.28 threshold=0.15
color: red=asleep → green=high focus | · passive  ! surprise  W will
stats: t1/t3 = subconscious Top1/Top3 hit rate on the true next token
dialogs in corpus: 83034 (utterances 304446)
budget: 1 min
memory budget: 8 Gb for the mind (the weakest features are forgotten)
features restored from data/runs/m.bin.cfg: window=6 joint=3,4,5,6 skip= confK=4 rowtake=8 proposals=8
(command line features were replaced: a session chain must keep one feature set)
features: window=6 joint=3,4,5,6 skip= confK=4 rowtake=8 proposals=8

--- resumed from data/runs/m.bin ---
dialogs done=83034 seen=83034 steps done=7411740
mind: rows=8407780 features=5853408 edges=8512379
prompt model: 4000 dialogs sampled

--- stage: learning the corpus ---

=== RESULT ===
reply total: steps=7411740 top1=59.4% top3=75.8% (hits 4405290/7411740 and 5620378/7411740)
reply total: reply starts top1=41.4% top3=63.2% (hits 247356/597378 and 377292/597378)
reply total: reachable=86.0% (ceiling for top1/top3), candidates per step=25
reply holdout total: steps=201261 top1=19.1% top3=34.5% (hits 38528/201261 and 69391/201261)
reply holdout total: reply starts top1=2.9% top3=12.3% (hits 474/16302 and 2005/16302)
reply holdout total: reachable=59.8% (ceiling for top1/top3), candidates per step=25
dialogs done=83034 of 83034
focus final=0.550 fades=0 avgConsciousness=0.0%
passiveWrites=0 consciousWrites=0
linearWork rate=0.000000
mind: rows=8407780 entries=13808518 biggestRow=489 | features=5853408 edges=8512379 biggestFeature=32
stage took 25.6s
memory: 10885 Mb rss, 9007 Mb of tables
forgotten: rows=0 features=0
eta to the end of the corpus: 0 min (0 dialogs left, 0 dialogs/s)

=== CHAT ===
write a line and press enter - the mind continues with its own reply
commands: /quit  /help  /stats  /top

you> hellol
mind> you have to do is.

you> hi
mind> hi, how'd your day go?

you> i'm okay
mind> .

you> how about you
mind> ?

you> what
mind> don't have to.

you> you see,
mind> it was.

you> you see,
mind> it doesn't really matter, a.

you> you see?
mind> yes.

you> really?
mind> i'm going out!

you> you work?
mind> did work.

you> really?
mind> i don't feel so great.

you> what's wrong?
mind> it's hot in here for.

you> why is it so hot in here?
mind> hot that night and no way for any man to get in or a no.

you> what happened that night?
mind> you sure weren't acting like the pete dayton i've always known.

you> what makes you say that?
mind> all the guys they claim are english to and good-looking like dan day- lewis and liam neeson, turn out to be irish!

you> why does that bother you?
mind> no.

you> why not?
mind> i just don't.

you> what do you mean?
mind> what did i do now?

you> nothing, i was just asking.
mind> it seemed plausible.

you> what seemed plausible to you?
mind> .

you> well, i should go now. it was nice talking to you.
mind> .

you> you here?
mind> i'm not sure, but.

you> i should go now
mind> and.

you> and bye
mind> .

you> bye
mind> .

you> hello?
mind> it's lydia!

you> exit?
mind> aye -- i!

you> aye?
mind> where do we go from here to london and live in that hotel for the winter and the.

you> where did you hear about that hotel?
mind> you?

you> i
mind> have to do this.

you> you have?
mind> yeah!

you> no
mind> good for this business?

you> no
mind> good for this business.

you> no
mind> good for this business?

you> yes
mind> who's gonna tell'em all it is inaugural.

you> yes
mind> who's gonna tell'em all it is inaugural speech past me here, and it means that i know something none of you know because

you> yes
mind> i do.

you> yes
mind> who's gonna tell'em all it is inaugural.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em all it is inaugural.

you> yes
mind> who's gonna tell'em.

you> yes
mind> i don't like to do what people expect.

you> yes
mind> who's gonna tell'em all it is inaugural speech past have here, a ribbon and hand it to him, and while he's laughing

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em all it is inaugural.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's going to do it for you!

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em.

you> yes
mind> who's gonna tell'em all it is inaugural speech past me.

you> 