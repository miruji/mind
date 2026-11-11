> ./target/release/mind --load data/runs/m.bin --chat-only --chat-context 128
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
stage took 25.1s
memory: 10885 Mb rss, 9007 Mb of tables
forgotten: rows=0 features=0
eta to the end of the corpus: 0 min (0 dialogs left, 0 dialogs/s)

=== CHAT ===
write a line and press enter - the mind continues with its own reply
commands: /quit  /exit  /help  /stats  /top

you>hello
mind> .

you> you okay?
mind> i'm, uh, in and out, no passport problems.

you> okay
mind> wait till the islands to the mainland.

you> yes
mind> who's gonna tell'em all it?

you> you know?
mind> yeah.

you> okay
mind> momma!

you> yes 
mind> i don't like it here!

you> what you like?
mind> .

you> you don't like?
mind> well it's.

you> you feel?
mind> it reminded me of my dog callie still.

you> you have dog?
mind> .

you> you don't have dog
mind> !

you> yep
mind> !

you> yeah
mind> so why are!

you> you like music?
mind> yes!

you> you like games?
mind> what.

you> you like sun?
mind> ?

you> you like me?
mind> .

you> you like you?
mind> .

you> you know?
mind> yeah.

you> you like words?
mind> ?

you> you like songs?
mind> .

you> you like music?
mind> yes.

you> next?
mind> yes.

you> you like cats?
mind> ?

you> you like dogs?
mind> the.

you> you know dogs?
mind> the!

you> you like cars?
mind> a little.

you> you like children?
mind> !

you> you like people?
mind> .

you> you know people?
mind> !

you> yes?
mind> i have to see you.

you> okay. next?
mind> a pneumoencephelogram!

you> okay
mind> momma i!

you> you like rain?
mind> .

you> you like summer?
mind> .

you> you like cooking?
mind> .

you> you like music?
mind> yes.

you> you like movies?
mind> very much?

you> yes?
mind> and the tiny hairs on your arm.

you> you like movies?
mind> very much?

you> how much
mind> .

you> how much?
mind> i think two million dollars?

you> okay next?
mind> a pneumoencephelogram, i would think, pin down that lesion and.

you> you know music?
mind> a!

you> you like music?
mind> yes.

you> you see music?
mind> .

you> you feel music?
mind> .

you> you like?
mind> me like a!

you> you like music
mind> !

you> okay. you like music
mind> .

you> i like music?
mind> .

you> you know?
mind> yeah i?

you> i like? 
mind> forget?

you> music
mind> !

you> i like music?
mind> a!

you> you like?
mind> me!

you> yes
mind> who's gonna tell'em.

you> what music?
mind> there.

you> there music?
mind> .

you> you music?
mind> !

you> how much you music?
mind> a!

you> a
mind> !

you> a!
mind> !

you> tell me
mind> .

you> no?
mind> no!

you> why
mind> because it's not my!

you> why
mind> because it's not my.

you> music?
mind> yeah!

you> not you?
mind> no.

you> my?
mind> .

you> not your?
mind> ?

you> 