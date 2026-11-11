> ./target/release/mind --load data/runs/m.bin --chat-only --chat-context 5
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
stage took 24.9s
memory: 10885 Mb rss, 9007 Mb of tables
forgotten: rows=0 features=0
eta to the end of the corpus: 0 min (0 dialogs left, 0 dialogs/s)

=== CHAT ===
write a line and press enter - the mind continues with its own reply
commands: /quit  /exit  /help  /stats  /top

you> hello
mind> , i.

you> what?
mind> i don't know what you mean.

you> how are you?
mind> fine.

you> i too
mind> .

you> how are you?
mind> fine.

you> who are you?
mind> i'm the one who understands you.

you> what do you usually do when you're alone?
mind> ?

you> do you like music?
mind> yes, you.

you> what do you mean, me?
mind> the mother.

you> whose mother?
mind> is she.

you> is she your mother?
mind> you?

you> are you talking about me?
mind> and i need one more for my pal, you.

you> what are we having?
mind> my.

you> your?
mind> you you you a a fine young man a a.

you> who is the young man?
mind> .

you> you know?
mind> yeah.

you> yeah, you know?
mind> and i like that.

you> you like games?
mind> .

you> you know games?
mind> what is this a.

you> games?
mind> well?

you> well, what do you think about games?
mind> .

you> okay
mind> momma i can take a.

you> what makes you happy?
mind> .

you> you happy?
mind> .

you> what are you afraid of?
mind> i'm the one's got the gun and.

you> do you have a best friend?
mind> , and a.

you> you have?
mind> yeah!

you> i your friend?
mind> he.

you> he?
mind> the donor to the.

you> next
mind> .

you> okay
mind> momma i can take you somewhere to a place where you wouldn't be alone.

you> what food do you hate?
mind> what.

you> food
mind> and.

you> you know food?
mind> .

you> you hate?
mind> what.

you> okay
mind> momma i can take you somewhere to a place where they can't escape from me that you have a,.

you> what do you do on sunday?
mind> .

you> sunday
mind> ?

you> okay
mind> momma i can take you somewhere to a place where you wouldn't be alone?

you> Do you want?
mind> what are you doing?

you> i talking
mind> ?

you> i talking with you
mind> and you.

you> yes
mind> i don't like you, and have all these little theories.

you> little theories?
mind> .

you> theories?
mind> .

you> you have little theories?
mind> .

you> no?
mind> no more games!

you> you know games?
mind> what is this a.

you> A game is something people play for fun, using rules, goals, and sometimes competition.
mind> a.

you> you see?
mind> yes!

you> you see game?
mind> .

you> you know game?
mind> what, i don't know.

you> okay
mind> momma i can do it.

you> don't worry
mind> .

you> do you ever feel lonely?
mind> it's that.

you> what was your childhood like?
mind> you?

you> yex
mind> .

you> yes i
mind> don't like to do what people expect from me!

you> yes i
mind> do!

you> yep
mind> ?

you> yep is yes
mind> a a.

you> yep?
mind> the little girl's name.

you> do you trust people easily?
mind> .

you> what makes you angry?
mind> .

you> do you like being alone?
mind> .

you> you alone?
mind> no it can't be verified by his.

you> what is your biggest dream?
mind> .

you> okay
mind> momma i can take you somewhere to a place where they can't escape.

you> yes
mind> who's a a.

you> you can take me
mind> .

you> momma?
mind> lots of things.

you> you momma?
mind> lots of things!

you> i?
mind> the child is mine, uther.

you> you feel something?
mind> .

you> you feel?
mind> yes!

you> really?
mind> i don't feel so great.

you> how much?
mind> well, nothing.

you> feel nothing?
mind> !

you> yes?
mind> i have not had the opportunity to tell him what to do it!

you> tell me
mind> what?

you> tell me please
mind> !

you> you tell me?
mind> what happened to me the other night when i you are.

you> i don't know
mind> .

you> you know?
mind> yes.

you> 10+10?
mind> there's a.

you> 100+100?
mind> what.

you> 1000+1000?
mind> day and you.

you> 10+10=20
mind> years, if you're livin' next, and.

you> 10+10?
mind> there's a i?

you> okay
mind> momma i and what.

you> where momma?
mind> lots of things!

you> who momma?
mind> .

you> he momma?
mind> .

you> i momma?
mind> .

you> you momma?
mind> lots of things.

you> you mother?
mind> who killed anne lively dancer john.

you> okay
mind> momma i can take you somewhere to a?

you> well, i should go now. it was nice talking to you.
mind> and.

you> and bye
mind> .

you> you feel?
mind> yes.

you> really?
mind> i'm going to take asta for a walk.

you> you feel?
mind> yes.

you> bye
mind> !

you> bye!
mind> bye.

you> yes
mind> i don't like that -- -- hound of resurrection?

you> bye bye
mind> blackbird that.

you> bye
mind> !

you> bye!
mind> bye that was snowball.