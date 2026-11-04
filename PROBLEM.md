# Track 1 problem statement: Isogenies on Oasis

Qiskit Fall Fest 2026 × BITS Pilani, Pilani Campus. Problem statements introduced on 27 October 2026; deadline 4 November 2026.

Copied from the hackathon web page. Only the page menus were removed; the sub- and superscripts lost in the copy were restored (y², x³, i², A², E_A, r_b, 2^L, 2^256, 2^254).

---

## Isogenies on Oasis

- How Ankan sees campus
- 1 · The password
- 2 · The phone on the Clock Tower
- 3 · The Oasis map
- 4 · The Djinns
- Details

With Oasis coming up in a few days, our beloved physics assoc members, Shashank, Gaurav, Shreyash, Aditya and Soham, were looking forward to having some fun. But suddenly, they get a call from Ankan Pal telling them to meet him in his chambers for a “fun game”. The members, knowing that the “fun” was not going to be very fun, hesitantly went to meet him.

Upon meeting him, they see him excited and a bit concerned as well. “We have a problem at hand, and if you guys want a Domino’s treat from me, you are going to need to help me a bit.”

Not particularly interested in Domino’s, but liking Ankan Pal and being curious students, our protagonists decided to hear it out.

Ankan explains. He suspects that with Oasis coming up, Djinns have started showing up on campus. He has MFA set up on his Google account, and last night one of them wiped the password clean out of his head and ran off with his phone. He has a crypto test to make and needs to log in to do it. All he remembers is that he took inspiration for the password from the routes between the Maths department and ANC. And when he used Find My Device, it showed his phone on top of the Clock Tower, which is exactly where you’d leave a phone if you could fly.

Our protagonists decide they want more than a Domino’s treat, and that their reward should be proportional to how hard it is to get to the top of the Clock Tower. Ankan, a bit too quickly, agrees.

Also, it turns out that with Djinns wandering around and so many people running so many events all over campus during Oasis, Ankan Pal was tasked with making a map, and he is too lazy to do it himself. So, multiple birds with one step.

## How Ankan sees campus

Ankan, of course, doesn’t see campus the way normal people do. He goes to the board.

“Every spot on campus is a curve,” he says.

```text
E_A : y² = x³ + Ax² + x,  with A ∈ 𝔽 = 𝔽_p(i),  p ≡ 3 (mod 4),  i² = −1
```

“From every spot there are three roads, which are 2-isogenies. You came in on one, so there are two ways forward, 0 and 1. To know which is which, take the two 2-torsion points other than (0, 0) and sort their x-coordinates by key(u + v·i) = u + p·v to get r_0 and r_1. Bit b means you take the road through r_b. You land on a curve you can write as E_A′ or E_−A′ (the two forms where (0, 0) is the road back to where you came from, i.e., it generates the kernel of the dual isogeny); take whichever has the smaller key. A password is just a string of bits, so it’s a route, and where you end up is its j-invariant. Call that H(route), and call the length of the route L.” (Every spot has A² ≠ 4, so the curve is never singular.)

| | |
|---|---|
| p | 1019 |
| Maths department | `A0 = 977 + 297i` |
| Route length | `L = 12` |
| ANC | `j* = 797 + 921i` |

“Some checks so you know you’re doing it right,” he adds:

| Route | Where you are after it | j |
|---|---|---|
| 0 | 747 + 288i | 98 + 85i |
| 1 | 79 + 218i | 138 |
| 01 | — | 967 + 249i |
| 000000000000 | — | 348 + 660i |
| 111111111111 | — | 34 |
| 011010010110 | — | 98 + 934i |
| 101100111000 | — | 119 |

## Part 1

### The password

Ankan’s password is one of the 12-step routes from the Maths department to ANC, but he doesn’t remember which one, so our protagonists need to find all of them. And since he’s going to sit there typing them in one by one, he wants to be sure the list is complete, and he wants them to prove it to him with a quantum procedure and tell him how confident they are. He also wants to know how many queries it took (one query is one use of the one-step circuit), compared with what a classical search and the best possible quantum search would need, in terms of N = 2^L and the number of routes k.

Aditya points out that it’s only 4096 routes and they could just walk all of them. Ankan laughs a bit. That works on this campus, sure, but he only drew them a small one. The real campus, the one his test is about, has p around 2^256 and something like 2^254 spots, so they have to do it the way they’d have to do it there.

So the rules are as follows:

- They’re allowed to walk every road once before they know where they’re going, which is at most two moves per spot, and turn that into a quantum circuit that takes one step coherently. This survey can’t depend on A0, L or j\*.
- After that, no planning routes on paper. That means no counting paths, no going backwards from ANC, no meet-in-the-middle, and no listing routes. The only classical thing allowed is walking a route they measured, once, to check it.
- They can’t build an oracle out of evaluations of H either. The “are we at ANC?” check has to look at the spot itself, inside the circuit. On the small campus it can come from the survey, but on the real one it has to be arithmetic, which is what Part 3 is for.
- And if it turns out no route actually gets to ANC, their code should say so instead of dying.

## Part 2

### The phone on the Clock Tower

On Ankan’s campus, the top of the Clock Tower is the set of all curves whose j-invariant is in 𝔽p. On the small campus almost a third of routes end up there, so it’s not that impressive. On the real campus it’s more like one spot in √p, which is where it gets hard.

Since our protagonists wanted their reward to be proportional to how hard it is to get up there, Ankan says fine: tell me how hard it is. Concretely:

- What does it cost a quantum computer, as a power of p, to find a route of length Θ(log p) that ends on top of the Clock Tower?
- They have to prove the upper bound, saying which theorems they borrowed.
- They have to prove a matching lower bound in a model they define properly, and explain why nobody can prove one for the real arithmetic.

He also wants data on how often routes end up there as p grows, with the exponent fitted. For the data they’re allowed to walk random routes classically, since that’s measuring the campus, not looking for the phone. And he wants to see them actually do it on a campus where at most 1 in 32 routes gets there, with success plotted against Grover iterations, next to the theory curve and next to just walking around classically.

## Part 3

### The Oasis map

Now for the map Ankan was too lazy to make. The catch is that the Djinns don’t like being written down, and anything that looks like a table of roads just goes missing overnight. Plus, during Oasis the campus is huge and nobody is going to write down every road anyway. So the map can’t be a table this time. It has to work out every road by itself.

For p = 1019, our protagonists have to build one step as a reversible circuit using only arithmetic in 𝔽. That means doing the square root, the key comparison and the sign choice, and cleaning up every ancilla afterwards. They check it on every spot and both bits. Then they tell Ankan what the same circuit costs at p ≈ 2^256, including checking whether you’re on the Clock Tower, and what getting his phone back (Part 2) would cost there.

## Part 4

### The Djinns

There’s one more thing bothering Ankan. The Djinns aren’t done. The night before the fest, they’re going to go around campus and swap every signboard according to some secret random shuffle that only they know. The roads stay the same; only the names change. After that, all anyone has is:

- the name of the spot they start from;
- a Djinn who, if asked, tells them the name behind road 0 or road 1;
- a volunteer at every venue who just says yes or no to “is this the Clock Tower?” and “is this ANC?”.

So he wants to know what exactly their speed-ups were depending on. What still works after the Djinns are done, what doesn’t, and why? Which classical tricks still work? And can they prove the Part 2 lower bound in a world run by Djinns?

Shashank, Gaurav, Shreyash, Aditya and Soham look at each other, then at the Clock Tower, and decide the Domino’s had better be a large.

## Details

### Rules

- All circuits are built and run in Qiskit, on any simulator.
- Report resources after transpiling to a basis you state. For every run, include the number of queries, the qubit count, the two-qubit gate count and the depth. For Part 3, also include the Toffoli (or T) count.
- One command reproduces every result and takes p, A0, L and j\* as arguments.
- AI tools are explicitly allowed throughout the build. Understanding still counts for a lot, though, and it will be tested in the viva and demo: you should be able to explain and modify anything you submit.

### Teams

Teams of 1 to 3 people.

### Submission

A repository containing the code and a report of at most 6 pages.

### Evaluation

Each participant has 15 minutes with the judges, without AI tools. Any part of the submission you cannot explain or modify is treated as not submitted.

| Criterion | Points |
|---|---|
| Code | 35 |
| Viva | 45 |
| Demo | 20 |

Submission for the hackathon will open soon. We'll share the link here and in the WhatsApp group.
